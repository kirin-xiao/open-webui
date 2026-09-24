"""Shared fakes for the subagent backend tests (T3).

Not a test module: pytest only collects ``test_*.py``. Provides an in-memory
``Chats`` stand-in that reuses the real history-maintenance helper, a fake
``Config``, a fake task registry and a fake async DB session, so the delegation
paths in ``open_webui.utils.subagents`` can be exercised without a database or
Redis.

Keeping these here (rather than in each test) means the tests can call the real
``delegate`` / ``process_pending_internal_messages`` coroutines with faithful
patching, instead of re-implementing the behaviour under test.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from types import SimpleNamespace

from open_webui.models.chats import ChatModel, Chats


def user_data(**overrides) -> dict:
    """A minimal but complete ``UserModel`` payload."""
    data = {
        'id': 'user-1',
        'email': 'user@example.com',
        'name': 'Test User',
        'role': 'user',
        'last_active_at': 0,
        'updated_at': 0,
        'created_at': 0,
    }
    data.update(overrides)
    return data


class FakeChats:
    """In-memory equivalent of the ``Chats`` repository methods used by delegate."""

    def __init__(self):
        self.chats: dict[str, ChatModel] = {}
        self.messages: dict[str, dict] = {}

    async def insert_new_chat(self, id, user_id, form_data, db=None, *, internal_meta=None, timer_at=None):
        chat = copy.deepcopy(form_data.chat)
        history = chat.setdefault('history', {})
        messages = history.setdefault('messages', {})
        now = int(time.time())
        model = ChatModel(
            id=id,
            user_id=user_id,
            title=chat.get('title', 'New Chat'),
            chat=chat,
            created_at=now,
            updated_at=now,
            meta=internal_meta or {},
            current_message_id=Chats.get_current_message_id(chat),
        )
        self.chats[id] = model
        self.messages.setdefault(id, {})
        for message_id, message in messages.items():
            self.messages[id][message_id] = copy.deepcopy(message)
        return model

    async def get_chat_by_id(self, id, db=None):
        return self.chats.get(id)

    async def update_chat_meta_by_id(self, id, meta, db=None):
        chat = self.chats.get(id)
        if chat is None or not meta:
            return False
        chat.meta = {**(chat.meta or {}), **meta}
        return True

    async def upsert_message_to_chat_by_id_and_message_id(self, id, message_id, message, *, touch=True):
        chat = self.chats.get(id)
        if chat is None:
            return None
        # Reuse the production helper so parentId/childrenIds/currentId behave
        # exactly as a real save would.
        history = chat.chat.setdefault('history', {})
        Chats.upsert_message_to_history(history, message_id, message)
        chat.updated_at = int(time.time())
        self.messages.setdefault(id, {})[message_id] = copy.deepcopy(message)
        return chat

    async def update_chat_by_id(self, id, chat, db=None, *, touch=True):
        target = self.chats.get(id)
        if target is None:
            return None
        stored = target.chat or {}
        updated = {**stored, **chat}
        if 'history' in chat:
            # Mirror the production merge so a stale writer cannot drop messages.
            updated['history'] = Chats.merge_history(stored.get('history'), chat['history'])
        target.chat = updated
        if any(key in chat for key in ('history', 'messages', 'currentId', 'branchPointMessageId')):
            target.current_message_id = Chats.get_current_message_id(updated)
        if touch:
            target.updated_at = int(time.time())
        return target

    async def get_message_by_id_and_message_id(self, id, message_id):
        return self.messages.get(id, {}).get(message_id)

    async def get_chat_folder_id(self, id, user_id, db=None):
        chat = self.chats.get(id)
        if chat is None or chat.user_id != user_id:
            return None
        return getattr(chat, 'folder_id', None)


class FakeConfig:
    def __init__(self, values: dict | None = None):
        self.values = dict(values or {})

    async def get_many(self, *keys):
        return {key: self.values.get(key) for key in keys}

    async def get(self, key, default=None):
        return self.values.get(key, default)


class FakeTaskRegistry:
    """Replacement for ``tasks.create_task`` / ``tasks.has_active_tasks``.

    ``run=True`` creates a real asyncio task so foreground awaits resolve and
    background completions can be awaited by the test. ``run=False`` captures
    and closes the coroutine so a background call returns its handle without
    executing the child turn.
    """

    def __init__(self, *, run: bool = True, active: bool = False):
        self.run = run
        self.active = active
        self.created: list[tuple[str | None, object]] = []

    async def create_task(self, redis, coroutine, id=None, task_id=None):
        if not self.run:
            coroutine.close()
            return (task_id or 'captured', None)
        task = asyncio.ensure_future(coroutine)
        self.created.append((id, task))
        return (task_id or 'task', task)

    async def has_active_tasks(self, redis, chat_id):
        return self.active


class FakeResult:
    def __init__(self, chat):
        self._chat = chat

    def scalar_one_or_none(self):
        return self._chat


def _equality_filters(stmt) -> dict[str, object]:
    """Column-name -> bound value for the ``col == bind`` filters in a select.

    Only used to keep the fake session honest: it lets ``FakeDB`` evaluate the
    real ``Chat.id``/``Chat.user_id`` predicate instead of returning whatever
    row the test happens to hold.
    """
    from sqlalchemy.sql.elements import BinaryExpression, BindParameter

    filters: dict[str, object] = {}

    def walk(clause) -> None:
        if isinstance(clause, BinaryExpression):
            column = clause.left
            name = getattr(getattr(column, 'element', column), 'name', None) or getattr(column, 'name', None)
            if name and isinstance(clause.right, BindParameter):
                filters[name] = clause.right.value
        for child in clause.get_children():
            walk(child)

    if stmt.whereclause is not None:
        walk(stmt.whereclause)
    return filters


class FakeDB:
    """Fake async SQLAlchemy session for the direct ``select(Chat)`` paths.

    ``execute`` enforces the production ownership predicate (``Chat.id`` and
    ``Chat.user_id``) rather than blindly yielding the seeded row, so removing
    or breaking that filter fails the test.
    """

    def __init__(self, chat):
        self._chat = chat
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name='sqlite'))
        self.commits = 0

    async def execute(self, stmt):
        filters = _equality_filters(stmt)
        assert 'id' in filters and 'user_id' in filters, f'query lost its ownership predicate: {stmt}'
        chat = self._chat
        if chat is None or filters['id'] != chat.id or filters['user_id'] != chat.user_id:
            return FakeResult(None)
        return FakeResult(chat)

    async def commit(self):
        self.commits += 1


def fake_get_async_db(chat):
    """Build a ``get_async_db`` replacement that always yields ``chat``."""

    @contextlib.asynccontextmanager
    async def _cm():
        yield FakeDB(chat)

    return _cm


class FakeChatMessages:
    def __init__(self):
        self.upserted: list[dict] = []
        self.deleted: list[tuple[str, set]] = []

    async def upsert_message(self, message_id, chat_id, user_id, data):
        self.upserted.append({'message_id': message_id, 'chat_id': chat_id, 'user_id': user_id, 'data': data})

    async def delete_message_ids_by_chat_id(self, chat_id, message_ids):
        self.deleted.append((chat_id, set(message_ids)))


class FakeSio:
    def __init__(self):
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, event, data, room=None):
        self.emitted.append((event, data))


def make_app(handler, redis=None, models=None):
    return SimpleNamespace(
        state=SimpleNamespace(CHAT_COMPLETION_HANDLER=handler, redis=redis, MODELS=models),
    )


def make_source_request(app):
    return SimpleNamespace(app=app, state=SimpleNamespace(internal=False))


def make_chat(chat_id, *, user_id='user-1', meta=None, history=None):
    """A mutable stand-in for the ORM chat row returned by the fake session."""
    return SimpleNamespace(
        id=chat_id,
        user_id=user_id,
        chat={'history': history or {'currentId': None, 'messages': {}}},
        meta=meta or {},
        updated_at=0,
    )
