"""Tests for the user-steer chat endpoint (Slice C).

The endpoint must be ownership-checked, mark the steer as a user steer in the
durable pending-node store, and never start a second completion turn. It also
has to behave sanely at the turn boundary: a task that ended while the request
was in flight must return a fallback instead of stranding a pending node, and a
retried request must not create or supersede a second steer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.routers.chats import SteerChatForm, steer_chat_by_id


class FakeChats:
    def __init__(self, messages: dict | None = None, meta: dict | None = None):
        self.chat = SimpleNamespace(
            id='chat-1',
            user_id='user-1',
            chat={'history': {'currentId': 'tip', 'messages': messages or {}}},
            meta=meta or {},
        )

    async def get_chat_by_id_and_user_id(self, id, user_id, db=None):
        if id == 'chat-1' and user_id == 'user-1':
            return self.chat
        return None


def make_request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))


def make_user(user_id='user-1'):
    return SimpleNamespace(id=user_id, role='user')


def steer_node(steer_id: str, status: str = 'pending'):
    return {
        'id': steer_id,
        'role': 'user',
        'content': 'inspect the failing test',
        'childrenIds': [],
        'meta': {'internal': True, 'type': 'steer', 'source': 'user', 'status': status},
    }


@pytest.mark.asyncio
async def test_steer_endpoint_queues_user_steer_without_starting_a_turn(monkeypatch):
    import open_webui.routers.chats as chats_router

    calls: list[tuple] = []
    active_checks: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        active_checks.append((redis, chat_id))
        return True

    async def fake_enqueue_steer(chat_id, user_id, content, **kwargs):
        calls.append((chat_id, user_id, content, kwargs))
        return 'steer-1'

    async def fake_mark_superseded(*args, **kwargs):
        raise AssertionError('a live task must not supersede the fresh steer')

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)
    monkeypatch.setattr(chats_router, 'mark_steer_superseded', fake_mark_superseded)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='  inspect the failing test  ', model_id='model-1'),
        user=make_user(),
        db=None,
    )

    assert response.queued is True
    assert response.id == 'steer-1'
    assert active_checks == [(None, 'chat-1'), (None, 'chat-1')]
    # Exactly one durable enqueue; no task id, completion, or second turn.
    assert calls == [
        (
            'chat-1',
            'user-1',
            'inspect the failing test',
            {'source': 'user', 'model_id': 'model-1', 'message_id': None},
        )
    ]


@pytest.mark.asyncio
async def test_steer_endpoint_targets_an_internal_subagent_child_chat(monkeypatch):
    """A steer on a child chat must be selectable by that child's drain."""
    import open_webui.routers.chats as chats_router

    calls: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        return chat_id == 'chat-1'

    async def fake_enqueue_steer(chat_id, user_id, content, **kwargs):
        calls.append((chat_id, user_id, content, kwargs))
        return 'steer-child'

    async def fake_mark_superseded(*args, **kwargs):
        raise AssertionError('a live child must not supersede its fresh steer')

    child_meta = {'internal': True, 'type': 'subagent', 'parent_chat_id': 'parent-1', 'state': 'running'}
    monkeypatch.setattr(chats_router, 'Chats', FakeChats(meta=child_meta))
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)
    monkeypatch.setattr(chats_router, 'mark_steer_superseded', fake_mark_superseded)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='focus on the child', model_id='model-1'),
        user=make_user(),
        db=None,
    )

    assert response.queued is True
    assert response.id == 'steer-child'
    # child_id marks the record for the child's own boundary drain; source stays
    # 'user' and nothing starts a second turn.
    assert calls == [
        (
            'chat-1',
            'user-1',
            'focus on the child',
            {'source': 'user', 'child_id': 'chat-1', 'model_id': 'model-1', 'message_id': None},
        )
    ]


@pytest.mark.asyncio
async def test_steer_endpoint_falls_back_when_no_active_task(monkeypatch):
    import open_webui.routers.chats as chats_router

    enqueued: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        return False

    async def fake_enqueue_steer(*args, **kwargs):
        enqueued.append((args, kwargs))
        return 'steer-1'

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='too late'),
        user=make_user(),
        db=None,
    )

    assert response.queued is False
    assert response.reason == 'no_active_task'
    assert enqueued == []


@pytest.mark.asyncio
async def test_steer_endpoint_supersedes_when_the_task_ends_mid_request(monkeypatch):
    import open_webui.routers.chats as chats_router

    active = iter([True, False])
    superseded: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        return next(active)

    async def fake_enqueue_steer(*args, **kwargs):
        return 'steer-1'

    async def fake_mark_superseded(chat_id, user_id, steer_id):
        superseded.append((chat_id, user_id, steer_id))
        return [steer_id]

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)
    monkeypatch.setattr(chats_router, 'mark_steer_superseded', fake_mark_superseded)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='raced'),
        user=make_user(),
        db=None,
    )

    assert response.queued is False
    assert response.reason == 'no_active_task'
    assert superseded == [('chat-1', 'user-1', 'steer-1')]


@pytest.mark.asyncio
async def test_steer_endpoint_reuses_a_client_idempotency_key(monkeypatch):
    import open_webui.routers.chats as chats_router

    enqueued: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        return True

    async def fake_enqueue_steer(*args, **kwargs):
        enqueued.append((args, kwargs))
        return kwargs.get('message_id') or 'steer-1'

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='retry me', steer_id='stable-steer'),
        user=make_user(),
        db=None,
    )

    assert response.queued is True
    assert response.id == 'stable-steer'
    assert enqueued == [
        (
            ('chat-1', 'user-1', 'retry me'),
            {'source': 'user', 'model_id': None, 'message_id': 'stable-steer'},
        )
    ]


@pytest.mark.asyncio
async def test_steer_endpoint_retry_does_not_create_a_second_steer(monkeypatch):
    import open_webui.routers.chats as chats_router

    enqueued: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        return True

    async def fake_enqueue_steer(*args, **kwargs):
        enqueued.append((args, kwargs))
        return 'stable-steer'

    async def fake_mark_superseded(*args, **kwargs):
        raise AssertionError('an idempotent retry must not supersede the steer')

    monkeypatch.setattr(chats_router, 'Chats', FakeChats({'stable-steer': steer_node('stable-steer')}))
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)
    monkeypatch.setattr(chats_router, 'mark_steer_superseded', fake_mark_superseded)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='retry me', steer_id='stable-steer'),
        user=make_user(),
        db=None,
    )

    assert response.queued is True
    assert response.id == 'stable-steer'
    assert enqueued == []


@pytest.mark.asyncio
async def test_steer_endpoint_retry_of_superseded_steer_reports_no_active_task(monkeypatch):
    import open_webui.routers.chats as chats_router

    async def fake_has_active_tasks(redis, chat_id):
        return True

    async def fake_enqueue_steer(*args, **kwargs):
        raise AssertionError('a superseded idempotency key must not be re-enqueued')

    monkeypatch.setattr(
        chats_router,
        'Chats',
        FakeChats({'stable-steer': steer_node('stable-steer', status='superseded')}),
    )
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='retry me', steer_id='stable-steer'),
        user=make_user(),
        db=None,
    )

    assert response.queued is False
    assert response.reason == 'no_active_task'


@pytest.mark.asyncio
async def test_steer_endpoint_rejects_non_owner(monkeypatch):
    import open_webui.routers.chats as chats_router

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(
        chats_router,
        'has_active_tasks',
        lambda *args, **kwargs: pytest.fail('ownership must be checked before task state'),
    )

    with pytest.raises(HTTPException) as exc:
        await steer_chat_by_id(
            make_request(),
            'chat-1',
            SteerChatForm(content='steer'),
            user=make_user('other-user'),
            db=None,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_steer_endpoint_requires_content(monkeypatch):
    import open_webui.routers.chats as chats_router

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())

    with pytest.raises(HTTPException) as exc:
        await steer_chat_by_id(
            make_request(),
            'chat-1',
            SteerChatForm(content='   '),
            user=make_user(),
            db=None,
        )

    assert exc.value.status_code == 400
