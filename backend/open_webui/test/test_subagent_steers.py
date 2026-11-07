"""Focused tests for the steering foundation (Slice 0).

Covers the durable steer record contract, stale completion supersession,
child chat lifecycle state, and the ownership-checked subagent listing
endpoint. The final section also exercises the Slice A boundary helpers against
Slice B's real store, so the cross-slice delivery contracts are pinned here.
These use the same in-memory ``Chats`` fake as ``test_subagents`` so the real
delegate / drain coroutines run without a database or Redis.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.models.chats import ChatModel, Chats
from open_webui.routers.chats import get_chat_subagents_by_id, steer_chat_by_id, SteerChatForm
from open_webui.test.subagent_test_utils import (
    FakeChatMessages,
    FakeChats,
    FakeConfig,
    FakeSio,
    FakeTaskRegistry,
    fake_get_async_db,
    make_app,
    make_source_request,
    user_data,
)
from open_webui.utils import subagents
from open_webui.utils.misc import (
    get_message_list,
    is_internal_steer_node,
    is_pending_internal_message,
    resolve_history_tip,
)


class Harness:
    """Wire the real ``subagents`` coroutines to in-memory fakes."""

    def __init__(self, monkeypatch, *, run=True, active=False, summary='Subagent finished the work.'):
        self.fake_chats = FakeChats()
        self.chat_messages = FakeChatMessages()
        self.registry = FakeTaskRegistry(run=run, active=active)
        self.completions: list[tuple] = []
        self.states_seen: list[str | None] = []
        self.summary = summary
        self._db_chat = None
        self._fail_with: Exception | None = None
        self.block_event: asyncio.Event | None = None

        async def handler(request, form_data, user=None):
            self.completions.append((request, form_data))
            child = self.fake_chats.chats.get(form_data['chat_id'])
            self.states_seen.append((child.meta or {}).get('state') if child else None)
            if self.block_event is not None:
                await self.block_event.wait()
            if self._fail_with is not None:
                raise self._fail_with
            await self.fake_chats.upsert_message_to_chat_by_id_and_message_id(
                form_data['chat_id'],
                form_data['id'],
                {'id': form_data['id'], 'content': self.summary, 'done': True},
            )

        app = make_app(handler)
        self.source = make_source_request(app)
        self.config = FakeConfig(
            {
                'subagents.max_iterations': 30,
                'subagents.max_output': 30_000,
                'subagents.system_prompt': '',
                'code_interpreter.engine': 'pyodide',
            }
        )

        async def _get_user_by_id(user_id):
            return SimpleNamespace(id=user_id, role='user')

        async def _get_auth_by_id(user_id, db=None):
            return SimpleNamespace(id=user_id, active=True, session_stamp='stamp')

        monkeypatch.setattr(subagents, 'Chats', self.fake_chats)
        monkeypatch.setattr(subagents, 'ChatMessages', self.chat_messages)
        monkeypatch.setattr(subagents, 'Config', self.config)
        monkeypatch.setattr(subagents, 'create_task', self.registry.create_task)
        monkeypatch.setattr(subagents, 'has_active_tasks', self.registry.has_active_tasks)
        monkeypatch.setattr(subagents, 'Users', SimpleNamespace(get_user_by_id=_get_user_by_id))
        monkeypatch.setattr(subagents, 'Auths', SimpleNamespace(get_auth_by_id=_get_auth_by_id))
        monkeypatch.setattr(subagents, 'create_token', lambda data, expires_delta=None: 'test-token')
        monkeypatch.setattr(subagents, 'get_async_db', lambda: fake_get_async_db(self._db_chat)())

        import open_webui.socket.main as socket_main

        self.sio = FakeSio()
        monkeypatch.setattr(socket_main, 'sio', self.sio)

    def seed_parent(self, chat_id='chat-1', *, user_id='user-1', history=None, meta=None):
        now = 1
        chat = ChatModel(
            id=chat_id,
            user_id=user_id,
            title='Parent',
            chat={'history': history or {'currentId': None, 'messages': {}}},
            created_at=now,
            updated_at=now,
            meta=meta or {},
            current_message_id=(history or {}).get('currentId'),
        )
        self.fake_chats.chats[chat_id] = chat
        self.fake_chats.messages.setdefault(chat_id, {})
        self._db_chat = chat
        return chat

    def seed_child(
        self,
        parent_chat_id='chat-1',
        *,
        child_id='child-1',
        user_id='user-1',
        state='completed',
        history=None,
    ):
        history = history or {
            'currentId': 'child-assistant',
            'messages': {
                'child-user': {
                    'id': 'child-user',
                    'parentId': None,
                    'childrenIds': ['child-assistant'],
                    'role': 'user',
                    'content': 'first task',
                    'timestamp': 1,
                },
                'child-assistant': {
                    'id': 'child-assistant',
                    'parentId': 'child-user',
                    'childrenIds': [],
                    'role': 'assistant',
                    'content': 'done',
                    'done': True,
                    'timestamp': 2,
                },
            },
        }
        child = ChatModel(
            id=child_id,
            user_id=user_id,
            title='Subagent: child',
            chat={'history': history},
            created_at=1,
            updated_at=1,
            meta={
                'internal': True,
                'type': 'subagent',
                'parent_chat_id': parent_chat_id,
                'state': state,
            },
            current_message_id=history.get('currentId'),
        )
        self.fake_chats.chats[child_id] = child
        return child

    def find_child(self, parent_chat_id='chat-1'):
        children = [
            (chat_id, chat)
            for chat_id, chat in self.fake_chats.chats.items()
            if (chat.meta or {}).get('type') == 'subagent' and (chat.meta or {}).get('parent_chat_id') == parent_chat_id
        ]
        assert len(children) == 1, f'expected exactly one child, got {children}'
        return children[0]


async def run_delegate(harness, *, parent_chat_id='chat-1', **overrides):
    kwargs = {
        'description': 'a task',
        'prompt': 'do the thing',
        'context': '',
        'background': False,
        'file_ids': None,
        'session_id': None,
        'model': None,
        'request': harness.source,
        'user_data': user_data(),
        'metadata': {'model_id': 'test-model', 'session_id': 'sess'},
        'parent_chat_id': parent_chat_id,
        'parent_message_id': 'parent-msg',
    }
    kwargs.update(overrides)
    return await subagents.delegate(**kwargs)


def _parent_history(*nodes, tip='tip'):
    messages = {
        tip: {
            'id': tip,
            'parentId': None,
            'childrenIds': [node['id'] for node in nodes],
            'role': 'assistant',
            'content': 'answer',
            'done': True,
            'timestamp': 10,
        }
    }
    for node in nodes:
        messages[node['id']] = copy.deepcopy(node)
    return {'currentId': tip, 'messages': messages}


def _pending_result(message_id='pending-1', *, child_id='child-1', status='pending', parent_id='tip'):
    meta = {
        'internal': True,
        'type': 'subagent',
        'source': 'subagent',
        'childID': child_id,
        'subagent_chat_id': child_id,
        'state': 'completed',
        'status': status,
    }
    return {
        'id': message_id,
        'parentId': parent_id,
        'childrenIds': [],
        'role': 'user',
        'content': f'<subagent sessionID="{child_id}" state="completed">\nold\n</subagent>',
        'model': 'test-model',
        'meta': meta,
        'timestamp': 5,
    }


def _steer_message(message_id, *, status='pending', child_id=None, parent_id='tip'):
    meta = {'internal': True, 'type': 'steer', 'source': 'user', 'status': status}
    if child_id:
        meta['childID'] = child_id
    return {
        'id': message_id,
        'parentId': parent_id,
        'childrenIds': [],
        'role': 'user',
        'content': f'steer {message_id}',
        'meta': meta,
        'timestamp': 5,
    }


def _completion_steer(
    message_id='notice-1',
    *,
    child_id='child-1',
    status='pending',
    generation_id='gen-old',
    state='completed',
    body='old answer',
    parent_id='tip',
):
    meta = {
        'internal': True,
        'type': 'steer',
        'source': 'subagent',
        'status': status,
        'childID': child_id,
        'generation_id': generation_id,
        'state': state,
    }
    return {
        'id': message_id,
        'parentId': parent_id,
        'childrenIds': [],
        'role': 'user',
        'content': f'<subagent sessionID="{child_id}" state="{state}">\n{body}\n</subagent>',
        'model': 'test-model',
        'meta': meta,
        'timestamp': 5,
    }


def _parent_completions(harness):
    """Completion handler calls for the parent chat (not the child's own run)."""
    return [call for call in harness.completions if call[1].get('chat_id') == 'chat-1']


def _completion_notices(messages, *, pending_only=True):
    return [
        message
        for message in messages.values()
        if (message.get('meta') or {}).get('type') == 'steer'
        and (message.get('meta') or {}).get('source') == 'subagent'
        and (not pending_only or (message.get('meta') or {}).get('status') in (None, 'pending'))
    ]


def _running_child_history(child_id='child-1'):
    return {
        'currentId': 'child-assistant',
        'messages': {
            'child-user': {
                'id': 'child-user',
                'parentId': None,
                'childrenIds': ['child-assistant'],
                'role': 'user',
                'content': 'original task',
                'timestamp': 1,
            },
            'child-assistant': {
                'id': 'child-assistant',
                'parentId': 'child-user',
                'childrenIds': [],
                'role': 'assistant',
                'content': '',
                'done': False,
                'timestamp': 2,
            },
        },
    }


async def _task_started(harness):
    for _ in range(50):
        if harness.states_seen:
            return
        await asyncio.sleep(0)
    raise AssertionError('child task never started')


async def _finish_registered_tasks(harness):
    for _ in range(20):
        pending = [task for _, task in harness.registry.created if task is not None and not task.done()]
        if not pending:
            await asyncio.sleep(0)
            pending = [task for _, task in harness.registry.created if task is not None and not task.done()]
            if not pending:
                return
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Steer record contract
# ---------------------------------------------------------------------------


def test_merge_history_keeps_steers_out_of_branch_children():
    """A steer keeps its anchor but never becomes a branch sibling.

    `update_chat_by_id` runs `merge_history` twice, so the guard must hold on a
    follow-up merge as well. Normal messages, legacy subagent notices, and timer
    nodes must still be linked exactly as before.
    """
    existing = _parent_history()
    next_user = {
        'id': 'next-user',
        'parentId': 'tip',
        'childrenIds': [],
        'role': 'user',
        'content': 'next question',
        'timestamp': 11,
    }
    timer = {
        'id': 'timer-1',
        'parentId': 'tip',
        'childrenIds': [],
        'role': 'user',
        'content': 'timer fired',
        'meta': {'internal': True, 'type': 'timer'},
        'timestamp': 12,
    }
    incoming = {
        'messages': {
            'steer-1': _steer_message('steer-1'),
            'next-user': next_user,
            'legacy-result': _pending_result('legacy-result'),
            'timer-1': timer,
        }
    }

    merged = Chats.merge_history(existing, incoming)
    children = merged['messages']['tip']['childrenIds']

    assert 'steer-1' not in children
    assert 'next-user' in children
    assert 'legacy-result' in children
    assert 'timer-1' in children
    # The steer stays readable and keeps its anchor for drain/update.
    assert merged['messages']['steer-1']['parentId'] == 'tip'
    assert merged['messages']['steer-1']['content'] == 'steer steer-1'
    assert merged['currentId'] == 'tip'

    # Second pass mirrors `update_chat_by_id`'s final normalisation merge.
    again = Chats.merge_history(merged, {})
    assert 'steer-1' not in again['messages']['tip']['childrenIds']
    assert 'steer-1' in again['messages']
    assert again['currentId'] == 'tip'


def _repair_history(history):
    chat = {'history': copy.deepcopy(history)}
    changed = Chats._repair_chat_current_id(chat)
    return chat['history'], changed


def test_repair_keeps_boundary_steer_off_branch_and_current_id():
    """A delivered boundary steer is not a branch child and never the tip."""
    history = _parent_history(_steer_message('steer-1', status='delivered'))
    history['currentId'] = 'steer-1'
    repaired, changed = _repair_history(history)

    assert changed is True
    assert repaired['currentId'] == 'tip'
    assert 'steer-1' not in repaired['messages']['tip']['childrenIds']
    assert is_internal_steer_node(repaired['messages']['steer-1']) is True
    # The visible chain never contains the non-promoted steer.
    assert [message['id'] for message in get_message_list(repaired['messages'], repaired['currentId'])] == ['tip']


def test_repair_keeps_pending_steer_off_branch_and_current_id():
    history = _parent_history(_steer_message('steer-1', status='pending'))
    history['currentId'] = 'steer-1'
    repaired, _ = _repair_history(history)

    assert repaired['currentId'] == 'tip'
    assert 'steer-1' not in repaired['messages']['tip']['childrenIds']
    assert [message['id'] for message in get_message_list(repaired['messages'], repaired['currentId'])] == ['tip']


def test_get_message_list_skips_non_promoted_steer_links():
    """A stale parent link cannot replay a boundary steer as a user message."""
    messages = {
        'tip': {
            'id': 'tip',
            'parentId': None,
            'childrenIds': ['steer-1'],
            'role': 'assistant',
            'content': 'answer',
            'timestamp': 10,
        },
        'steer-1': _steer_message('steer-1', status='delivered', parent_id='tip'),
        'next-user': {
            'id': 'next-user',
            'parentId': 'steer-1',
            'childrenIds': [],
            'role': 'user',
            'content': 'next question',
            'timestamp': 11,
        },
    }

    assert [message['id'] for message in get_message_list(messages, 'next-user')] == ['tip', 'next-user']


def test_repair_promoted_steer_keeps_its_assistant_turn_reachable():
    """A promoted steer is a real visible user turn with an assistant child."""
    history = {
        'currentId': 'steer-1',
        'messages': {
            'tip': {
                'id': 'tip',
                'parentId': None,
                'childrenIds': ['steer-1'],
                'role': 'assistant',
                'content': 'answer',
                'done': True,
                'timestamp': 10,
            },
            'steer-1': _steer_message('steer-1', status='delivered', parent_id='tip')
            | {'childrenIds': ['assistant-2']},
            'assistant-2': {
                'id': 'assistant-2',
                'parentId': 'steer-1',
                'childrenIds': [],
                'role': 'assistant',
                'content': 'steered answer',
                'done': True,
                'timestamp': 11,
            },
        },
    }

    repaired, _ = _repair_history(history)

    # A steer itself is still never a branch child, but its turn stays on the
    # currentId chain so the assistant response is visible.
    assert repaired['currentId'] == 'assistant-2'
    assert 'steer-1' not in repaired['messages']['tip']['childrenIds']
    chain = [message['id'] for message in get_message_list(repaired['messages'], repaired['currentId'])]
    assert chain == ['tip', 'steer-1', 'assistant-2']


def test_repair_trusts_the_promoted_marker_without_an_assistant_child():
    """The promote path marks its node, so detection cannot be fooled by a
    boundary steer that happens to have a child."""
    history = {
        'currentId': 'next-user',
        'messages': {
            'tip': {
                'id': 'tip',
                'parentId': None,
                'childrenIds': [],
                'role': 'assistant',
                'content': 'answer',
                'done': True,
                'timestamp': 10,
            },
            'steer-1': {
                'id': 'steer-1',
                'parentId': 'tip',
                'childrenIds': ['next-user'],
                'role': 'user',
                'content': 'promoted steer',
                'meta': {
                    'internal': True,
                    'type': 'steer',
                    'source': 'user',
                    'status': 'delivered',
                    'promoted': True,
                },
                'timestamp': 10,
            },
            'next-user': {
                'id': 'next-user',
                'parentId': 'steer-1',
                'childrenIds': [],
                'role': 'user',
                'content': 'next question',
                'timestamp': 11,
            },
        },
    }

    repaired, _ = _repair_history(history)

    # The promoted node is a real turn: its child stays anchored on it and the
    # steer still resolves as the visible user message for that turn.
    assert repaired['messages']['next-user']['parentId'] == 'steer-1'
    assert repaired['currentId'] == 'next-user'
    chain = [message['id'] for message in get_message_list(repaired['messages'], repaired['currentId'])]
    assert chain == ['tip', 'steer-1', 'next-user']


def test_current_message_id_is_steer_detects_a_stale_column():
    steer = _steer_message('steer-1', status='delivered')
    model = ChatModel(
        id='chat-1',
        user_id='user-1',
        title='Chat',
        chat={'history': {'currentId': 'tip', 'messages': {'steer-1': steer}}},
        created_at=1,
        updated_at=1,
        current_message_id='steer-1',
    )

    assert Chats._current_message_id_is_steer(model) is True
    model.current_message_id = 'tip'
    assert Chats._current_message_id_is_steer(model) is False


def test_repair_reanchors_polluted_next_turn_and_moves_current_id_off_steer():
    """A steer currentId left by an old save self-heals; the next turn survives."""
    history = {
        'currentId': 'steer-1',
        'messages': {
            'tip': {
                'id': 'tip',
                'parentId': None,
                'childrenIds': ['steer-1'],
                'role': 'assistant',
                'content': 'answer',
                'done': True,
                'timestamp': 10,
            },
            'steer-1': _steer_message('steer-1', status='delivered', parent_id='tip') | {'childrenIds': ['next-user']},
            'next-user': {
                'id': 'next-user',
                'parentId': 'steer-1',
                'childrenIds': [],
                'role': 'user',
                'content': 'next question',
                'timestamp': 11,
            },
        },
    }

    repaired, changed = _repair_history(history)

    assert changed is True
    assert repaired['messages']['next-user']['parentId'] == 'tip'
    assert 'next-user' in repaired['messages']['tip']['childrenIds']
    assert 'steer-1' not in repaired['messages']['tip']['childrenIds']
    # The repaired tip is the re-anchored user turn, never the steer.
    assert repaired['currentId'] == 'next-user'
    chain = [message['id'] for message in get_message_list(repaired['messages'], repaired['currentId'])]
    assert 'steer-1' not in chain
    assert chain == ['tip', 'next-user']


def test_repair_reanchors_past_a_chain_of_steers():
    history = {
        'currentId': 'steer-1',
        'messages': {
            'tip': {
                'id': 'tip',
                'parentId': None,
                'childrenIds': ['steer-1'],
                'role': 'assistant',
                'content': 'answer',
                'done': True,
                'timestamp': 10,
            },
            'steer-1': _steer_message('steer-1', status='delivered') | {'childrenIds': ['steer-2']},
            'steer-2': _steer_message('steer-2', status='delivered') | {'childrenIds': ['next-user']},
            'next-user': {
                'id': 'next-user',
                'parentId': 'steer-2',
                'childrenIds': [],
                'role': 'user',
                'content': 'next question',
                'timestamp': 11,
            },
        },
    }
    history['messages']['steer-2']['parentId'] = 'steer-1'

    repaired, _ = _repair_history(history)

    assert repaired['messages']['next-user']['parentId'] == 'tip'
    assert repaired['currentId'] == 'next-user'
    chain = [message['id'] for message in get_message_list(repaired['messages'], repaired['currentId'])]
    assert chain == ['tip', 'next-user']


def test_repair_leaves_healthy_history_and_internal_result_nodes_alone():
    timer = {
        'id': 'timer-1',
        'parentId': 'tip',
        'childrenIds': [],
        'role': 'user',
        'content': 'timer fired',
        'meta': {'internal': True, 'type': 'timer'},
        'timestamp': 11,
    }
    legacy = _pending_result('legacy-result')
    history = _parent_history(timer, legacy)
    history['currentId'] = 'legacy-result'

    repaired, changed = _repair_history(history)

    assert changed is False
    assert repaired['currentId'] == 'legacy-result'
    assert set(repaired['messages']['tip']['childrenIds']) == {'timer-1', 'legacy-result'}


@pytest.mark.asyncio
async def test_steer_enqueue_drain_deliver_supersede_roundtrip(monkeypatch):
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(history=_parent_history())

    steer_id = await subagents.enqueue_steer('chat-1', 'user-1', 'focus on tests', source='user')
    assert steer_id
    message = parent.chat['history']['messages'][steer_id]
    assert message['meta']['type'] == 'steer'
    assert message['meta']['status'] == 'pending'
    assert message['meta']['source'] == 'user'
    assert message['parentId'] == 'tip'
    assert message['childrenIds'] == []
    # The pending steer is in the history tree but not a branch child or the tip.
    assert steer_id not in parent.chat['history']['messages']['tip']['childrenIds']
    assert parent.chat['history']['currentId'] == 'tip'
    assert is_pending_internal_message(message) is True

    drained = await subagents.drain_pending_steers('chat-1', 'user-1')
    assert [item['id'] for item in drained] == [steer_id]
    assert [item['content'] for item in drained] == ['focus on tests']

    delivered = await subagents.mark_steer_delivered('chat-1', 'user-1', steer_id)
    assert delivered == [steer_id]
    assert parent.chat['history']['messages'][steer_id]['meta']['status'] == 'delivered'
    assert await subagents.drain_pending_steers('chat-1', 'user-1') == []

    child_steer_id = await subagents.enqueue_steer(
        'chat-1',
        'user-1',
        'narrow the scope',
        source='parent',
        child_id='child-1',
        generation_id='gen-1',
    )
    assert child_steer_id
    child_steer = parent.chat['history']['messages'][child_steer_id]
    assert child_steer['meta']['childID'] == 'child-1'
    assert child_steer['meta']['generation_id'] == 'gen-1'
    # Parent-targeted drain ignores child-targeted steers.
    assert await subagents.drain_pending_steers('chat-1', 'user-1') == []
    assert [item['id'] for item in await subagents.drain_pending_steers('chat-1', 'user-1', child_id='child-1')] == [
        child_steer_id
    ]

    superseded = await subagents.mark_steer_superseded('chat-1', 'user-1', [child_steer_id])
    assert superseded == [child_steer_id]
    assert parent.chat['history']['messages'][child_steer_id]['meta']['status'] == 'superseded'
    assert await subagents.drain_pending_steers('chat-1', 'user-1', child_id='child-1') == []


@pytest.mark.asyncio
async def test_steer_helpers_reject_non_owner(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())

    assert await subagents.enqueue_steer('chat-1', 'someone-else', 'hello') is None
    assert await subagents.drain_pending_steers('chat-1', 'someone-else') == []


@pytest.mark.asyncio
async def test_enqueue_steer_marks_active_child_steer_queued(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    child = harness.seed_child(state='running')

    await subagents.enqueue_steer(
        'chat-1',
        'user-1',
        'also check lint',
        source='parent',
        child_id=child.id,
    )

    assert child.meta['state'] == 'steer_queued'


@pytest.mark.asyncio
async def test_steer_endpoint_delivers_into_an_internal_subagent_chat(monkeypatch):
    """The endpoint's child target reaches the child drain and state machine."""
    import open_webui.routers.chats as chats_router

    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    child = harness.seed_child(state='running')

    async def has_active_child(redis, chat_id):
        return chat_id == child.id

    monkeypatch.setattr(chats_router, 'Chats', harness.fake_chats)
    monkeypatch.setattr(chats_router, 'enqueue_steer', subagents.enqueue_steer)
    monkeypatch.setattr(chats_router, 'has_active_tasks', has_active_child)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))
    response = await steer_chat_by_id(
        request,
        child.id,
        SteerChatForm(content='inspect the child output'),
        user=SimpleNamespace(id='user-1', role='user'),
        db=None,
    )

    assert response.queued is True
    assert response.id
    steer = child.chat['history']['messages'][response.id]
    assert steer['meta']['type'] == 'steer'
    assert steer['meta']['source'] == 'user'
    assert steer['meta']['childID'] == child.id
    assert child.meta['state'] == 'steer_queued'

    drained = await subagents.drain_pending_steers(child.id, 'user-1', child_id=child.id)
    assert [message['id'] for message in drained] == [response.id]
    # The instruction lives in the child chat, not in the parent transcript.
    assert response.id not in harness.fake_chats.chats['chat-1'].chat['history']['messages']


def test_steer_never_anchors_history_tip_in_any_status():
    messages = _parent_history(_steer_message('steer-1'))['messages']
    assert is_pending_internal_message(messages['steer-1']) is True
    assert resolve_history_tip(messages, 'tip', blocked=is_pending_internal_message) == 'tip'

    # A delivered steer was only appended to the model request; it is not a
    # visible branch node and must not become the next anchor (which would
    # replay it into a later turn).
    messages['steer-1']['meta']['status'] = 'delivered'
    assert is_pending_internal_message(messages['steer-1']) is True
    assert resolve_history_tip(messages, 'tip', blocked=is_pending_internal_message) == 'tip'

    messages['steer-1']['meta']['status'] = 'superseded'
    assert is_pending_internal_message(messages['steer-1']) is True
    assert resolve_history_tip(messages, 'tip', blocked=is_pending_internal_message) == 'tip'

    # Once a real turn is attached it is traversed like any other node.
    messages['steer-1']['childrenIds'] = ['next-assistant']
    messages['next-assistant'] = {
        'id': 'next-assistant',
        'parentId': 'steer-1',
        'childrenIds': [],
        'role': 'assistant',
        'content': 'answer',
        'done': True,
        'timestamp': 6,
    }
    assert is_pending_internal_message(messages['steer-1']) is False
    assert resolve_history_tip(messages, 'tip', blocked=is_pending_internal_message) == 'next-assistant'


@pytest.mark.asyncio
async def test_enqueue_steer_is_idempotent_for_a_message_id(monkeypatch):
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(history=_parent_history())

    first = await subagents.enqueue_steer('chat-1', 'user-1', 'first text', message_id='stable-steer')
    assert first == 'stable-steer'
    assert await subagents.mark_steer_delivered('chat-1', 'user-1', first) == [first]

    # Re-enqueue with the same id must not reset a consumed steer to pending.
    again = await subagents.enqueue_steer('chat-1', 'user-1', 'changed text', message_id='stable-steer')
    assert again == 'stable-steer'
    steers = [m for m in parent.chat['history']['messages'].values() if m.get('meta', {}).get('type') == 'steer']
    assert len(steers) == 1
    assert steers[0]['meta']['status'] == 'delivered'
    assert steers[0]['content'] == 'first text'
    assert await subagents.drain_pending_steers('chat-1', 'user-1') == []

    # Reusing an id that belongs to a normal message is rejected, not overwritten.
    normal = _steer_message('normal')
    normal['meta'] = {}
    parent.chat['history']['messages']['normal'] = normal
    assert await subagents.enqueue_steer('chat-1', 'user-1', 'oops', message_id='normal') is None
    assert parent.chat['history']['messages']['normal']['content'] == 'steer normal'


@pytest.mark.asyncio
async def test_steer_status_transitions_are_terminal(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())

    delivered_id = await subagents.enqueue_steer('chat-1', 'user-1', 'already seen')
    await subagents.mark_steer_delivered('chat-1', 'user-1', delivered_id)
    assert await subagents.mark_steer_superseded('chat-1', 'user-1', delivered_id) == []
    assert await subagents.mark_steer_delivered('chat-1', 'user-1', delivered_id) == []

    superseded_id = await subagents.enqueue_steer('chat-1', 'user-1', 'never seen')
    assert await subagents.mark_steer_superseded('chat-1', 'user-1', superseded_id) == [superseded_id]
    assert await subagents.mark_steer_delivered('chat-1', 'user-1', superseded_id) == []
    assert await subagents.mark_steer_superseded('chat-1', 'user-1', superseded_id) == []


@pytest.mark.asyncio
async def test_pending_user_steer_is_promoted_at_turn_end_not_as_a_result(monkeypatch):
    """The turn-end backstop promotes an unconsumed user steer into a normal turn.

    It must not be treated as a completion notice, and the promoted node must
    not remain pending (which would deliver it a second time).
    """
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(history=_parent_history(_steer_message('steer-1')))

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    await harness.registry.created[-1][1]

    completions = _parent_completions(harness)
    assert len(completions) == 1
    assert completions[0][1]['messages'][-1]['content'] == 'steer steer-1'

    messages = parent.chat['history']['messages']
    # Collapsed into the new visible turn, not left pending for a later drain.
    assert 'steer-1' not in messages
    promoted = next(
        message
        for message in messages.values()
        if message.get('role') == 'user' and message.get('content') == 'steer steer-1'
    )
    assert promoted['meta']['type'] == 'steer'
    assert promoted['meta']['source'] == 'user'
    assert promoted['meta']['status'] == 'delivered'
    # Explicit marker: readers must not have to infer promotion from the tree.
    assert promoted['meta']['promoted'] is True

    # A later turn-end drain has nothing left to deliver.
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    assert len(_parent_completions(harness)) == 1


# ---------------------------------------------------------------------------
# Stale completion supersession (the 11:51-result fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_continuation_supersedes_stale_pending_result(monkeypatch):
    harness = Harness(monkeypatch)
    stale = _pending_result('stale-result', child_id='child-1')
    parent = harness.seed_parent(history=_parent_history(stale))
    harness.seed_child(child_id='child-1')

    result = await run_delegate(harness, session_id='child-1', prompt='continue the work')
    assert result.startswith('<subagent sessionID="child-1" state="completed">')

    stale_node = parent.chat['history']['messages']['stale-result']
    assert stale_node['meta']['status'] == 'superseded'
    assert is_pending_internal_message(stale_node) is True

    completions_before_drain = len(harness.completions)
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    # The superseded result is not drained into a second synthesis turn.
    assert len(harness.completions) == completions_before_drain


@pytest.mark.asyncio
async def test_supersede_only_matches_the_target_child(monkeypatch):
    harness = Harness(monkeypatch)
    stale_a = _pending_result('stale-a', child_id='child-a')
    stale_b = _pending_result('stale-b', child_id='child-b')
    parent = harness.seed_parent(history=_parent_history(stale_a, stale_b))

    updated = await subagents.supersede_pending_child_results('chat-1', 'user-1', 'child-a')

    assert updated == ['stale-a']
    assert parent.chat['history']['messages']['stale-a']['meta']['status'] == 'superseded'
    assert parent.chat['history']['messages']['stale-b']['meta']['status'] == 'pending'


@pytest.mark.asyncio
async def test_supersede_matches_legacy_envelope_only_result(monkeypatch):
    harness = Harness(monkeypatch)
    stale = _pending_result('legacy-result', child_id='child-1')
    stale['meta'] = {'internal': True, 'type': 'subagent', 'source': 'subagent'}
    stale['content'] = '<subagent sessionID="child-1" state="completed">\nold\n</subagent>'
    parent = harness.seed_parent(history=_parent_history(stale))

    updated = await subagents.supersede_pending_child_results('chat-1', 'user-1', 'child-1')

    assert updated == ['legacy-result']
    assert parent.chat['history']['messages']['legacy-result']['meta']['status'] == 'superseded'


@pytest.mark.asyncio
async def test_supersede_matches_legacy_delegation_only_result(monkeypatch):
    harness = Harness(monkeypatch)
    stale = _pending_result('legacy-result', child_id='child-1')
    stale['meta'] = {'internal': True, 'type': 'subagent', 'delegation_id': 'deleg-old'}
    stale['content'] = 'old subagent result without an envelope'
    parent = harness.seed_parent(history=_parent_history(stale))
    child = harness.seed_child(child_id='child-1')
    child.meta['delegation_id'] = 'deleg-old'

    updated = await subagents.supersede_pending_child_results('chat-1', 'user-1', 'child-1')

    assert updated == ['legacy-result']
    assert parent.chat['history']['messages']['legacy-result']['meta']['status'] == 'superseded'


@pytest.mark.asyncio
async def test_supersede_does_not_touch_batched_child_results(monkeypatch):
    """A batched notice covers several children; one continuation must not drop it."""
    harness = Harness(monkeypatch)
    batched = _pending_result('batched', child_id='child-1')
    batched['meta'] = {
        'internal': True,
        'type': 'subagent',
        'source': 'subagent',
        'childIDs': ['child-1', 'child-2'],
    }
    parent = harness.seed_parent(history=_parent_history(batched))

    updated = await subagents.supersede_pending_child_results('chat-1', 'user-1', 'child-1')

    assert updated == []
    assert 'status' not in parent.chat['history']['messages']['batched']['meta']


# ---------------------------------------------------------------------------
# Child chat lifecycle state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_foreground_child_state_running_then_completed(monkeypatch):
    harness = Harness(monkeypatch, summary='done')
    harness.seed_parent(history=_parent_history())

    await run_delegate(harness)
    child_id, child = harness.find_child()

    assert harness.states_seen == ['running']
    assert child.meta['state'] == 'completed'
    assert (await harness.fake_chats.get_chat_by_id(child_id)).meta['state'] == 'completed'


@pytest.mark.asyncio
async def test_foreground_child_state_error(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    harness._fail_with = RuntimeError('boom')

    result = await run_delegate(harness)

    assert result == 'Error: boom'
    _, child = harness.find_child()
    assert child.meta['state'] == 'error'


@pytest.mark.asyncio
async def test_background_handle_reports_running_then_completes(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    harness.seed_parent(history=_parent_history())

    handle = json.loads(await run_delegate(harness, background=True))
    child = harness.fake_chats.chats[handle['sessionID']]
    assert child.meta['state'] == 'running'

    _, child_task = harness.registry.created[-1]
    await child_task
    assert child.meta['state'] == 'completed'


@pytest.mark.asyncio
async def test_cancelled_background_child_state_interrupted(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    harness.seed_parent(history=_parent_history())
    harness.block_event = asyncio.Event()

    handle = json.loads(await run_delegate(harness, background=True))
    child = harness.fake_chats.chats[handle['sessionID']]
    _, child_task = harness.registry.created[-1]

    # Let the child turn start and prove the state is running while it blocks.
    for _ in range(50):
        if harness.states_seen:
            break
        await asyncio.sleep(0)
    assert harness.states_seen == ['running']
    assert child.meta['state'] == 'running'

    child_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child_task
    assert child.meta['state'] == 'interrupted'


@pytest.mark.asyncio
async def test_new_completion_supersedes_an_older_notice_for_the_same_child(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    parent = harness.seed_parent(history=_parent_history())
    harness.block_event = asyncio.Event()

    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    _, child_task = harness.registry.created[-1]
    for _ in range(50):
        if harness.states_seen:
            break
        await asyncio.sleep(0)

    # A notice from an earlier generation is still waiting when this one lands.
    parent.chat['history']['messages']['old-notice'] = _pending_result('old-notice', child_id=child_id)
    harness.block_event.set()
    await child_task

    messages = parent.chat['history']['messages']
    assert messages['old-notice']['meta']['status'] == 'superseded'
    new_notices = [
        m
        for m in messages.values()
        if (m.get('meta') or {}).get('type') == 'steer'
        and (m.get('meta') or {}).get('source') == 'subagent'
        and (m.get('meta') or {}).get('childID') == child_id
        and (m.get('meta') or {}).get('generation_id')
    ]
    assert len(new_notices) == 1
    assert new_notices[0]['meta'].get('status') in (None, 'pending')


@pytest.mark.asyncio
async def test_cancel_before_first_step_still_finalizes_interrupted(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    harness.seed_parent(history=_parent_history())

    handle = json.loads(await run_delegate(harness, background=True))
    child = harness.fake_chats.chats[handle['sessionID']]
    _, child_task = harness.registry.created[-1]
    assert child.meta['state'] == 'running'

    # Cancel before the coroutine gets its first step; its own handler never runs.
    child_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child_task
    for _ in range(10):
        await asyncio.sleep(0)
        if child.meta['state'] == 'interrupted':
            break
    assert child.meta['state'] == 'interrupted'


@pytest.mark.asyncio
async def test_continuation_generation_ignores_a_stale_finishing_run(monkeypatch):
    """A run that finishes after a continuation must not clobber it or emit a notice."""
    harness = Harness(monkeypatch, active=True)
    parent = harness.seed_parent(history=_parent_history())
    harness.block_event = asyncio.Event()

    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    child = harness.fake_chats.chats[child_id]
    old_generation = child.meta['generation_id']
    _, child_task = harness.registry.created[-1]

    for _ in range(50):
        if harness.states_seen:
            break
        await asyncio.sleep(0)
    assert child.meta['state'] == 'running'

    # Simulate the continuation claiming the child before the old run finishes.
    child.meta['generation_id'] = 'next-generation'
    child.meta['state'] = 'running'
    harness.block_event.set()
    await child_task

    # The old run's terminal state and late notice are both discarded.
    assert child.meta['state'] == 'running'
    assert child.meta['generation_id'] == 'next-generation'
    assert old_generation != child.meta['generation_id']
    messages = parent.chat['history']['messages']
    assert not any(
        (m.get('meta') or {}).get('type') == 'subagent' and (m.get('meta') or {}).get('childID') == child_id
        for m in messages.values()
    )


# ---------------------------------------------------------------------------
# GET /api/v1/chats/{id}/subagents ownership
# ---------------------------------------------------------------------------


class FakeRouterChats:
    def __init__(self):
        self.parent = SimpleNamespace(id='chat-1', user_id='user-1')
        self.children = {
            'child-1': SimpleNamespace(
                id='child-1',
                title='Subagent: one',
                meta={'state': 'running'},
                updated_at=3,
            )
        }

    async def get_chat_by_id_and_user_id(self, id, user_id, db=None):
        if id == 'chat-1' and user_id == 'user-1':
            return self.parent
        return None

    async def get_internal_chat_ids_by_parent_id(self, parent_chat_id, user_id):
        if parent_chat_id == 'chat-1' and user_id == 'user-1':
            return ['child-1']
        return []

    async def get_chat_by_id(self, id, db=None):
        return self.children.get(id)


@pytest.mark.asyncio
async def test_subagent_listing_returns_child_state_for_owner(monkeypatch):
    import open_webui.routers.chats as chats_router

    monkeypatch.setattr(chats_router, 'Chats', FakeRouterChats())
    user = SimpleNamespace(id='user-1', role='user')

    items = await get_chat_subagents_by_id('chat-1', user=user, db=None)

    assert len(items) == 1
    assert items[0].id == 'child-1'
    assert items[0].title == 'Subagent: one'
    assert items[0].state == 'running'
    assert items[0].updated_at == 3


@pytest.mark.asyncio
async def test_subagent_listing_rejects_non_owner(monkeypatch):
    import open_webui.routers.chats as chats_router

    monkeypatch.setattr(chats_router, 'Chats', FakeRouterChats())
    user = SimpleNamespace(id='other-user', role='user')

    with pytest.raises(HTTPException) as exc:
        await get_chat_subagents_by_id('chat-1', user=user, db=None)
    assert exc.value.status_code == 401

    # A missing/foreign parent is the same non-disclosing 401, not a 500.
    with pytest.raises(HTTPException) as exc:
        await get_chat_subagents_by_id('missing-chat', user=SimpleNamespace(id='user-1'), db=None)
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Slice B: parent -> child steering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_steering_a_running_child_enqueues_and_acks(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    child = harness.seed_child(state='running', history=_running_child_history())
    before_ids = set(child.chat['history']['messages'])

    result = json.loads(await run_delegate(harness, session_id='child-1', prompt='focus on the failing tests'))

    assert result['status'] == 'steer_queued'
    assert result['sessionID'] == 'child-1'
    assert 'queued' in result['output']
    # No second child turn and no new task: the live run consumes the steer.
    assert harness.registry.created == []
    assert child.meta['state'] == 'steer_queued'

    added = set(child.chat['history']['messages']) - before_ids
    assert len(added) == 1
    steer = child.chat['history']['messages'][added.pop()]
    assert steer['meta']['type'] == 'steer'
    assert steer['meta']['source'] == 'parent'
    assert steer['meta']['childID'] == 'child-1'
    assert steer['meta']['status'] == 'pending'
    assert 'focus on the failing tests' in steer['content']
    assert is_pending_internal_message(steer) is True

    # The child drain sees a steer addressed to this chat; delivering it resumes
    # the run from steer_queued.
    drained = await subagents.drain_pending_steers('child-1', 'user-1')
    assert [item['id'] for item in drained] == [steer['id']]
    assert await subagents.mark_steer_delivered('child-1', 'user-1', steer['id']) == [steer['id']]
    assert child.meta['state'] == 'running'
    assert await subagents.drain_pending_steers('child-1', 'user-1') == []


@pytest.mark.asyncio
async def test_steering_a_running_child_works_for_background_calls(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    harness.seed_child(state='running', history=_running_child_history())

    result = json.loads(await run_delegate(harness, session_id='child-1', prompt='also run lint', background=True))

    assert result['status'] == 'steer_queued'
    assert harness.registry.created == []


@pytest.mark.asyncio
async def test_steer_that_loses_the_running_race_falls_back_to_continuation(monkeypatch):
    """A child finishing between the activity check and the write must not strand a steer."""
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    child = harness.seed_child(state='running', history=_running_child_history())

    real_enqueue = subagents.enqueue_steer

    async def racing_enqueue(*args, **kwargs):
        # The child writes its final tip and terminal state just before the
        # steer is persisted.
        child.meta['state'] = 'completed'
        child.chat['history']['messages']['child-assistant']['done'] = True
        child.chat['history']['messages']['child-assistant']['content'] = 'finished first'
        return await real_enqueue(*args, **kwargs)

    monkeypatch.setattr(subagents, 'enqueue_steer', racing_enqueue)

    result = await run_delegate(harness, session_id='child-1', prompt='too late to steer')

    # The stale steer is not queued; the call becomes a normal continuation.
    assert result.startswith('<subagent sessionID="child-1" state="completed">')
    steers = [
        message
        for message in child.chat['history']['messages'].values()
        if (message.get('meta') or {}).get('type') == 'steer' and (message.get('meta') or {}).get('source') == 'parent'
    ]
    assert len(steers) == 1
    assert steers[0]['meta']['status'] == 'superseded'
    assert child.meta['state'] == 'completed'
    assert _completion_notices(harness.fake_chats.chats['chat-1'].chat['history']['messages']) == []


# ---------------------------------------------------------------------------
# Slice B: child -> parent completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_notice_uses_child_latest_tip_at_completion(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    parent = harness.seed_parent(history=_parent_history())
    harness.block_event = asyncio.Event()

    before_tasks = len(harness.registry.created)
    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    child = harness.fake_chats.chats[child_id]
    child_task = harness.registry.created[before_tasks][1]
    await _task_started(harness)

    # A steer/continuation attached a newer final assistant leaf while the run
    # was still finishing. The notice must describe that tip, not the spawn-time
    # assistant snapshot the handler writes into.
    assistant_id = next(
        message['id'] for message in child.chat['history']['messages'].values() if message.get('role') == 'assistant'
    )
    await harness.fake_chats.upsert_message_to_chat_by_id_and_message_id(
        child_id,
        'final-after-steer',
        {
            'id': 'final-after-steer',
            'parentId': assistant_id,
            'childrenIds': [],
            'role': 'assistant',
            'content': 'final answer after the steer',
            'done': True,
            'timestamp': 99,
        },
    )
    harness.block_event.set()
    await child_task

    notices = _completion_notices(parent.chat['history']['messages'])
    assert len(notices) == 1
    notice = notices[0]
    assert notice['meta']['childID'] == child_id
    assert notice['meta']['generation_id']
    assert 'final answer after the steer' in notice['content']
    assert 'state="completed"' in notice['content']
    # No legacy pending result node is created alongside the steer notice.
    assert not [
        message
        for message in parent.chat['history']['messages'].values()
        if (message.get('meta') or {}).get('type') == 'subagent'
        and (message.get('meta') or {}).get('status') in (None, 'pending')
    ]
    assert child.meta['state'] == 'completed'
    # Parent is active: Slice A owns boundary delivery, so no synthesis here.
    assert _parent_completions(harness) == []


@pytest.mark.asyncio
async def test_idle_parent_delivers_completion_steer_once(monkeypatch):
    harness = Harness(monkeypatch, active=False)
    parent = harness.seed_parent(history=_parent_history())

    before_tasks = len(harness.registry.created)
    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    child_task = harness.registry.created[before_tasks][1]
    await child_task
    await _finish_registered_tasks(harness)

    parent_completions = _parent_completions(harness)
    assert len(parent_completions) == 1
    payload = parent_completions[0][1]
    completion_content = payload['messages'][-1]['content']
    assert '<subagent' in completion_content
    assert harness.summary in completion_content
    # The completion steer was consumed by the drain, not left pending.
    assert _completion_notices(parent.chat['history']['messages']) == []

    # A second turn-end drain must not synthesize the completion again.
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    assert len(_parent_completions(harness)) == 1
    assert child_id


@pytest.mark.asyncio
async def test_turn_end_drain_ignores_child_targeted_instructions(monkeypatch):
    """A parent -> child steer stays for the child's boundary, never this chat."""
    harness = Harness(monkeypatch)
    child_steer = _steer_message('child-steer', child_id='child-1')
    child_steer['meta']['source'] = 'parent'
    parent = harness.seed_parent(history=_parent_history(child_steer))

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})

    assert _parent_completions(harness) == []
    assert parent.chat['history']['messages']['child-steer']['meta']['status'] == 'pending'


@pytest.mark.asyncio
async def test_turn_end_drain_delivers_steer_and_completion_once(monkeypatch):
    """The two pending shapes drain on their own turns without duplication."""
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(
        history=_parent_history(
            _steer_message('user-steer'),
            _completion_steer('notice-1', child_id='child-1', body='old answer'),
        )
    )

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    await _finish_registered_tasks(harness)

    completions = _parent_completions(harness)
    assert len(completions) == 2
    contents = [call[1]['messages'][-1]['content'] for call in completions]
    assert sum('steer user-steer' in content for content in contents) == 1
    assert sum('old answer' in content for content in contents) == 1
    assert _completion_notices(parent.chat['history']['messages']) == []


@pytest.mark.asyncio
async def test_delivered_completion_steer_is_not_drained_again(monkeypatch):
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(history=_parent_history(_completion_steer('notice-1', status='delivered')))

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})

    assert harness.completions == []
    assert parent.chat['history']['messages']['notice-1']['meta']['status'] == 'delivered'


@pytest.mark.asyncio
async def test_completion_steer_is_parent_deliverable_not_a_child_instruction(monkeypatch):
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history(_completion_steer('notice-1', child_id='child-1')))

    parent_steers = await subagents.drain_pending_steers('chat-1', 'user-1')
    assert [item['id'] for item in parent_steers] == ['notice-1']
    # A completion notice names its source child; it must never be re-injected
    # into that child as a parent instruction.
    assert await subagents.drain_pending_steers('chat-1', 'user-1', child_id='child-1') == []


# ---------------------------------------------------------------------------
# Slice B: supersession and late steer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_continuation_supersedes_a_pending_completion_steer(monkeypatch):
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(history=_parent_history(_completion_steer('old-notice', generation_id='gen-old')))
    harness.seed_child(child_id='child-1', state='completed')

    result = await run_delegate(harness, session_id='child-1', prompt='continue the work')

    assert result.startswith('<subagent sessionID="child-1" state="completed">')
    assert parent.chat['history']['messages']['old-notice']['meta']['status'] == 'superseded'
    assert _completion_notices(parent.chat['history']['messages']) == []
    assert _parent_completions(harness) == []


@pytest.mark.asyncio
async def test_late_steer_returns_pre_steer_note_and_queues_new_turn(monkeypatch):
    harness = Harness(monkeypatch, active=False)
    parent = harness.seed_parent(history=_parent_history(_completion_steer('old-notice', body='old final answer')))
    harness.seed_child(child_id='child-1', state='completed')

    before_tasks = len(harness.registry.created)
    result = await run_delegate(
        harness,
        session_id='child-1',
        prompt='change direction',
        background=True,
    )

    assert 'pre_steer="true"' in result
    assert 'old final answer' in result
    assert 'generated before your steering message' in result
    assert 'new response is coming' in result
    assert parent.chat['history']['messages']['old-notice']['meta']['status'] == 'superseded'

    # The new turn runs even though the caller could have expected a running
    # handle; it is not synthesized from the old completion.
    new_tasks = harness.registry.created[before_tasks:]
    assert len(new_tasks) == 1
    child_task = new_tasks[0][1]
    child = harness.fake_chats.chats['child-1']
    assert child.meta['state'] == 'running'

    await child_task
    await _finish_registered_tasks(harness)

    assert child.meta['state'] == 'completed'
    parent_completions = _parent_completions(harness)
    assert len(parent_completions) == 1
    synthesis_content = parent_completions[0][1]['messages'][-1]['content']
    assert harness.summary in synthesis_content
    assert 'old final answer' not in synthesis_content


@pytest.mark.asyncio
async def test_cancelled_child_with_queued_steer_stays_interrupted(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    parent = harness.seed_parent(history=_parent_history())
    harness.block_event = asyncio.Event()

    before_tasks = len(harness.registry.created)
    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    child = harness.fake_chats.chats[child_id]
    child_task = harness.registry.created[before_tasks][1]
    await _task_started(harness)
    assert child.meta['state'] == 'running'

    ack = json.loads(await run_delegate(harness, session_id=child_id, prompt='change direction'))
    assert ack['status'] == 'steer_queued'
    assert child.meta['state'] == 'steer_queued'

    child_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child_task
    assert child.meta['state'] == 'interrupted'

    steer = next(
        message
        for message in child.chat['history']['messages'].values()
        if (message.get('meta') or {}).get('type') == 'steer'
    )
    assert await subagents.mark_steer_delivered(child_id, 'user-1', steer['id']) == [steer['id']]
    # Delivery must not resurrect a finished child as running.
    assert child.meta['state'] == 'interrupted'

    # The interrupted completion is delivered to the parent as a completion
    # steer, so the parent learns the child stopped instead of hanging.
    notices = _completion_notices(parent.chat['history']['messages'])
    assert len(notices) == 1
    assert notices[0]['meta']['state'] == 'interrupted'


# ---------------------------------------------------------------------------
# Cross-slice integration: Slice B's durable store <-> Slice A's boundaries
# ---------------------------------------------------------------------------


def _bind_middleware_to_harness(monkeypatch, harness):
    """Point the middleware steers at the same fake store the harness uses."""
    from open_webui.utils import middleware

    monkeypatch.setattr(middleware, 'Chats', harness.fake_chats)
    monkeypatch.setattr(middleware, 'drain_pending_steers', subagents.drain_pending_steers)
    monkeypatch.setattr(middleware, 'mark_steer_delivered', subagents.mark_steer_delivered)
    return middleware


@pytest.mark.asyncio
async def test_parent_user_steer_is_delivered_once_at_a_boundary(monkeypatch):
    """B enqueues in the parent chat; A's parent boundary consumes and annotates."""
    harness = Harness(monkeypatch, active=False)
    harness.seed_parent(history=_parent_history())
    middleware = _bind_middleware_to_harness(monkeypatch, harness)
    user = SimpleNamespace(id='user-1')

    steer_id = await subagents.enqueue_steer(
        'chat-1',
        'user-1',
        'focus on the failing test',
        source='user',
        model_id='test-model',
    )
    assert steer_id

    # A parent turn (not internal) reads its own chat with the parent selector.
    injected, pending, store_chat_id = await middleware._inject_pending_steers(
        {'messages': [{'role': 'user', 'content': 'start'}]},
        {'chat_id': 'chat-1'},
        user,
    )
    assert [steer['id'] for steer in pending] == [steer_id]
    assert store_chat_id == 'chat-1'
    assert middleware.STEER_BLOCK_OPEN in injected['messages'][-1]['content']

    annotations = await middleware._mark_boundary_steers_delivered(
        {'chat_id': 'chat-1'},
        user,
        pending,
        SimpleNamespace(status_code=200),
        delivery_chat_id=store_chat_id,
    )
    assert annotations == [{'id': steer_id, 'source': 'user', 'content': 'focus on the failing test'}]

    # Delivered is terminal: neither a later boundary nor the idle turn-end
    # drain can promote the same steer into a second turn.
    assert await subagents.drain_pending_steers('chat-1', 'user-1') == []
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    assert harness.completions == []


@pytest.mark.asyncio
async def test_parent_child_steer_stays_in_the_child_store_and_resumes_it(monkeypatch):
    """B writes the instruction to the child; A's child boundary consumes it."""
    harness = Harness(monkeypatch)
    harness.seed_parent(history=_parent_history())
    child = harness.seed_child(child_id='child-1', state='running')
    middleware = _bind_middleware_to_harness(monkeypatch, harness)
    user = SimpleNamespace(id='user-1')

    steer_id = await subagents.enqueue_steer(
        'child-1',
        'user-1',
        'focus on the failing test',
        source='parent',
        child_id='child-1',
        model_id='test-model',
    )
    assert steer_id
    assert child.meta['state'] == 'steer_queued'

    # The parent turn must not see the child's instruction.
    parent_injected, parent_pending, parent_store = await middleware._inject_pending_steers(
        {'messages': [{'role': 'user', 'content': 'start'}]},
        {'chat_id': 'chat-1'},
        user,
    )
    assert parent_pending == []
    assert parent_store == 'chat-1'
    assert all(middleware.STEER_BLOCK_OPEN not in (m.get('content') or '') for m in parent_injected['messages'])

    # The internal child turn reads the child chat with `child_id=<child>`.
    child_injected, child_pending, child_store = await middleware._inject_pending_steers(
        {'messages': [{'role': 'tool', 'tool_call_id': 'call-1', 'content': 'ok'}]},
        {'chat_id': 'child-1', 'internal': True},
        user,
    )
    assert [steer['id'] for steer in child_pending] == [steer_id]
    assert child_store == 'child-1'
    assert child_injected['messages'][-1]['content'].count('focus on the failing test') == 1

    annotations = await middleware._mark_boundary_steers_delivered(
        {'chat_id': 'child-1', 'internal': True},
        user,
        child_pending,
        SimpleNamespace(status_code=200),
        delivery_chat_id=child_store,
    )
    assert annotations == [{'id': steer_id, 'source': 'parent', 'content': 'focus on the failing test'}]
    # Delivery writes to the same store and resumes the paused child.
    assert child.meta['state'] == 'running'
    assert await subagents.drain_pending_steers('child-1', 'user-1', child_id='child-1') == []


@pytest.mark.asyncio
async def test_completion_steer_is_parent_only_and_drains_once(monkeypatch):
    """A parent boundary sees the completion; a child boundary never does."""
    harness = Harness(monkeypatch, active=False)
    parent = harness.seed_parent(history=_parent_history())
    harness.seed_child(child_id='child-1', state='completed')
    middleware = _bind_middleware_to_harness(monkeypatch, harness)
    user = SimpleNamespace(id='user-1')

    steer_id = await subagents.enqueue_steer(
        'chat-1',
        'user-1',
        '<subagent sessionID="child-1" state="completed">\nfinished\n</subagent>',
        source='subagent',
        child_id='child-1',
        generation_id='gen-1',
        model_id='test-model',
        message_id='notice-1',
        state='completed',
        description='a task',
    )
    assert steer_id == 'notice-1'

    # A's parent boundary includes the completion notice.
    _, parent_pending, parent_store = await middleware._inject_pending_steers(
        {'messages': [{'role': 'user', 'content': 'start'}]},
        {'chat_id': 'chat-1'},
        user,
    )
    assert [steer['id'] for steer in parent_pending] == ['notice-1']
    assert parent_store == 'chat-1'

    # A's internal child boundary reads the child chat and excludes it.
    _, child_pending, child_store = await middleware._inject_pending_steers(
        {'messages': [{'role': 'user', 'content': 'start'}]},
        {'chat_id': 'child-1', 'internal': True},
        user,
    )
    assert child_pending == []
    assert child_store == 'child-1'

    # With the parent idle, the turn-end drain consumes it exactly once.
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    await _finish_registered_tasks(harness)
    assert len(harness.completions) == 1
    assert 'finished' in harness.completions[0][1]['messages'][-1]['content']
    assert _completion_notices(parent.chat['history']['messages']) == []

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    assert len(harness.completions) == 1
