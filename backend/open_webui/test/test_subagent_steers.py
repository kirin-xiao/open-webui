"""Focused tests for the steering foundation (Slice 0).

Covers the durable steer record contract, stale completion supersession,
child chat lifecycle state, and the ownership-checked subagent listing
endpoint. These use the same in-memory ``Chats`` fake as ``test_subagents`` so
the real delegate / drain coroutines run without a database or Redis.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.models.chats import ChatModel, Chats
from open_webui.routers.chats import get_chat_subagents_by_id
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
from open_webui.utils.misc import is_pending_internal_message, resolve_history_tip


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
async def test_pending_steer_is_not_drained_as_a_subagent_result(monkeypatch):
    harness = Harness(monkeypatch)
    parent = harness.seed_parent(history=_parent_history(_steer_message('steer-1')))

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})

    assert harness.completions == []
    assert parent.chat['history']['messages']['steer-1']['meta']['status'] == 'pending'
    assert parent.chat['history']['currentId'] == 'tip'


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
        if (m.get('meta') or {}).get('type') == 'subagent'
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
