"""Focused tests for boundary-safe steer injection in the tool loop.

Slice A appends pending steer records to the *request* message list only,
immediately before the provider call and after the complete tool batch. These
tests exercise the middleware helpers directly with in-memory fakes so they pin
the boundary invariants without a provider, Redis, or a database:

* the steer block is the last message and never splits a tool call from its
  results;
* the frozen turn messages are never mutated, so a steer cannot accumulate
  across rounds;
* a failed provider call leaves the records pending, while a started request
  marks them delivered;
* compare-mode fan-out injects from the primary branch only;
* the final assistant message meta carries the consumed steer annotations.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from open_webui.utils import middleware


def _steer(
    steer_id: str = 'steer-1',
    content: str = 'also check the tests',
    *,
    source: str = 'user',
    child_id: str | None = None,
) -> dict:
    meta = {
        'internal': True,
        'type': 'steer',
        'source': source,
        'status': 'pending',
    }
    if child_id:
        meta['childID'] = child_id
    return {
        'id': steer_id,
        'role': 'user',
        'content': content,
        'meta': meta,
    }


def _tool_round_messages() -> list[dict]:
    """A request at the moment after all tool results have been appended."""
    return [
        {'role': 'system', 'content': 'be helpful'},
        {'role': 'user', 'content': 'run the build'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'run', 'arguments': '{}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'build ok'},
        {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'tests ok'},
    ]


def _fake_drain(by_child_id: dict[str | None, list[dict]], calls: list | None = None):
    async def drain(chat_id, user_id, child_id=None):
        if calls is not None:
            calls.append((chat_id, user_id, child_id))
        return list(by_child_id.get(child_id) or [])

    return drain


@pytest.mark.asyncio
async def test_parent_steer_is_appended_after_all_tool_results(monkeypatch):
    steer = _steer()
    monkeypatch.setattr(middleware, 'drain_pending_steers', _fake_drain({None: [steer]}))

    frozen = _tool_round_messages()
    form_data = {'model': 'test-model', 'messages': frozen}
    new_form_data = {'model': 'test-model', 'messages': copy.deepcopy(frozen)}

    injected, pending = await middleware._inject_pending_steers(
        new_form_data,
        {'chat_id': 'chat-1'},
        SimpleNamespace(id='user-1'),
    )

    assert pending == [steer]
    messages = injected['messages']

    # The synthetic block is the last message, after both tool results.
    assert messages[-1]['role'] == 'user'
    assert middleware.STEER_BLOCK_OPEN in messages[-1]['content']
    assert '[user] also check the tests' in messages[-1]['content']
    assert messages[-1]['content'].rstrip().endswith(middleware.STEER_BLOCK_CLOSE)

    # The tool batch stays contiguous: assistant(tool_calls) -> tool -> tool -> block.
    assert [message['role'] for message in messages[-4:]] == ['assistant', 'tool', 'tool', 'user']
    assert [message.get('tool_call_id') for message in messages[-3:-1]] == ['call-1', 'call-2']

    # Neither the post-transform request nor the frozen turn messages were mutated.
    assert new_form_data['messages'] == frozen
    assert not any('check the tests' in (message.get('content') or '') for message in form_data['messages'])


@pytest.mark.asyncio
async def test_steer_does_not_accumulate_across_rounds(monkeypatch):
    steer = _steer()
    monkeypatch.setattr(middleware, 'drain_pending_steers', _fake_drain({None: [steer]}))

    form_data = {'model': 'test-model', 'messages': _tool_round_messages()}
    user = SimpleNamespace(id='user-1')
    metadata = {'chat_id': 'chat-1'}

    # Each round rebuilds new_form_data from the frozen form_data list, exactly
    # like the tool loop does, so only the current round's steer is present.
    first, _ = await middleware._inject_pending_steers(
        {'model': 'test-model', 'messages': list(form_data['messages'])}, metadata, user
    )
    second, _ = await middleware._inject_pending_steers(
        {'model': 'test-model', 'messages': list(form_data['messages'])}, metadata, user
    )

    for injected in (first, second):
        blocks = [
            message for message in injected['messages'] if middleware.STEER_BLOCK_OPEN in (message.get('content') or '')
        ]
        assert len(blocks) == 1

    assert not any(middleware.STEER_BLOCK_OPEN in (message.get('content') or '') for message in form_data['messages'])


@pytest.mark.asyncio
async def test_failed_provider_leaves_steers_pending_and_success_marks_delivered(monkeypatch):
    marked: list[tuple] = []

    async def fake_mark(chat_id, user_id, message_ids):
        marked.append((chat_id, user_id, list(message_ids)))
        return list(message_ids)

    monkeypatch.setattr(middleware, 'mark_steer_delivered', fake_mark)

    user = SimpleNamespace(id='user-1')
    metadata = {'chat_id': 'chat-1'}
    steers = [_steer()]

    async def failing_send():
        raise RuntimeError('provider unavailable')

    # A provider exception propagates before any status write: the record stays
    # pending for the turn-end drain.
    with pytest.raises(RuntimeError):
        await middleware._send_with_boundary_steers(failing_send, metadata, user, steers)
    assert marked == []

    # An error response is not a started request either.
    async def error_response():
        return SimpleNamespace(status_code=500)

    _, annotations = await middleware._send_with_boundary_steers(error_response, metadata, user, steers)
    assert annotations == []
    assert marked == []

    async def successful_send():
        return SimpleNamespace(status_code=200)

    _, annotations = await middleware._send_with_boundary_steers(successful_send, metadata, user, steers)
    assert marked == [('chat-1', 'user-1', ['steer-1'])]
    assert annotations == [{'id': 'steer-1', 'source': 'user', 'content': 'also check the tests'}]


@pytest.mark.asyncio
async def test_compare_mode_only_primary_branch_injects(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        middleware,
        'drain_pending_steers',
        _fake_drain({None: [_steer()]}, calls),
    )

    user = SimpleNamespace(id='user-1')
    form_data = {'messages': _tool_round_messages()}

    secondary, secondary_steers = await middleware._inject_pending_steers(
        {'messages': list(form_data['messages'])},
        {'chat_id': 'chat-1', 'compare_mode': True, 'is_primary_branch': False},
        user,
    )
    assert secondary_steers == []
    assert secondary['messages'] == form_data['messages']
    assert calls == []  # the secondary branch never even reads the pending set

    primary, primary_steers = await middleware._inject_pending_steers(
        {'messages': list(form_data['messages'])},
        {'chat_id': 'chat-1', 'compare_mode': True, 'is_primary_branch': True},
        user,
    )
    assert len(primary_steers) == 1
    assert middleware.STEER_BLOCK_OPEN in primary['messages'][-1]['content']
    assert calls == [('chat-1', 'user-1', None), ('chat-1', 'user-1', 'chat-1')]


@pytest.mark.asyncio
async def test_child_turn_drains_child_targeted_steers_once(monkeypatch):
    child_steer = _steer(
        'child-steer',
        'focus on the failing test',
        child_id='child-1',
    )
    # Return the same record for both selectors on purpose: the merge must
    # de-duplicate by id so a steer can never be injected twice.
    monkeypatch.setattr(middleware, 'drain_pending_steers', _fake_drain({None: [child_steer]}))

    injected, pending = await middleware._inject_pending_steers(
        {'messages': [{'role': 'tool', 'tool_call_id': 'call-1', 'content': 'ok'}]},
        {'chat_id': 'child-1'},
        SimpleNamespace(id='user-1'),
    )

    assert [steer['id'] for steer in pending] == ['child-steer']
    assert injected['messages'][-1]['role'] == 'user'
    assert injected['messages'][-1]['content'].count('focus on the failing test') == 1


def test_multiple_steers_merge_into_one_block():
    block = middleware._build_steer_block(
        [
            _steer('steer-1', 'first instruction'),
            _steer('steer-2', 'second instruction', source='parent'),
        ]
    )

    assert block.count(middleware.STEER_BLOCK_OPEN) == 1
    assert block.count(middleware.STEER_BLOCK_CLOSE) == 1
    assert '[user] first instruction' in block
    assert '[parent] second instruction' in block

    appended = middleware._append_steer_block(
        {'messages': [{'role': 'tool', 'tool_call_id': 'call-1', 'content': 'ok'}]},
        [_steer('steer-1', 'first instruction')],
    )
    assert len(appended['messages']) == 2
    assert appended['messages'][-1]['role'] == 'user'


@pytest.mark.asyncio
async def test_tool_approval_pause_preserves_steer_annotations(monkeypatch):
    captured: dict = {}

    class FakeChats:
        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(chat_id, message_id, data, touch=True):
            captured['data'] = data

    async def fake_event_emitter(metadata, update_db=True):
        return None

    monkeypatch.setattr(middleware, 'Chats', FakeChats)
    monkeypatch.setattr(middleware, 'get_event_emitter', fake_event_emitter)

    annotations = middleware._steer_annotations([_steer(source='parent')])
    await middleware.pause_for_tool_approval(
        'chat-1',
        'message-1',
        [{'type': 'function_call', 'call_id': 'call-1', 'status': 'in_progress'}],
        {'messages': []},
        {'session_id': 'session-1'},
        extra_meta={'steers': annotations},
    )

    assert captured['data']['meta']['steers'] == annotations


def test_final_message_meta_annotates_consumed_steers():
    consumed = middleware._steer_annotations([_steer(source='parent')])

    meta = middleware._build_final_message_meta({'task_id': 'task-1'}, consumed, ended_at=123.5)

    assert meta['task_id'] == 'task-1'
    assert meta['response_ended_at'] == 123.5
    assert meta['steers'] == [
        {
            'id': 'steer-1',
            'source': 'parent',
            'content': 'also check the tests',
        }
    ]

    # Re-finalizing a resumed turn merges by id instead of duplicating.
    again = middleware._build_final_message_meta(meta, consumed, ended_at=456.0)
    assert again['steers'] == meta['steers']
    assert again['response_ended_at'] == 456.0
