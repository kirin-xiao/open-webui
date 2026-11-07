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

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_webui.utils import middleware

MIDDLEWARE_PATH = Path(middleware.__file__)


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

    injected, pending, steer_chat_id = await middleware._inject_pending_steers(
        new_form_data,
        {'chat_id': 'chat-1'},
        SimpleNamespace(id='user-1'),
    )

    assert pending == [steer]
    assert steer_chat_id == 'chat-1'
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
    first, _, _ = await middleware._inject_pending_steers(
        {'model': 'test-model', 'messages': list(form_data['messages'])}, metadata, user
    )
    second, _, _ = await middleware._inject_pending_steers(
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
async def test_failed_delivery_write_is_not_reinjected_in_a_later_round(monkeypatch):
    """The provider saw the block, so a status-write failure must not repeat it."""
    steer = _steer()
    monkeypatch.setattr(middleware, 'drain_pending_steers', _fake_drain({None: [steer]}))

    async def failing_mark(chat_id, user_id, message_ids):
        raise RuntimeError('history write failed')

    monkeypatch.setattr(middleware, 'mark_steer_delivered', failing_mark)

    metadata = {'chat_id': 'chat-1'}
    user = SimpleNamespace(id='user-1')
    form_data = {'messages': _tool_round_messages()}

    first, pending, _ = await middleware._inject_pending_steers(form_data, metadata, user)
    assert [item['id'] for item in pending] == ['steer-1']

    async def successful_send():
        return SimpleNamespace(status_code=200)

    # The write failed, so the record stays pending and the annotation is empty.
    annotations = await middleware._mark_boundary_steers_delivered(
        metadata, user, pending, await successful_send()
    )
    assert annotations == []

    # A later boundary in the same turn excludes what was already sent, so the
    # model input does not contain the steer twice even though it is still pending.
    second, pending_again, _ = await middleware._inject_pending_steers(
        form_data, metadata, user, exclude_ids={'steer-1'}
    )
    assert pending_again == []
    assert not any(middleware.STEER_BLOCK_OPEN in (message.get('content') or '') for message in second['messages'])


def test_streaming_error_status_is_not_a_started_request():
    error_stream = middleware.StreamingResponse(iter(()), status_code=502)
    assert middleware._response_started(error_stream) is False
    assert middleware._response_started(SimpleNamespace(status_code=200)) is True


def test_stateful_responses_replay_disables_boundary_injection(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', True)

    # Stateless calls and the first response are injectable; only a replay via
    # previous_response_id is not.
    assert middleware._boundary_steering_enabled(None) is True
    assert middleware._boundary_steering_enabled('resp-1') is False

    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', False)
    assert middleware._boundary_steering_enabled('resp-1') is True


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

    secondary, secondary_steers, secondary_chat = await middleware._inject_pending_steers(
        {'messages': list(form_data['messages'])},
        {'chat_id': 'chat-1', 'compare_mode': True, 'is_primary_branch': False},
        user,
    )
    assert secondary_steers == []
    assert secondary_chat is None
    assert secondary['messages'] == form_data['messages']
    assert calls == []  # the secondary branch never even reads the pending set

    primary, primary_steers, primary_chat = await middleware._inject_pending_steers(
        {'messages': list(form_data['messages'])},
        {'chat_id': 'chat-1', 'compare_mode': True, 'is_primary_branch': True},
        user,
    )
    assert len(primary_steers) == 1
    assert primary_chat == 'chat-1'
    assert middleware.STEER_BLOCK_OPEN in primary['messages'][-1]['content']
    assert calls == [('chat-1', 'user-1', None)]


@pytest.mark.asyncio
async def test_child_turn_reads_its_steers_from_its_own_history(monkeypatch):
    child_steer = _steer(
        'child-steer',
        'focus on the failing test',
        child_id='child-1',
    )
    calls: list = []

    class FakeChats:
        @staticmethod
        async def get_chat_by_id(chat_id):
            assert chat_id == 'child-1'
            return SimpleNamespace(
                meta={
                    'internal': True,
                    'type': 'subagent',
                    'parent_chat_id': 'parent-1',
                }
            )

    async def fake_drain(chat_id, user_id, child_id=None):
        calls.append((chat_id, user_id, child_id))
        # Slice B stores parent -> child instructions in the child chat itself.
        # A child turn therefore selects them from its own history.
        if chat_id == 'child-1' and child_id == 'child-1':
            return [child_steer]
        return []

    monkeypatch.setattr(middleware, 'Chats', FakeChats)
    monkeypatch.setattr(middleware, 'drain_pending_steers', fake_drain)

    injected, pending, steer_chat_id = await middleware._inject_pending_steers(
        {'messages': [{'role': 'tool', 'tool_call_id': 'call-1', 'content': 'ok'}]},
        {'chat_id': 'child-1', 'internal': True},
        SimpleNamespace(id='user-1'),
    )

    assert [steer['id'] for steer in pending] == ['child-steer']
    assert steer_chat_id == 'child-1'
    # One selector only: the child must not also drain parent-targeted steers.
    assert calls == [('child-1', 'user-1', 'child-1')]
    assert injected['messages'][-1]['role'] == 'user'
    assert injected['messages'][-1]['content'].count('focus on the failing test') == 1


@pytest.mark.asyncio
async def test_internal_non_child_turn_uses_the_parent_selector(monkeypatch):
    """Synthesis/timer turns are internal but still belong to the parent chat."""
    calls: list = []

    class FakeChats:
        @staticmethod
        async def get_chat_by_id(chat_id):
            return SimpleNamespace(meta={'internal': True, 'type': 'note'})

    async def fake_drain(chat_id, user_id, child_id=None):
        calls.append((chat_id, user_id, child_id))
        return []

    monkeypatch.setattr(middleware, 'Chats', FakeChats)
    monkeypatch.setattr(middleware, 'drain_pending_steers', fake_drain)

    _, pending, steer_chat_id = await middleware._inject_pending_steers(
        {'messages': [{'role': 'user', 'content': 'hi'}]},
        {'chat_id': 'chat-1', 'internal': True},
        SimpleNamespace(id='user-1'),
    )

    assert pending == []
    assert steer_chat_id == 'chat-1'
    assert calls == [('chat-1', 'user-1', None)]


@pytest.mark.asyncio
async def test_child_steer_delivery_is_written_to_the_child_history(monkeypatch):
    marked: list = []

    async def fake_mark(chat_id, user_id, message_ids):
        marked.append((chat_id, user_id, list(message_ids)))
        return list(message_ids)

    monkeypatch.setattr(middleware, 'mark_steer_delivered', fake_mark)

    async def successful_send():
        return SimpleNamespace(status_code=200)

    annotations = await middleware._mark_boundary_steers_delivered(
        {'chat_id': 'child-1'},
        SimpleNamespace(id='user-1'),
        [_steer('child-steer', child_id='child-1')],
        await successful_send(),
        delivery_chat_id='child-1',
    )

    assert marked == [('child-1', 'user-1', ['child-steer'])]
    assert annotations == [
        {'id': 'child-steer', 'source': 'user', 'content': 'also check the tests'}
    ]


def test_malformed_history_content_does_not_break_injection():
    steer = _steer('steer-1', ['first block', 'second block'])

    assert middleware._steer_annotations([steer])[0]['content'] == "['first block', 'second block']"
    block = middleware._build_steer_block([steer])
    assert middleware.STEER_BLOCK_OPEN in block
    assert 'first block' in block


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


def test_resumed_turn_seeds_prior_annotations_for_a_second_pause():
    earlier = middleware._steer_annotations([_steer('steer-1', 'first instruction')])
    later = middleware._steer_annotations([_steer('steer-2', 'second instruction', source='parent')])

    seed = middleware._initial_steer_annotations({'meta': {'steers': earlier}})
    assert seed == earlier

    # A second approval pause passes the union, and finalization still merges by
    # id instead of listing the earlier steer twice.
    paused = middleware._merge_steer_annotations(seed, later)
    assert [item['id'] for item in paused] == ['steer-1', 'steer-2']

    final = middleware._build_final_message_meta({'steers': earlier}, paused, ended_at=1.0)
    assert [item['id'] for item in final['steers']] == ['steer-1', 'steer-2']


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


# ---------------------------------------------------------------------------
# Wiring guards
#
# The helper tests above pin the invariants. These two parse the middleware
# source so a future edit that moves the call before the transforms, targets the
# frozen form_data, or drops the stateful guard fails loudly even though the
# helpers themselves still pass.
# ---------------------------------------------------------------------------


def _streaming_handler() -> ast.AsyncFunctionDef:
    tree = ast.parse(MIDDLEWARE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'streaming_chat_response_handler':
            return node
    raise AssertionError('streaming_chat_response_handler not found')


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ''


def _call_lines(node: ast.AST, name: str) -> list[int]:
    return [
        call.lineno for call in ast.walk(node) if isinstance(call, ast.Call) and _call_name(call) == name
    ]


def _call_nodes(node: ast.AST, name: str) -> list[ast.Call]:
    return [call for call in ast.walk(node) if isinstance(call, ast.Call) and _call_name(call) == name]


def test_injection_is_wired_after_transforms_and_before_the_provider_call():
    handler = _streaming_handler()
    inject_lines = _call_lines(handler, '_inject_pending_steers')
    assert len(inject_lines) == 1
    inject_line = inject_lines[0]

    # No context transform that can rewrite or compact the tool batch may run
    # after the steer is appended.
    for transform in ('process_messages_with_output', 'compact_messages_for_request'):
        transform_lines = _call_lines(handler, transform)
        assert transform_lines, f'{transform} call disappeared'
        assert max(transform_lines) < inject_line

    # The nearest normalization runs just before injection, not after it.
    normalize_lines = _call_lines(handler, 'normalize_messages_for_model')
    assert max(line for line in normalize_lines if line < inject_line) < inject_line
    assert _call_lines(handler, '_send_with_boundary_steers')[0] > inject_line

    # The request copy, never the frozen turn messages, is the injection target.
    inject = _call_nodes(handler, '_inject_pending_steers')[0]
    assert isinstance(inject.args[0], ast.Name) and inject.args[0].id == 'new_form_data'

    # The provider call consumes the injected copy.
    send = _call_nodes(handler, '_send_with_boundary_steers')[0]
    assert isinstance(send.args[0], ast.Lambda)
    provider_args = [argument for argument in send.args[0].body.args if isinstance(argument, ast.Name)]
    assert any(argument.id == 'new_form_data' for argument in provider_args)


def test_stateful_guard_and_annotation_wiring():
    handler = _streaming_handler()

    # Injection lives in the else branch of the stateful-replay guard.
    guards = []
    for node in ast.walk(handler):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
        if isinstance(test, ast.Call) and _call_name(test) == '_boundary_steering_enabled':
            guards.append(node)
    assert len(guards) == 1
    assert len(guards[0].body) == 1 and isinstance(guards[0].body[0], ast.Pass)
    assert _call_nodes(guards[0].orelse[0], '_inject_pending_steers')

    # Every approval pause carries the steers consumed before it, and the final
    # (and cancelled) message meta merges them back in.
    pauses = _call_nodes(handler, 'pause_for_tool_approval')
    assert len(pauses) == 2
    for pause in pauses:
        assert any(keyword.arg == 'extra_meta' for keyword in pause.keywords)

    finalizers = _call_nodes(handler, '_build_final_message_meta')
    assert len(finalizers) >= 2
    for finalizer in finalizers:
        assert any(
            isinstance(argument, ast.Name) and argument.id == 'consumed_steer_annotations'
            for argument in finalizer.args
        )


def test_steering_output_parts_are_display_only_and_follow_the_boundary():
    content = '<subagent sessionID="child-1" state="completed">\nCHILD_TIP\n</subagent>'
    annotation = {'id': 'steer-1', 'source': 'subagent', 'content': content}
    steer = {
        'id': 'steer-1',
        'content': content,
        'meta': {
            'internal': True,
            'type': 'steer',
            'source': 'subagent',
            'childID': 'child-1',
            'description': 'Child marker',
            'state': 'completed',
        },
    }

    parts = middleware._steering_output_parts(
        [annotation], [steer], SimpleNamespace(id='user-1', name='Ada')
    )

    assert len(parts) == 1
    part = parts[0]
    assert part['type'] == 'open_webui:steering'
    assert part['role'] == 'user'
    assert part['steer_id'] == 'steer-1'
    assert part['source'] == 'subagent'
    assert part['text'] == content
    assert part['user'] == {'id': 'user-1', 'name': 'Ada'}
    assert part['childID'] == 'child-1'
    assert part['description'] == 'Child marker'
    assert part['state'] == 'completed'

    # Appended after the tool result, not inside the tool batch.
    output = [
        {'type': 'function_call', 'call_id': 'call-1'},
        {'type': 'function_call_output', 'call_id': 'call-1'},
        {'type': 'message'},
    ]
    appended = middleware._append_steering_output_parts(output, parts)
    assert appended == [part]
    assert output[-1] is part
    assert [item['type'] for item in output[:3]] == [
        'function_call',
        'function_call_output',
        'message',
    ]

    # A retry/replay must not duplicate the same steer part.
    assert middleware._append_steering_output_parts(output, parts) == []

    # A dict-shaped user is supported as well (some internal callers use one).
    dict_user_parts = middleware._steering_output_parts(
        [annotation], [steer], {'id': 'user-2', 'name': 'Grace'}
    )
    assert dict_user_parts[0]['user'] == {'id': 'user-2', 'name': 'Grace'}


def test_only_delivered_annotations_become_inline_parts():
    delivered = _steer('steer-1', 'show it inline', source='subagent')
    failed = _steer('steer-2', 'failed provider, stays pending')
    annotations = middleware._steer_annotations([delivered])

    parts = middleware._steering_output_parts(
        annotations, [delivered, failed], SimpleNamespace(id='user-1', name='Ada')
    )

    # `_send_with_boundary_steers` returns annotations only for the records the
    # delivery write confirmed, so an undelivered steer never gets a part.
    assert [part['steer_id'] for part in parts] == ['steer-1']
    assert all(part['type'] == 'open_webui:steering' for part in parts)


def test_steering_part_is_wired_after_delivery_and_before_the_index_reset():
    """Pin the handler ordering: deliver -> insert part -> snapshot -> reset."""
    handler = _streaming_handler()

    send_line = _call_lines(handler, '_send_with_boundary_steers')[0]

    build = _call_nodes(handler, '_steering_output_parts')
    assert len(build) == 1
    # Built from the *delivered* annotations, never from the drained records.
    assert isinstance(build[0].args[0], ast.Name)
    assert build[0].args[0].id == 'steer_annotations'
    assert build[0].lineno > send_line

    append = _call_nodes(handler, '_append_steering_output_parts')
    assert len(append) == 1
    assert isinstance(append[0].args[0], ast.Name)
    assert append[0].args[0].id == 'output'
    assert append[0].lineno > build[0].lineno

    # The snapshot with delivered ids is emitted before the prior-output reset;
    # the new stream's output_index therefore starts after the inserted part.
    delivered_key_lines = [
        node.lineno
        for node in ast.walk(handler)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == 'delivered_steer_ids' for key in node.keys)
    ]
    assert len(delivered_key_lines) == 1
    assert delivered_key_lines[0] > append[0].lineno

    prior_reset_lines = [
        node.lineno
        for node in ast.walk(handler)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'prior_output' for target in node.targets)
        and node.lineno > append[0].lineno
    ]
    assert prior_reset_lines
    assert delivered_key_lines[0] < min(prior_reset_lines)
