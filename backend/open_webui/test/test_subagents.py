"""Backend tests for the subagent systems refactor (T3, defect W11).

Covers the behaviours T1/T2 introduced: the v2 ``subagent`` form with legacy
``delegate_task`` / ``task`` aliases, foreground and background delegation, the
depth cap, ``sessionID`` continuation, the ``subagents.depth`` / ``W5`` / ``W7``
fixes, and legacy replay plus timer batching in
``process_pending_internal_messages``.

The middleware alias translation / batch dedup live inside a large closure, so
those two paths are exercised by compiling the actual functions out of
``utils/middleware.py`` (the AST-extraction tier documented in
``.opencode/skills/openwebui-backend-testing``) and calling them with a stub
tools dict. Everything else calls the real ``delegate`` coroutine through the
fakes in ``subagent_test_utils``.
"""

from __future__ import annotations

import ast
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

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
from open_webui.tools.builtin import SUBAGENT_CANONICAL_NAME, SUBAGENT_TOOL_NAMES
from open_webui.utils import subagents
from open_webui.utils.json_codec import JSONCodec

MIDDLEWARE_PATH = Path(__file__).resolve().parents[1] / 'utils' / 'middleware.py'


# ---------------------------------------------------------------------------
# Test harness around the real `delegate` coroutine
# ---------------------------------------------------------------------------


class Harness:
    def __init__(self, monkeypatch, *, run=True, active=False, summary='Subagent finished the work.'):
        self.fake_chats = FakeChats()
        self.chat_messages = FakeChatMessages()
        self.registry = FakeTaskRegistry(run=run, active=active)
        self.completions: list[tuple] = []
        self.summary = summary
        self._db_chat = None
        self._fail_with: Exception | None = None

        async def handler(request, form_data, user=None):
            self.completions.append((request, form_data))
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

        monkeypatch.setattr(subagents, 'Chats', self.fake_chats)
        monkeypatch.setattr(subagents, 'ChatMessages', self.chat_messages)
        monkeypatch.setattr(subagents, 'Config', self.config)
        monkeypatch.setattr(subagents, 'create_task', self.registry.create_task)
        monkeypatch.setattr(subagents, 'has_active_tasks', self.registry.has_active_tasks)
        monkeypatch.setattr(subagents, 'Users', SimpleNamespace(get_user_by_id=_get_user_by_id))
        monkeypatch.setattr(subagents, 'get_async_db', lambda: fake_get_async_db(self._db_chat)())
        monkeypatch.setattr(subagents, 'sio', FakeSio(), raising=False)

        # socket.main.sio is imported locally inside process_pending; patch the
        # module attribute it will resolve.
        import open_webui.socket.main as socket_main

        self.sio = FakeSio()
        monkeypatch.setattr(socket_main, 'sio', self.sio)

    def _get_async_db(self):
        return fake_get_async_db(self._db_chat)

    def seed_parent(self, chat_id='chat-1', *, user_id='user-1', history=None, meta=None):
        chat = ChatModelLike(chat_id, user_id=user_id, history=history, meta=meta)
        self.fake_chats.chats[chat_id] = chat
        self._db_chat = chat
        self.fake_chats.messages.setdefault(chat_id, {})
        return chat

    async def seed_parent_via_insert(self, chat_id='chat-1', *, user_id='user-1', history=None, meta=None):
        from open_webui.models.chats import ChatForm

        parent_history = history or {'currentId': None, 'messages': {}}
        chat = await self.fake_chats.insert_new_chat(
            chat_id,
            user_id,
            ChatForm(chat={'id': chat_id, 'title': 'Parent', 'history': parent_history}),
            internal_meta=meta,
        )
        self._db_chat = chat
        return chat

    def find_child(self, parent_chat_id='chat-1'):
        children = [
            (chat_id, chat)
            for chat_id, chat in self.fake_chats.chats.items()
            if (chat.meta or {}).get('type') == 'subagent' and (chat.meta or {}).get('parent_chat_id') == parent_chat_id
        ]
        assert len(children) == 1, f'expected exactly one child, got {children}'
        return children[0]


class ChatModelLike:
    """Minimal mutable chat row; only the fields subagents reads/writes."""

    def __init__(self, id, *, user_id='user-1', history=None, meta=None):
        self.id = id
        self.user_id = user_id
        self.chat = {'history': history or {'currentId': None, 'messages': {}}}
        self.meta = meta or {}
        self.updated_at = 0
        self.current_message_id = self.chat['history'].get('currentId')


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


async def _wait_until(predicate, *, attempts: int = 200) -> bool:
    """Yield to the loop until `predicate()` is true (or attempts run out)."""
    for _ in range(attempts):
        if predicate():
            return True
        await asyncio.sleep(0)
    return predicate()


# ---------------------------------------------------------------------------
# 1. Alias execution (R1 translation) — extracted from the middleware closure
# ---------------------------------------------------------------------------


def _load_middleware_namespace():
    """Compile parse_tool_params/execute_tool_call/execute_subagent_call from source."""
    tree = ast.parse(MIDDLEWARE_PATH.read_text())
    wanted = {'parse_tool_params', 'execute_tool_call', 'execute_subagent_call'}
    ns = {
        'JSONCodec': JSONCodec,
        'ast': ast,
        'log': SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
        'uuid4': uuid4,
        'asyncio': asyncio,
        'SUBAGENT_CANONICAL_NAME': SUBAGENT_CANONICAL_NAME,
        'SUBAGENT_TOOL_NAMES': SUBAGENT_TOOL_NAMES,
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(MIDDLEWARE_PATH), 'exec'), ns)
            found.add(node.name)
    assert found == wanted, f'missing: {wanted - found}'
    return ns


def _tool_spec():
    return {
        'parameters': {
            'properties': {
                'description': {'type': 'string'},
                'prompt': {'type': 'string'},
                'context': {'type': 'string'},
                'file_ids': {'type': 'array'},
                'background': {'type': 'boolean'},
                'sessionID': {'type': 'string'},
                'model': {'type': 'string'},
            }
        }
    }


def _install_tool(ns, calls):
    async def _subagent_callable(**kwargs):
        calls.append(kwargs)
        return '<subagent sessionID="child-1" state="completed">\nok\n</subagent>'

    ns['tools'] = {'subagent': {'callable': _subagent_callable, 'type': 'builtin', 'spec': _tool_spec()}}
    ns['metadata'] = {'session_id': 's'}
    ns['form_data'] = {'messages': []}

    async def _get_updated_tool_function(function, extra_params):
        return function

    ns['get_updated_tool_function'] = _get_updated_tool_function
    ns['seen_subagent_sessions'] = set()
    return _subagent_callable


def _legacy_call(name, params):
    return {
        'id': 'tc-1',
        'function': {'name': name, 'arguments': json.dumps(params)},
    }


@pytest.mark.asyncio
async def test_legacy_delegate_task_alias_translates_task_argument():
    """R1: a persisted `delegate_task` call using `task` executes as `subagent`."""
    ns = _load_middleware_namespace()
    calls: list[dict] = []
    _install_tool(ns, calls)

    params, result, tool, tool_type, direct = await ns['execute_tool_call'](
        _legacy_call('delegate_task', {'task': 'summarise the repo'})
    )

    assert params == {'description': 'summarise the repo', 'prompt': 'summarise the repo'}
    assert tool is ns['tools']['subagent']
    assert result.startswith('<subagent sessionID="child-1"')
    # The callable actually executed, with the translated arguments.
    assert calls == [{'description': 'summarise the repo', 'prompt': 'summarise the repo'}]


@pytest.mark.asyncio
async def test_legacy_task_alias_translates_task_argument():
    ns = _load_middleware_namespace()
    calls: list[dict] = []
    _install_tool(ns, calls)

    params, result, tool, tool_type, direct = await ns['execute_tool_call'](
        _legacy_call('task', {'task': 'write the changelog'})
    )

    assert params == {'description': 'write the changelog', 'prompt': 'write the changelog'}
    assert calls == [{'description': 'write the changelog', 'prompt': 'write the changelog'}]


@pytest.mark.asyncio
async def test_canonical_subagent_name_is_not_rewritten():
    ns = _load_middleware_namespace()
    calls: list[dict] = []
    _install_tool(ns, calls)

    params, result, tool, tool_type, direct = await ns['execute_tool_call'](
        _legacy_call('subagent', {'description': 'label', 'prompt': 'the real prompt', 'sessionID': 'child-1'})
    )

    assert params == {'description': 'label', 'prompt': 'the real prompt', 'sessionID': 'child-1'}
    assert calls == [{'description': 'label', 'prompt': 'the real prompt', 'sessionID': 'child-1'}]


@pytest.mark.asyncio
async def test_tool_absent_is_rejected():
    """A denied (unregistered) subagent call is rejected, not silently run."""
    ns = _load_middleware_namespace()
    ns['tools'] = {}
    ns['metadata'] = {}
    ns['form_data'] = {}
    ns['seen_subagent_sessions'] = set()

    async def _never(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError('tool should not have a callable')

    ns['get_updated_tool_function'] = _never
    params, result, tool, tool_type, direct = await ns['execute_tool_call'](_legacy_call('subagent', {'prompt': 'x'}))

    assert result == 'Error: Tool "subagent" not found.'
    assert tool is None


@pytest.mark.asyncio
async def test_duplicate_sessionid_in_one_batch_is_rejected():
    """Two continuations of the same child in one gathered batch race on history."""
    ns = _load_middleware_namespace()
    calls: list[dict] = []
    _install_tool(ns, calls)

    first = await ns['execute_subagent_call'](_legacy_call('subagent', {'prompt': 'one', 'sessionID': 'child-1'}))
    second = await ns['execute_subagent_call'](_legacy_call('subagent', {'prompt': 'two', 'sessionID': 'child-1'}))

    assert calls == [{'prompt': 'one', 'sessionID': 'child-1'}]
    assert first[1].startswith('<subagent')
    assert second[1].startswith('Error: sessionID "child-1" was passed more than once')

    # A different sessionID in the same batch is allowed.
    third = await ns['execute_subagent_call'](_legacy_call('subagent', {'prompt': 'three', 'sessionID': 'child-2'}))
    assert third[1].startswith('<subagent')


@pytest.mark.asyncio
async def test_duplicate_sessionid_concurrent_batch_executes_exactly_once():
    """The same child continued twice in one `asyncio.gather` batch must not race.

    The check-and-add is synchronous, so the first coroutine to run wins before
    either can await. This is the actual fan-out shape, unlike the sequential
    case above.
    """
    ns = _load_middleware_namespace()
    calls: list[dict] = []
    _install_tool(ns, calls)

    results = await asyncio.gather(
        ns['execute_subagent_call'](_legacy_call('subagent', {'prompt': 'one', 'sessionID': 'child-1'})),
        ns['execute_subagent_call'](_legacy_call('subagent', {'prompt': 'two', 'sessionID': 'child-1'})),
    )

    executed = [result for _, result, *_ in results if str(result).startswith('<subagent')]
    rejected = [result for _, result, *_ in results if str(result).startswith('Error: sessionID')]
    # Exactly one continuation executed; the other was rejected before dispatch.
    assert calls == [{'prompt': 'one', 'sessionID': 'child-1'}]
    assert len(executed) == 1
    assert len(rejected) == 1


# ---------------------------------------------------------------------------
# 2. Foreground delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_foreground_delegation_returns_completed_envelope(monkeypatch):
    harness = Harness(monkeypatch, summary='the answer is 42')
    await harness.seed_parent_via_insert()

    result = await run_delegate(harness, description='label', prompt='what is the answer?')

    chat_id, _ = harness.find_child()
    assert result == f'<subagent sessionID="{chat_id}" state="completed">\nthe answer is 42\n</subagent>'
    # Child chat is linked to the parent and runs the requested prompt.
    child = harness.fake_chats.chats[chat_id]
    assert child.meta['internal'] is True
    assert child.meta['type'] == 'subagent'
    assert child.meta['parent_chat_id'] == 'chat-1'
    assert child.meta['delegation_id'].startswith('deleg_')
    assert child.meta['mode'] == 'foreground'
    assert harness.completions[-1][1]['messages'][-1]['content'] == 'what is the answer?'


@pytest.mark.asyncio
async def test_foreground_failure_returns_error_and_child_stays_continuable(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()
    harness._fail_with = RuntimeError('child turn exploded')

    result = await run_delegate(harness)

    assert result == 'Error: child turn exploded'
    # The child chat survives the failed turn, so its sessionID can be continued.
    child_id, _ = harness.find_child()
    assert harness.fake_chats.chats[child_id].meta['type'] == 'subagent'

    harness._fail_with = None
    harness.summary = 'recovered'
    continued = await run_delegate(harness, session_id=child_id, prompt='try again')
    assert continued == f'<subagent sessionID="{child_id}" state="completed">\nrecovered\n</subagent>'


@pytest.mark.asyncio
async def test_foreground_rejects_empty_prompt(monkeypatch):
    harness = Harness(monkeypatch)
    result = await run_delegate(harness, prompt='   ')
    assert result == 'Error: prompt must not be empty.'


@pytest.mark.asyncio
async def test_foreground_child_error_message_returns_error_string(monkeypatch):
    """A completed child turn that reports an error surfaces as an Error: string."""
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()

    async def _error_handler(request, form_data, user=None):
        harness.completions.append((request, form_data))
        await harness.fake_chats.upsert_message_to_chat_by_id_and_message_id(
            form_data['chat_id'],
            form_data['id'],
            {'id': form_data['id'], 'done': True, 'error': {'content': 'provider down'}},
        )

    harness.source.app.state.CHAT_COMPLETION_HANDLER = _error_handler
    result = await run_delegate(harness)

    assert result.startswith('Error:')
    assert 'provider down' in result


def _emitter_metadata(**overrides):
    return {
        'model_id': 'test-model',
        'session_id': 'sess',
        'user_id': 'user-1',
        'chat_id': 'chat-1',
        'message_id': 'parent-msg',
        **overrides,
    }


def _capture_events(monkeypatch):
    """Patch socket.main.get_event_emitter and collect emitted (request_info, event)."""
    events: list[tuple] = []

    async def fake_event_emitter(request_info, update_db=True):
        async def _emit(event):
            events.append((request_info, event))

        return _emit

    import open_webui.socket.main as socket_main

    monkeypatch.setattr(socket_main, 'get_event_emitter', fake_event_emitter)
    return events


@pytest.mark.asyncio
async def test_delegation_announces_child_against_parent_tool_call(monkeypatch):
    """The created child is reported against the parent's tool-call id so the row
    can link before the `<subagent ...>` result envelope exists."""
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()
    events = _capture_events(monkeypatch)

    result = await run_delegate(
        harness, description='label', tool_call_id='call-42', metadata=_emitter_metadata()
    )

    child_id, child = harness.find_child()
    assert child.meta['tool_call_id'] == 'call-42'
    assert len(events) == 1
    _, event = events[0]
    assert event['type'] == 'subagent:created'
    assert event['data']['call_id'] == 'call-42'
    assert event['data']['sessionID'] == child_id
    assert event['data']['message_id'] == 'parent-msg'
    assert not event['data']['background']
    assert f'sessionID="{child_id}"' in result


@pytest.mark.asyncio
async def test_continuation_reseeds_tool_call_id_on_child_meta(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()
    events = _capture_events(monkeypatch)

    await run_delegate(harness, tool_call_id='call-1', metadata=_emitter_metadata())
    child_id, child = harness.find_child()
    assert child.meta['tool_call_id'] == 'call-1'

    await run_delegate(
        harness,
        session_id=child_id,
        prompt='do more',
        tool_call_id='call-2',
        metadata=_emitter_metadata(),
    )
    assert harness.fake_chats.chats[child_id].meta['tool_call_id'] == 'call-2'
    assert [e['data']['call_id'] for _, e in events] == ['call-1', 'call-2']


@pytest.mark.asyncio
async def test_delegation_without_tool_call_id_emits_nothing(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()
    events = _capture_events(monkeypatch)

    await run_delegate(harness)

    assert events == []


# ---------------------------------------------------------------------------
# 3. Background delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_delegation_returns_json_handle(monkeypatch):
    harness = Harness(monkeypatch, run=False)
    await harness.seed_parent_via_insert()

    result = await run_delegate(harness, background=True)

    handle = json.loads(result)
    child_id, _ = harness.find_child()
    assert set(handle) == {'sessionID', 'status', 'output'}
    assert handle['sessionID'] == child_id
    assert handle['status'] == 'running'
    assert child_id in handle['output']
    # The handle carries the full three-sentence non-polling guidance.
    assert 'You will be notified automatically when it finishes' in handle['output']
    assert 'DO NOT sleep, poll for progress' in handle['output']
    assert 'Work on non-overlapping tasks' in handle['output']
    # No child turn ran yet: the captured coroutine was not awaited.
    assert harness.completions == []


@pytest.mark.asyncio
async def test_background_completion_injects_legacy_compatible_synthetic(monkeypatch):
    harness = Harness(monkeypatch, active=True, summary='background result')
    await harness.seed_parent_via_insert()

    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']

    # Let the captured background turn run to completion.
    _, task = harness.registry.created[-1]
    await task

    messages = harness._db_chat.chat['history']['messages']
    injected = [m for m in messages.values() if (m.get('meta') or {}).get('type') == 'subagent']
    assert len(injected) == 1
    meta = injected[0]['meta']
    assert meta['source'] == 'subagent'
    assert meta['childID'] == child_id
    assert meta['state'] == 'completed'
    # opencode parity: the label rides along so the row can show it.
    assert meta['description'] == 'a task'
    # Legacy handle fields remain so old chats keep matching.
    assert meta['delegation_id'].startswith('deleg_')
    assert meta['subagent_chat_id'] == child_id
    assert meta['status'] == 'pending'
    assert injected[0]['content'].startswith(f'<subagent sessionID="{child_id}" state="completed">')


@pytest.mark.asyncio
async def test_background_failure_preserves_child_id_and_error_state(monkeypatch):
    harness = Harness(monkeypatch, active=True)
    await harness.seed_parent_via_insert()
    harness._fail_with = RuntimeError('background boom')

    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    _, task = harness.registry.created[-1]
    await task  # run_background swallows the failure into an error result

    messages = harness._db_chat.chat['history']['messages']
    injected = [m for m in messages.values() if (m.get('meta') or {}).get('type') == 'subagent']
    assert len(injected) == 1
    assert injected[0]['meta']['childID'] == child_id
    assert injected[0]['meta']['state'] == 'error'
    assert injected[0]['meta']['subagent_chat_id'] == child_id
    assert 'state="error"' in injected[0]['content']
    assert 'background boom' in injected[0]['content']


@pytest.mark.asyncio
async def test_background_result_registers_synthesis_on_parent_chat(monkeypatch):
    """A finished background child registers its synthesis under the parent."""
    harness = Harness(monkeypatch, active=False, summary='background result')
    await harness.seed_parent_via_insert()

    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    assert [task_id for task_id, _ in harness.registry.created] == [child_id]

    # Let the background turn run; it drains its own result and registers the
    # synthesis as a parent-chat task rather than streaming it inline.
    _, child_task = harness.registry.created[-1]
    await child_task

    parent_tasks = [task for task_id, task in harness.registry.created if task_id == 'chat-1']
    assert len(parent_tasks) == 1

    await parent_tasks[0]
    assert harness.completions[-1][1]['chat_id'] == 'chat-1'


# ---------------------------------------------------------------------------
# Background pyodide code_interpreter liveness gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('pool', 'kept'),
    [
        ({}, False),
        ({'sess': {'id': 'user-1'}}, True),
        ({'sess': {'id': 'other-user'}}, False),
    ],
)
async def test_background_code_interpreter_gated_on_live_owned_session(monkeypatch, pool, kept):
    """Pyodide runs in the user's browser: keep code only when that session lives.

    With `active=True` the child completion does not trigger a synthesis, so the
    captured turn is the child's and carries the gated features.
    """
    import open_webui.socket.main as socket_main

    monkeypatch.setattr(socket_main, 'SESSION_POOL', pool)
    harness = Harness(monkeypatch, active=True)
    await harness.seed_parent_via_insert()

    metadata = {
        'model_id': 'test-model',
        'session_id': 'sess',
        'features': {'code_interpreter': True},
    }
    await run_delegate(harness, background=True, metadata=metadata)
    _, child_task = harness.registry.created[-1]
    await child_task

    features = harness.completions[-1][1]['features']
    assert bool(features.get('code_interpreter')) is kept


@pytest.mark.asyncio
async def test_foreground_code_interpreter_ignores_session_liveness(monkeypatch):
    """Foreground runs execute on the server path; the session gate must not apply."""
    import open_webui.socket.main as socket_main

    monkeypatch.setattr(socket_main, 'SESSION_POOL', {})
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()

    metadata = {
        'model_id': 'test-model',
        'session_id': 'sess',
        'features': {'code_interpreter': True},
    }
    await run_delegate(harness, background=False, metadata=metadata)

    assert harness.completions[-1][1]['features'].get('code_interpreter') is True


# ---------------------------------------------------------------------------
# process_pending_internal_messages expansion / timer coexistence (items 3, 9)
# ---------------------------------------------------------------------------


def _pending_subagent_message(
    message_id, *, child_id='child-1', legacy=True, parent_id='parent-msg', status=None, model='test-model'
):
    meta = {'internal': True, 'type': 'subagent', 'source': 'subagent', 'state': 'completed'}
    if legacy:
        # Old shape: no childID/state, only the old handle fields.
        meta = {'internal': True, 'type': 'subagent', 'delegation_id': 'deleg_old', 'subagent_chat_id': child_id}
    else:
        meta['childID'] = child_id
    if status is not None:
        meta['status'] = status
    return {
        'id': message_id,
        'parentId': parent_id,
        'childrenIds': [],
        'role': 'user',
        'content': f'<subagent sessionID="{child_id}" state="completed">\ndone\n</subagent>',
        'model': model,
        'meta': meta,
        'timestamp': int(time.time()),
    }


def _pending_timer_message(message_id, *, timer_id='timer-1', parent_id='parent-msg'):
    return {
        'id': message_id,
        'parentId': parent_id,
        'childrenIds': [],
        'role': 'user',
        'content': 'remind me',
        'model': 'test-model',
        'meta': {'internal': True, 'type': 'timer', 'timer_id': timer_id},
        'timestamp': int(time.time()),
    }


def _parent_history(*pending, parent_id='parent-msg'):
    messages = {parent_id: {'id': parent_id, 'role': 'assistant', 'childrenIds': [], 'content': 'hi'}}
    for message in pending:
        messages[message['id']] = message
    return {'currentId': parent_id, 'messages': messages}


@pytest.mark.asyncio
async def test_process_pending_expands_subagent_message(monkeypatch):
    harness = Harness(monkeypatch)
    pending = _pending_subagent_message('pending-1', legacy=False)
    parent = harness.seed_parent(history=_parent_history(pending))

    run = {'model_id': 'test-model', 'system_prompt': 'sys', 'session_id': 's'}
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', run)
    # The synthesis is registered as a tracked task, not awaited inline; run it
    # to completion before asserting on the turn it produced.
    await harness.registry.created[-1][1]

    # The single pending message is reused (status not 'pending') and expanded
    # in place into a fresh user/assistant pair.
    messages = parent.chat['history']['messages']
    assert messages['pending-1']['meta']['type'] == 'subagent'
    assert messages['pending-1']['meta']['childID'] == 'child-1'
    assert messages['pending-1']['meta']['state'] == 'completed'
    assert messages['pending-1']['parentId'] == 'parent-msg'
    assert messages['pending-1']['childrenIds']
    assistant_id = messages['pending-1']['childrenIds'][0]
    assert messages[assistant_id]['role'] == 'assistant'
    assert parent.chat['history']['currentId'] == assistant_id
    # The completion handler ran on the expanded turn, and the synthesis turn is
    # marked internal so it cannot re-advertise the subagent tool (W10).
    assert harness.completions[-1][1]['id'] == assistant_id
    assert harness.completions[-1][0].state.internal is True


@pytest.mark.asyncio
async def test_process_pending_replays_old_shape_subagent_message(monkeypatch):
    harness = Harness(monkeypatch)
    pending = _pending_subagent_message('pending-old', legacy=True)
    parent = harness.seed_parent(history=_parent_history(pending))

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})

    meta = parent.chat['history']['messages']['pending-old']['meta']
    # Legacy handle fields are still recognised and carried forward.
    assert meta['delegation_id'] == 'deleg_old'
    assert meta['subagent_chat_id'] == 'child-1'
    assert meta['childID'] == 'child-1'


@pytest.mark.asyncio
async def test_timer_batching_coexists_with_subagent_expansion(monkeypatch):
    harness = Harness(monkeypatch)
    timer = _pending_timer_message('timer-msg')
    subagent = _pending_subagent_message('sub-msg', legacy=False, child_id='child-9')
    parent = harness.seed_parent(history=_parent_history(timer, subagent))
    run = {'model_id': 'test-model'}

    # The timer is first: it is batched and expanded on its own.
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', run)
    messages = parent.chat['history']['messages']
    assert messages['timer-msg']['meta']['type'] == 'timer'
    assert messages['timer-msg']['meta']['timer_id'] == 'timer-1'

    # The subagent pending message is still there and expands on the next drain.
    assert messages['sub-msg']['meta']['type'] == 'subagent'
    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', run)
    assert messages['sub-msg']['meta']['type'] == 'subagent'
    assert messages['sub-msg']['meta']['childID'] == 'child-9'


@pytest.mark.asyncio
async def test_multiple_subagent_results_batch_into_one_synthesis_turn(monkeypatch):
    """Two completed subagents sharing a parent/model coalesce into one turn."""
    harness = Harness(monkeypatch)
    first = _pending_subagent_message('sub-1', legacy=False, child_id='child-1')
    second = _pending_subagent_message('sub-2', legacy=False, child_id='child-2')
    parent = harness.seed_parent(history=_parent_history(first, second))

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    # The batch's synthesis is a tracked task; run it before counting turns.
    await harness.registry.created[-1][1]

    # Exactly one synthesis turn ran for the batch.
    assert len(harness.completions) == 1
    messages = parent.chat['history']['messages']
    # Both pending records were consumed and replaced by one combined user turn.
    assert 'sub-1' not in messages
    assert 'sub-2' not in messages
    combined = next(
        m for m in messages.values() if m.get('role') == 'user' and m.get('meta', {}).get('type') == 'subagent'
    )
    assert combined['meta']['childIDs'] == ['child-1', 'child-2']
    # A single state is only recorded when unambiguous; two children means the
    # plural key is used and `state` is omitted.
    assert 'state' not in combined['meta']
    assert combined['content'].count('<subagent') == 2
    assert messages['parent-msg']['childrenIds'] == [combined['id']]


def _forked_parent_history(*, tip_id='followup-assistant', extra_pending=None, current_id=None):
    """The fork repro: a completed turn, a user followup, then an in-flight tip.

    Mirrors the production bug where a background result was anchored to the
    newest *completed* assistant (`completed-assistant`) instead of the user's
    active branch tip (`followup-assistant`, persisted with ``done: False``).
    """
    messages = {
        'root-user': {
            'id': 'root-user',
            'parentId': None,
            'childrenIds': ['completed-assistant'],
            'role': 'user',
            'content': 'first',
            'timestamp': 1,
        },
        'completed-assistant': {
            'id': 'completed-assistant',
            'parentId': 'root-user',
            'childrenIds': ['followup-user'],
            'role': 'assistant',
            'content': 'done',
            'done': True,
            'timestamp': 2,
        },
        'followup-user': {
            'id': 'followup-user',
            'parentId': 'completed-assistant',
            'childrenIds': [tip_id],
            'role': 'user',
            'content': 'second',
            'timestamp': 3,
        },
        tip_id: {
            'id': tip_id,
            'parentId': 'followup-user',
            'childrenIds': [],
            'role': 'assistant',
            'content': '',
            'done': False,
            'timestamp': 4,
        },
    }
    if extra_pending is not None:
        messages[extra_pending['id']] = extra_pending
    return {'currentId': current_id if current_id is not None else tip_id, 'messages': messages}


def _reachable_ids(history, message_id):
    """The client's `createMessagesList`: walk parentId up from `message_id`."""
    ids = []
    seen = set()
    while message_id is not None and message_id in history['messages'] and message_id not in seen:
        seen.add(message_id)
        ids.append(message_id)
        message_id = history['messages'][message_id].get('parentId')
    return list(reversed(ids))


@pytest.mark.asyncio
async def test_background_result_anchors_to_current_tip_not_completed_assistant(monkeypatch):
    """A result must not fork off the newest completed assistant.

    Regression: the user's followup turn was still streaming (`done: False`) when
    a background child finished, so a max-timestamp-over-completed-assistants
    anchor skipped it and attached the result as a sibling, orphaning the
    followup from the rendered branch.
    """
    harness = Harness(monkeypatch, active=True, summary='background result')
    await harness.seed_parent_via_insert(history=_forked_parent_history())

    handle = json.loads(await run_delegate(harness, background=True))
    child_id = handle['sessionID']
    _, task = harness.registry.created[-1]
    await task

    messages = harness._db_chat.chat['history']['messages']
    injected = [m for m in messages.values() if (m.get('meta') or {}).get('type') == 'subagent']
    assert len(injected) == 1
    # Anchored on the live tip, not the completed assistant.
    assert injected[0]['parentId'] == 'followup-assistant'
    assert injected[0]['meta']['childID'] == child_id
    assert injected[0]['meta']['status'] == 'pending'
    # The completed assistant did not gain a sibling child bypassing the tip.
    assert messages['completed-assistant']['childrenIds'] == ['followup-user']


@pytest.mark.asyncio
async def test_drain_reanchors_deferred_result_to_advanced_tip(monkeypatch):
    """A deferred (pending) result is re-anchored at the tip when it is drained.

    Mirrors opencode's promote step: the completion is appended at the session
    tip at delivery time, so a turn that finished between enqueue and drain does
    not leave the result forked off an older node.
    """
    harness = Harness(monkeypatch)
    # The result was enqueued while the user's turn was active, so it carries a
    # stale anchor (`completed-assistant`) and status `pending`.
    pending = _pending_subagent_message(
        'pending-1', child_id='child-1', legacy=False, parent_id='completed-assistant', status='pending'
    )
    history = _forked_parent_history(extra_pending=pending, current_id='followup-assistant')
    parent = harness.seed_parent(history=history)
    parent.chat['history']['messages']['completed-assistant']['childrenIds'].append('pending-1')

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})
    await harness.registry.created[-1][1]

    messages = parent.chat['history']['messages']
    combined = next(
        m for m in messages.values() if m.get('role') == 'user' and (m.get('meta') or {}).get('type') == 'subagent'
    )
    # Re-anchored to the tip, not the stale `completed-assistant`.
    assert combined['parentId'] == 'followup-assistant'
    assert messages['completed-assistant']['childrenIds'] == ['followup-user']
    assert 'pending-1' not in messages
    # The followup branch stays reachable from the new tip.
    reachable = set(_reachable_ids(parent.chat['history'], parent.chat['history']['currentId']))
    assert {'completed-assistant', 'followup-user', 'followup-assistant', combined['id']} <= reachable


@pytest.mark.asyncio
async def test_two_background_results_stay_siblings_and_both_drain(monkeypatch):
    """Two results finishing while a turn is active must batch, not chain.

    Regression: if the second result anchored *onto* the first pending result
    (descending into it), the drain's no-children filter would exclude the first
    and silently drop its envelope.
    """
    harness = Harness(monkeypatch, active=True)
    await harness.seed_parent_via_insert(history=_forked_parent_history())

    # First result (no active task yet? keep active so it stays `pending`).
    await run_delegate(harness, background=True)
    _, first_task = harness.registry.created[-1]
    await first_task
    # Second result, same parent chat.
    await run_delegate(harness, background=True)
    _, second_task = harness.registry.created[-1]
    await second_task

    messages = harness._db_chat.chat['history']['messages']
    injected = [m for m in messages.values() if (m.get('meta') or {}).get('type') == 'subagent']
    assert len(injected) == 2
    # Both are siblings under the live tip; neither chains onto the other.
    assert {m['parentId'] for m in injected} == {'followup-assistant'}
    assert all(m['meta']['status'] == 'pending' for m in injected)
    assert messages['followup-assistant']['childrenIds'] == [m['id'] for m in injected]
    assert messages['completed-assistant']['childrenIds'] == ['followup-user']


def test_resolve_history_tip_handles_missing_and_empty():
    """The shared anchor helper: tip via childrenIds, then latest leaf, then fallback."""
    from open_webui.utils.misc import resolve_history_tip

    messages = _forked_parent_history()['messages']
    assert resolve_history_tip(messages, 'completed-assistant') == 'followup-assistant'
    assert resolve_history_tip(messages, 'followup-user') == 'followup-assistant'
    # Unknown tip: newest leaf wins.
    assert resolve_history_tip(messages, 'does-not-exist') == 'followup-assistant'
    # Empty history: fall back to the caller's anchor.
    assert resolve_history_tip({}, 'x', 'fallback') == 'fallback'
    assert resolve_history_tip(None, None, 'fallback') == 'fallback'


def test_resolve_history_tip_skips_pending_internal_nodes():
    """Undelivered results must not be descended into (they must stay siblings)."""
    from open_webui.utils.misc import is_pending_internal_message, resolve_history_tip

    messages = _forked_parent_history()['messages']
    messages['pending-1'] = {
        'id': 'pending-1',
        'parentId': 'followup-assistant',
        'childrenIds': [],
        'role': 'user',
        'content': '<subagent/>',
        'meta': {'internal': True, 'type': 'subagent', 'status': 'pending'},
        'timestamp': 5,
    }
    messages['followup-assistant']['childrenIds'] = ['pending-1']
    assert is_pending_internal_message(messages['pending-1']) is True
    # With the predicate, the pending result is skipped and the anchor stays the tip.
    assert (
        resolve_history_tip(messages, 'followup-assistant', blocked=is_pending_internal_message) == 'followup-assistant'
    )
    # A delivered synthesis user turn (has a child) is still traversed.
    messages['pending-1']['childrenIds'] = ['synthesis-assistant']
    messages['synthesis-assistant'] = {
        'id': 'synthesis-assistant',
        'parentId': 'pending-1',
        'childrenIds': [],
        'role': 'assistant',
        'content': '',
        'done': True,
        'timestamp': 6,
    }
    assert is_pending_internal_message(messages['pending-1']) is False
    assert resolve_history_tip(messages, 'followup-assistant', blocked=is_pending_internal_message) == (
        'synthesis-assistant'
    )


def _active_flags(harness) -> list[bool]:
    return [
        event['data']['data']['active']
        for _, event in harness.sio.emitted
        if event.get('data', {}).get('type') == 'chat:active'
    ]


@pytest.mark.asyncio
async def test_pending_synthesis_registered_under_parent_chat_id(monkeypatch):
    """The synthesis runs as a tracked parent task, not inline (fixes refresh)."""
    harness = Harness(monkeypatch)
    pending = _pending_subagent_message('pending-1', legacy=False)
    harness.seed_parent(history=_parent_history(pending))

    await subagents.process_pending_internal_messages(
        harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'}
    )

    # Registered under the PARENT chat id and still pending: the turn is not run
    # inline, so the client can reconcile the task and Stop can cancel it.
    assert [task_id for task_id, _ in harness.registry.created] == ['chat-1']
    assert harness.completions == []

    # The denormalized tip column must advance with the in-JSON tip; otherwise
    # GET /chats/{id} (and `loadChat`) restores the pre-synthesis tip and the
    # streamed answer stays hidden until a later save.
    parent = harness.fake_chats.chats['chat-1']
    assert parent.current_message_id == parent.chat['history']['currentId']
    assert parent.current_message_id != 'parent-msg'

    _, synthesis_task = harness.registry.created[-1]
    await synthesis_task

    # The turn actually ran once its task was awaited.
    assert len(harness.completions) == 1
    assert harness.completions[-1][0].state.internal is True
    # Active is flipped on for the stream, then off once the drain settles.
    assert await _wait_until(lambda: _active_flags(harness) == [True, False])


@pytest.mark.asyncio
async def test_synthesis_summary_accepts_responses_text_parts(monkeypatch):
    """Responses-API message parts are typed `text`, not `output_text`.

    ``handle_responses_streaming_event`` creates ``{'type': 'text'}`` parts (see
    ``middleware.py``), so the summary extractor must not require ``output_text``
    or a Responses-API child would fall back to interim `content` (or empty).
    """
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()

    async def _responses_style_handler(request, form_data, user=None):
        harness.completions.append((request, form_data))
        await harness.fake_chats.upsert_message_to_chat_by_id_and_message_id(
            form_data['chat_id'],
            form_data['id'],
            {
                'id': form_data['id'],
                'done': True,
                # Accumulated narration from earlier tool-loop rounds.
                'content': 'Interim narration.\nfinal answer',
                'output': [
                    {
                        'type': 'message',
                        'content': [{'type': 'text', 'text': 'final answer'}],
                    }
                ],
            },
        )

    harness.source.app.state.CHAT_COMPLETION_HANDLER = _responses_style_handler

    result = await run_delegate(harness)

    child_id, _ = harness.find_child()
    assert result == f'<subagent sessionID="{child_id}" state="completed">\nfinal answer\n</subagent>'
    assert 'Interim narration' not in result


@pytest.mark.asyncio
async def test_synthesis_summary_uses_only_last_output_message(monkeypatch):
    """Interim tool-loop narration must not leak into the parent summary."""
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()

    async def _multi_round_handler(request, form_data, user=None):
        harness.completions.append((request, form_data))
        await harness.fake_chats.upsert_message_to_chat_by_id_and_message_id(
            form_data['chat_id'],
            form_data['id'],
            {
                'id': form_data['id'],
                'done': True,
                # `content` accumulates every tool-loop round on the production
                # streaming path (middleware.update_assistant_message_from_stream).
                # The summary must ignore it in favour of the last output message.
                'content': "I've made substantial progress but am not done.\nDone.",
                'output': [
                    {
                        'type': 'message',
                        'content': [
                            {'type': 'output_text', 'text': "I've made substantial progress but am not done."}
                        ],
                    },
                    {'type': 'function_call', 'name': 'read_file'},
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Done.'}]},
                ],
            },
        )

    harness.source.app.state.CHAT_COMPLETION_HANDLER = _multi_round_handler

    result = await run_delegate(harness)

    child_id, _ = harness.find_child()
    assert result == f'<subagent sessionID="{child_id}" state="completed">\nDone.\n</subagent>'
    assert 'substantial progress' not in result


# ---------------------------------------------------------------------------
# 4. Depth cap
# ---------------------------------------------------------------------------


def _nested_child(harness, child_id, parent_id):
    harness.fake_chats.chats[child_id] = ChatModelLike(
        child_id,
        meta={'internal': True, 'type': 'subagent', 'parent_chat_id': parent_id},
    )
    harness.fake_chats.messages.setdefault(child_id, {})
    return harness.fake_chats.chats[child_id]


@pytest.mark.asyncio
async def test_default_depth_rejects_first_level_nested_delegation(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    # A child of root has chain depth 1.
    _nested_child(harness, 'child-1', 'root')

    result = await run_delegate(harness, parent_chat_id='child-1')

    assert result.startswith('Error: subagent depth limit reached (1)')
    # No grandchild was created: the only chats are the root and its child.
    assert set(harness.fake_chats.chats) == {'root', 'child-1'}


@pytest.mark.asyncio
async def test_root_delegation_is_allowed_under_default_depth(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')

    result = await run_delegate(harness, parent_chat_id='root')
    assert result.startswith('<subagent sessionID="')


@pytest.mark.asyncio
async def test_zero_depth_blocks_root_delegation(monkeypatch):
    """`subagents.depth = 0` disables delegation entirely (depth 0 on the root)."""
    harness = Harness(monkeypatch)
    harness.config.values['subagents.depth'] = 0
    await harness.seed_parent_via_insert('root')

    result = await run_delegate(harness, parent_chat_id='root')

    assert result.startswith('Error: subagent depth limit reached (0)')
    assert set(harness.fake_chats.chats) == {'root'}


@pytest.mark.asyncio
async def test_depth_walk_is_cycle_safe(monkeypatch):
    harness = Harness(monkeypatch)
    harness.fake_chats.chats['a'] = ChatModelLike(
        'a', meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'b'}
    )
    harness.fake_chats.chats['b'] = ChatModelLike(
        'b', meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'a'}
    )

    depth = await asyncio.wait_for(subagents._subagent_chain_depth('a'), timeout=1.0)
    assert depth == 2


@pytest.mark.asyncio
async def test_negative_depth_means_unlimited(monkeypatch):
    harness = Harness(monkeypatch)
    harness.config.values['subagents.depth'] = -1
    await harness.seed_parent_via_insert('root')
    _nested_child(harness, 'child-1', 'root')

    result = await run_delegate(harness, parent_chat_id='child-1')

    assert result.startswith('<subagent sessionID="')
    # A grandchild was created under the nested child.
    assert any((chat.meta or {}).get('parent_chat_id') == 'child-1' for chat in harness.fake_chats.chats.values())


# ---------------------------------------------------------------------------
# 4b. Model precedence (#27598): explicit > admin default > parent chat model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_precedence_falls_back_to_parent_model(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert()

    await run_delegate(harness)

    assert harness.completions[-1][1]['model'] == 'test-model'


@pytest.mark.asyncio
async def test_model_precedence_admin_default_beats_parent(monkeypatch):
    harness = Harness(monkeypatch)
    harness.config.values['subagents.model'] = 'admin-model'
    await harness.seed_parent_via_insert()

    await run_delegate(harness)

    assert harness.completions[-1][1]['model'] == 'admin-model'


@pytest.mark.asyncio
async def test_model_precedence_explicit_beats_admin(monkeypatch):
    harness = Harness(monkeypatch)
    harness.config.values['subagents.model'] = 'admin-model'
    await harness.seed_parent_via_insert()

    await run_delegate(harness, model='explicit-model')

    assert harness.completions[-1][1]['model'] == 'explicit-model'


@pytest.mark.asyncio
async def test_explicit_model_must_be_available(monkeypatch):
    harness = Harness(monkeypatch)
    harness.source.app.state.MODELS = {'other-model': {}}

    result = await run_delegate(harness, model='missing-model')

    assert result == 'Error: model "missing-model" is not available.'


# ---------------------------------------------------------------------------
# 5. sessionID continuation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessionid_rejects_wrong_owner(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    harness.fake_chats.chats['child-1'] = ChatModelLike(
        'child-1',
        user_id='other-user',
        meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'root'},
    )

    result = await run_delegate(harness, parent_chat_id='root', session_id='child-1')

    assert result == 'Error: sessionID "child-1" was not found.'


@pytest.mark.asyncio
async def test_sessionid_rejects_non_subagent_chat(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    harness.fake_chats.chats['other'] = ChatModelLike('other', meta={})

    result = await run_delegate(harness, parent_chat_id='root', session_id='other')

    assert result == 'Error: sessionID "other" is not a subagent session.'


@pytest.mark.asyncio
async def test_sessionid_rejects_mismatched_parent(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    harness.fake_chats.chats['child-1'] = ChatModelLike(
        'child-1', meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'some-other-chat'}
    )

    result = await run_delegate(harness, parent_chat_id='root', session_id='child-1')

    assert result == 'Error: sessionID "child-1" does not belong to this chat.'


@pytest.mark.asyncio
async def test_sessionid_continuation_keeps_child_history_consistent(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    # Existing child with one completed exchange; currentId is the assistant tip.
    child = ChatModelLike(
        'child-1',
        meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'root'},
    )
    child.chat['history'] = {
        'currentId': 'old-assistant',
        'messages': {
            'old-user': {
                'id': 'old-user',
                'parentId': None,
                'childrenIds': ['old-assistant'],
                'role': 'user',
                'content': 'first',
                'timestamp': 1,
            },
            'old-assistant': {
                'id': 'old-assistant',
                'parentId': 'old-user',
                'childrenIds': [],
                'role': 'assistant',
                'content': 'ok',
                'done': True,
                'timestamp': 2,
            },
        },
    }
    harness.fake_chats.chats['child-1'] = child

    result = await run_delegate(harness, parent_chat_id='root', session_id='child-1', prompt='again')

    assert result.startswith('<subagent sessionID="child-1"')
    messages = child.chat['history']['messages']
    # The old tip gained the new user turn; the new user gained the assistant.
    new_user_id = next(m['id'] for m in messages.values() if m.get('role') == 'user' and m.get('content') == 'again')
    new_assistant_id = messages[new_user_id]['childrenIds'][0]
    assert messages[new_user_id]['parentId'] == 'old-assistant'
    assert 'old-assistant' in messages
    assert new_user_id in messages['old-assistant']['childrenIds']
    assert child.chat['history']['currentId'] == new_assistant_id
    assert messages[new_assistant_id]['parentId'] == new_user_id
    # The continuation turn carried the prior exchange into the request.
    sent = harness.completions[-1][1]['messages']
    assert [m['role'] for m in sent[-3:]] == ['user', 'assistant', 'user']
    assert sent[-1]['content'] == 'again'


@pytest.mark.asyncio
async def test_sessionid_continuation_inherits_child_files(monkeypatch):
    """A continued turn keeps the child's workspace unless files are re-selected."""
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    files = [{'id': 'f1', 'url': '/api/v1/files/f1'}]
    child = ChatModelLike('child-1', meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'root'})
    child.chat['history'] = {
        'currentId': 'old-assistant',
        'messages': {
            'old-user': {
                'id': 'old-user',
                'parentId': None,
                'childrenIds': ['old-assistant'],
                'role': 'user',
                'content': 'first',
                'files': files,
                'timestamp': 1,
            },
            'old-assistant': {
                'id': 'old-assistant',
                'parentId': 'old-user',
                'childrenIds': [],
                'role': 'assistant',
                'content': 'ok',
                'done': True,
                'timestamp': 2,
            },
        },
    }
    harness.fake_chats.chats['child-1'] = child

    result = await run_delegate(harness, parent_chat_id='root', session_id='child-1', prompt='again')

    assert result.startswith('<subagent sessionID="child-1"')
    assert harness.completions[-1][1]['files'] == files
    new_user_id = next(
        m['id']
        for m in child.chat['history']['messages'].values()
        if m.get('role') == 'user' and m.get('content') == 'again'
    )
    assert child.chat['history']['messages'][new_user_id]['files'] == files


@pytest.mark.asyncio
async def test_sessionid_continuation_rejects_running_child(monkeypatch):
    """Continuing a child whose turn is still running must not race its history."""
    harness = Harness(monkeypatch)
    await harness.seed_parent_via_insert('root')
    child = ChatModelLike('child-1', meta={'internal': True, 'type': 'subagent', 'parent_chat_id': 'root'})
    child.chat['history'] = {
        'currentId': 'running-assistant',
        'messages': {
            'old-user': {
                'id': 'old-user',
                'parentId': None,
                'childrenIds': ['running-assistant'],
                'role': 'user',
                'content': 'first',
                'timestamp': 1,
            },
            'running-assistant': {
                'id': 'running-assistant',
                'parentId': 'old-user',
                'childrenIds': [],
                'role': 'assistant',
                'content': '',
                'done': False,
                'timestamp': 2,
            },
        },
    }
    harness.fake_chats.chats['child-1'] = child

    result = await run_delegate(harness, parent_chat_id='root', session_id='child-1', prompt='again')

    assert result == 'Error: sessionID "child-1" is still running; wait for it to finish before continuing.'
    # The child was not mutated and no child turn was started.
    assert set(child.chat['history']['messages']) == {'old-user', 'running-assistant'}
    assert harness.completions == []


# ---------------------------------------------------------------------------
# 8. W5 / W7
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_iterations_minus_one_reaches_loop_guard_as_unlimited(monkeypatch):
    harness = Harness(monkeypatch)
    harness.config.values['subagents.max_iterations'] = -1
    await harness.seed_parent_via_insert()

    await run_delegate(harness)

    child_request = harness.completions[-1][0]
    # W5: -1 maps to None, which the middleware loop guard treats as unlimited.
    assert child_request.state.max_tool_call_iterations is None
    assert child_request.state.internal is True


@pytest.mark.asyncio
async def test_positive_max_iterations_is_forwarded(monkeypatch):
    harness = Harness(monkeypatch)
    harness.config.values['subagents.max_iterations'] = 7
    await harness.seed_parent_via_insert()

    await run_delegate(harness)

    assert harness.completions[-1][0].state.max_tool_call_iterations == 7


@pytest.mark.asyncio
async def test_bounded_wait_returns_after_timeout(monkeypatch):
    calls = {'n': 0}

    async def _always_active(redis, chat_id):
        calls['n'] += 1
        return True

    monkeypatch.setattr(subagents, 'has_active_tasks', _always_active)
    started = time.monotonic()
    await subagents._wait_for_active_tasks(None, 'chat-1', timeout=0.05)
    elapsed = time.monotonic() - started

    # It returned instead of hanging, after at least one poll plus the interval
    # sleep before the deadline re-check.
    assert elapsed < 1.0
    assert calls['n'] >= 2


@pytest.mark.asyncio
async def test_pending_message_not_dropped_when_tasks_still_active(monkeypatch):
    harness = Harness(monkeypatch)
    pending = _pending_subagent_message('pending-1', legacy=False)
    parent = harness.seed_parent(history=_parent_history(pending))

    async def _always_active(redis, chat_id):
        return True

    # Skip the (already separately tested) bounded wait so the recheck runs now.
    async def _no_wait(redis, chat_id, timeout=0):
        return None

    monkeypatch.setattr(subagents, 'has_active_tasks', _always_active)
    monkeypatch.setattr(subagents, '_wait_for_active_tasks', _no_wait)

    await subagents.process_pending_internal_messages(harness.source, 'chat-1', 'user-1', {'model_id': 'test-model'})

    # The pending record is untouched and no synthesis turn was started.
    assert 'pending-1' in parent.chat['history']['messages']
    assert parent.chat['history']['messages']['pending-1']['meta'].get('status') != 'completed'
    assert harness.chat_messages.upserted == []
    assert harness.completions == []


# ---------------------------------------------------------------------------
# W3: `_parent_locks` serializes the parent-history write and is popped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_lock_is_popped_after_drain():
    """The lock must not accumulate per parent chat after use (W3)."""
    assert 'lock-chat' not in subagents._parent_locks
    try:
        async with subagents._parent_lock_scope('lock-chat'):
            assert 'lock-chat' in subagents._parent_locks
        assert 'lock-chat' not in subagents._parent_locks
    finally:
        subagents._parent_locks.pop('lock-chat', None)


@pytest.mark.asyncio
async def test_parent_lock_serializes_concurrent_writers():
    """Two writers on the same parent must not interleave their critical section."""
    order: list[tuple[str, int]] = []

    async def worker(n: int):
        async with subagents._parent_lock_scope('lock-chat'):
            order.append(('start', n))
            await asyncio.sleep(0)
            order.append(('end', n))

    try:
        await asyncio.gather(worker(1), worker(2))
    finally:
        subagents._parent_locks.pop('lock-chat', None)

    # Whichever ran first finished before the other started.
    assert order in (
        [('start', 1), ('end', 1), ('start', 2), ('end', 2)],
        [('start', 2), ('end', 2), ('start', 1), ('end', 1)],
    )
    # Drained and therefore dropped.
    assert 'lock-chat' not in subagents._parent_locks


# ---------------------------------------------------------------------------
# Builtin `subagent` docstring / spec refinement (opencode v2 parity)
# ---------------------------------------------------------------------------


def test_subagent_docstring_refinement_parses_without_model():
    """The refined docstring parses cleanly and drops the `model` parameter."""
    from open_webui.tools.builtin import subagent
    from open_webui.utils.tools import get_builtin_tool_spec, parse_description, parse_docstring

    description = parse_description(subagent.__doc__)
    assert '### Returns' in description
    assert 'When to Use' in description

    params = parse_docstring(subagent.__doc__)
    assert 'model' not in params
    assert set(params) == {'description', 'prompt', 'context', 'file_ids', 'background', 'sessionID'}

    spec = get_builtin_tool_spec(subagent)
    assert 'model' not in spec['parameters']['properties']


# ---------------------------------------------------------------------------
# Internal fan-out must register the per-model task under the chat id (Q2)
# ---------------------------------------------------------------------------

MAIN_PATH = Path(__file__).resolve().parents[1] / 'main.py'


def _internal_fanout_branch():
    """Return the `if is_internal:` fan-out block (the one containing `continue`)."""
    tree = ast.parse(MAIN_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'is_internal':
            if any(isinstance(stmt, ast.Continue) for stmt in node.body):
                return node
    return None


def test_internal_fanout_registers_task_under_chat_id():
    """Guards the Q2 fix: the internal branch must register its per-model task
    under the chat id with the same `task_id` the streaming handler keys the live
    response stream on. Otherwise `get_response_streams_by_chat_id` cannot find a
    running subagent child's (or the parent synthesis') in-progress output."""
    branch = _internal_fanout_branch()
    assert branch is not None, 'internal fan-out branch not found'

    create_calls = [
        call
        for call in ast.walk(branch)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'create_task'
    ]
    assert create_calls, 'internal branch does not register its task via create_task'

    keywords = {kw.arg for call in create_calls for kw in call.keywords}
    assert 'id' in keywords and 'task_id' in keywords


# ---------------------------------------------------------------------------
# Per-call tool_call id reaches the wrapped builtin callable (Q1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_id_extra_param_reaches_callable():
    """The per-call id threaded through middleware extra_params must bind onto the
    wrapped builtin callable (the mechanism `subagent` relies on to report its
    child back to the exact tool-call row)."""
    from open_webui.utils.tools import (
        get_async_tool_function_and_apply_extra_params,
        get_builtin_function_introspection,
        get_updated_tool_function,
    )

    seen: dict = {}

    async def tool(prompt: str, __tool_call_id__: str = None):
        seen['prompt'] = prompt
        seen['tool_call_id'] = __tool_call_id__
        return 'ok'

    wrapped = await get_async_tool_function_and_apply_extra_params(
        tool, {'__request__': None}, get_builtin_function_introspection(tool)
    )
    updated = await get_updated_tool_function(wrapped, {'__tool_call_id__': 'call-7'})
    await updated(prompt='hi')

    assert seen == {'prompt': 'hi', 'tool_call_id': 'call-7'}


