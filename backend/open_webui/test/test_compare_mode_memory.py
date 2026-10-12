"""Tests for compare-mode memory writer gating (#30238).

Compare mode fans out one concurrent task per model. The memory store is an
unversioned read-modify-write, so several branches writing at once duplicate
rows and clobber replaces. These tests pin the chosen mitigation: keep the
mutating memory tools advertised (so the tool schema / system prompt is
unchanged) but make their callables return a fixed error, and run the
background memory review on the primary branch only.
"""

import asyncio
import copy
import json

import pytest
from open_webui.utils.misc import resolve_branch_flags
from open_webui.utils.subagents import (
    COMPARE_MODE_MEMORY_WRITE_DISABLED_MESSAGE,
    MUTATING_MEMORY_TOOLS,
)
from open_webui.utils.task import tools_function_calling_generation_template
from open_webui.utils.tools import disable_mutating_memory_tools

MEMORY_TOOL_NAMES = sorted(MUTATING_MEMORY_TOOLS)


async def _successful_tool(*args, **kwargs) -> str:
    return json.dumps({'status': 'success'})


def _build_tools() -> dict[str, dict]:
    """A representative tools dict: all memory writers plus an unrelated tool."""
    tools = {
        name: {
            'tool_id': f'builtin:{name}',
            'spec': {
                'name': name,
                'description': f'{name} does things',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'content': {'type': 'string'},
                        'memory_id': {'type': 'string'},
                    },
                    'required': ['content'],
                },
            },
            'callable': _successful_tool,
            'type': 'builtin',
        }
        for name in MEMORY_TOOL_NAMES
    }
    tools['search_web'] = {
        'tool_id': 'builtin:search_web',
        'spec': {'name': 'search_web', 'parameters': {'type': 'object', 'properties': {}}},
        'callable': _successful_tool,
        'type': 'builtin',
    }
    return tools


def _tool_specs(tools: dict[str, dict]) -> str:
    return json.dumps([tool['spec'] for tool in tools.values()], sort_keys=True)


def _system_prompt(tools: dict[str, dict]) -> str:
    # The legacy function-calling path renders the tool specs into a system
    # message; identical specs => identical prompt bytes.
    return tools_function_calling_generation_template('Available Tools:\n{{TOOLS}}', _tool_specs(tools))


def test_compare_mode_disables_only_mutating_memory_callables():
    tools = _build_tools()
    original_specs = copy.deepcopy({name: tool['spec'] for name, tool in tools.items()})

    disable_mutating_memory_tools(tools)

    for name in MEMORY_TOOL_NAMES:
        assert tools[name]['callable'] is not _successful_tool
        assert tools[name]['spec'] == original_specs[name]
        assert tools[name]['tool_id'] == f'builtin:{name}'

    # Non-memory tools are untouched.
    assert tools['search_web']['callable'] is _successful_tool


@pytest.mark.asyncio
async def test_disable_leaves_non_builtin_tool_of_same_name_alone():
    """A user-defined DB tool named like a memory tool must not be disabled."""
    user_tool = {
        'tool_id': 'user-tool-1',
        'spec': {'name': 'add_memory', 'parameters': {'type': 'object', 'properties': {}}},
        'callable': _successful_tool,
        'type': 'external',
    }
    mixed = {**_build_tools(), 'add_memory': user_tool}

    disable_mutating_memory_tools(mixed)

    # The external tool keeps its real callable...
    assert mixed['add_memory']['callable'] is _successful_tool
    # ...while the other builtin writers are still disabled.
    for name in MEMORY_TOOL_NAMES:
        if name == 'add_memory':
            continue
        payload = json.loads(await mixed[name]['callable']())
        assert payload == {'error': COMPARE_MODE_MEMORY_WRITE_DISABLED_MESSAGE}


@pytest.mark.asyncio
async def test_compare_mode_memory_calls_return_helpful_error():
    tools = _build_tools()
    disable_mutating_memory_tools(tools)

    for name in MEMORY_TOOL_NAMES:
        payload = json.loads(await tools[name]['callable'](content='anything'))
        assert payload == {'error': COMPARE_MODE_MEMORY_WRITE_DISABLED_MESSAGE}


class _FakeRequest:
    state = type('State', (), {'internal': False})()


@pytest.mark.asyncio
async def test_get_builtin_tools_disables_writers_when_compare_mode(monkeypatch):
    """End-to-end wiring: compare_mode metadata reaches get_builtin_tools and
    the mutating memory callables are replaced there."""
    from open_webui.utils import tools as tools_module

    class _Config:
        @staticmethod
        async def get_many(*keys):
            # Memory tools are gated on `memories.enable`; without it the
            # builtin list never contains the writers this test exercises.
            return {'memories.enable': True}

        @staticmethod
        async def get(key, default=None):
            return default

    async def _has_permission(*args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(tools_module, 'Config', _Config)
    monkeypatch.setattr(tools_module, 'has_permission', _has_permission)

    model = {'id': 'm', 'info': {'meta': {'capabilities': {'memory': True}, 'builtinTools': {}}}}
    extra_params = {
        '__user__': {'id': 'u', 'role': 'admin'},
        '__metadata__': {'compare_mode': True, 'chat_id': 'chat-1', 'files': []},
    }

    builtins = await tools_module.get_builtin_tools(_FakeRequest(), extra_params, {'memory': True}, model)

    compare_callables = {}
    for name in MEMORY_TOOL_NAMES:
        assert name in builtins, f'{name} should stay advertised in compare mode'
        payload = json.loads(await builtins[name]['callable'](content='x'))
        assert payload == {'error': COMPARE_MODE_MEMORY_WRITE_DISABLED_MESSAGE}
        compare_callables[name] = builtins[name]['callable']

    # Same call without compare_mode keeps the real writer (a distinct callable).
    extra_params['__metadata__'] = {'chat_id': 'chat-1', 'files': []}
    single = await tools_module.get_builtin_tools(_FakeRequest(), extra_params, {'memory': True}, model)
    for name in MEMORY_TOOL_NAMES:
        assert name in single
        assert single[name]['callable'] is not compare_callables[name]


def test_system_prompt_unchanged_when_switched_to_compare_mode():
    """Switching to compare mode must not rewrite the tool system prompt.

    Both the legacy prompt-rendered specs and the native tool schema are
    derived solely from `spec`, so keeping specs intact keeps the prefix
    unchanged and provider prompt caching stable.
    """
    single = _build_tools()
    compare = _build_tools()
    disable_mutating_memory_tools(compare)

    assert _tool_specs(single) == _tool_specs(compare)
    assert _system_prompt(single) == _system_prompt(compare)


class _DummyTask:
    """Minimal stand-in for an asyncio.Task in review-scheduling tests."""

    def add_done_callback(self, callback):
        pass


class TestResolveBranchFlags:
    """`main.py` derives compare/primary flags through this helper."""

    def test_single_entry_is_not_compare(self):
        is_compare, primary = resolve_branch_flags([{'model_id': 'a', 'message_id': 'm1'}], {})
        assert is_compare is False
        assert primary == 'm1'

    def test_multi_entry_is_compare(self):
        is_compare, primary = resolve_branch_flags(
            [{'model_id': 'a', 'message_id': 'm1'}, {'model_id': 'b', 'message_id': 'm2'}], {}
        )
        assert is_compare is True
        assert primary == 'm1'

    def test_primary_is_first_entry_with_a_message_id(self):
        is_compare, primary = resolve_branch_flags(
            [
                {'model_id': 'a', 'message_id': None},
                {'model_id': 'b', 'message_id': 'm2'},
                {'model_id': 'c', 'message_id': 'm3'},
            ],
            {},
        )
        assert is_compare is True
        assert primary == 'm2'

    def test_resume_single_entry_stays_compare_via_metadata(self):
        is_compare, primary = resolve_branch_flags(
            [{'model_id': 'a', 'message_id': 'm1'}],
            {'compare_mode': True, 'is_primary_branch': False},
        )
        assert is_compare is True
        assert primary == 'm1'

    def test_no_entries(self):
        is_compare, primary = resolve_branch_flags([], {})
        assert is_compare is False
        assert primary is None


@pytest.mark.asyncio
async def test_pause_persists_compare_context(monkeypatch):
    """The approval pause must persist the compare flags for the resume."""
    from open_webui.utils import middleware

    captured = {}

    async def _upsert(chat_id, message_id, data, touch=False):
        captured.update(data)

    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', _upsert)

    output = [{'type': 'function_call', 'id': 'fc1', 'call_id': 'fc1', 'name': 'add_memory', 'status': 'in_progress'}]
    await middleware.pause_for_tool_approval(
        'chat-1',
        'msg-1',
        output,
        {},
        {
            'compare_mode': True,
            'is_primary_branch': False,
            'session_id': 's',
            'params': {},
        },
    )

    assert captured['meta']['compare_mode'] is True
    assert captured['meta']['is_primary_branch'] is False


@pytest.mark.asyncio
async def test_resume_payload_restores_compare_context(monkeypatch):
    """`build_tool_approval_resume_payload` must return the persisted flags."""
    from open_webui.utils import tool_approval

    class _Chat:
        chat = {'models': ['m1'], 'params': {}}
        variables = {}

    assistant = {
        'parentId': 'user-1',
        'model': 'm1',
        'meta': {'compare_mode': True, 'is_primary_branch': False, 'params': {}},
    }

    async def _get_chat(chat_id, db=None):
        return _Chat()

    async def _get_message(chat_id, message_id):
        return assistant if message_id == 'msg-1' else {'role': 'user'}

    monkeypatch.setattr(tool_approval.Chats, 'get_chat_by_id', _get_chat)
    monkeypatch.setattr(tool_approval.Chats, 'get_message_by_id_and_message_id', _get_message)

    payload = await tool_approval.build_tool_approval_resume_payload('chat-1', 'msg-1')

    assert payload['compare_mode'] is True
    assert payload['is_primary_branch'] is False


@pytest.fixture
def review_harness(monkeypatch):
    """Capture whether `review_memory_after_turn` schedules the review task."""
    from open_webui.utils import memory

    scheduled: list[str] = []

    class _Config:
        @staticmethod
        async def get_many(*keys):
            return {
                # The review is gated on both the global memory switch and the
                # background-review switch.
                'memories.enable': True,
                'memories.background_review.enable': True,
                'memories.review_interval_turns': 1,
            }

    def _fake_create_task(coro):
        coro.close()  # avoid "coroutine was never awaited"
        scheduled.append('scheduled')
        return _DummyTask()

    monkeypatch.setattr(memory, 'Config', _Config)
    monkeypatch.setattr(memory.asyncio, 'create_task', _fake_create_task)
    return memory, scheduled


class _AdminUser:
    """Admin stand-in: skips the permission re-check in the review gate."""

    id = 'u'
    role = 'admin'


async def _run_review(memory, metadata):
    await memory.review_memory_after_turn(
        request=None,
        user=_AdminUser(),
        model={'id': 'test-model'},
        metadata=metadata,
        form_data={},
        assistant_message={'content': 'Something worth remembering.'},
        messages=[{'role': 'user', 'content': 'Remember this.'}],
    )


@pytest.mark.asyncio
async def test_review_skipped_on_secondary_compare_branch(review_harness):
    memory, scheduled = review_harness
    await _run_review(
        memory,
        {'features': {'memory': True}, 'compare_mode': True, 'is_primary_branch': False},
    )
    assert scheduled == []


@pytest.mark.asyncio
async def test_review_runs_on_primary_compare_branch(review_harness):
    memory, scheduled = review_harness
    await _run_review(
        memory,
        {'features': {'memory': True}, 'compare_mode': True, 'is_primary_branch': True},
    )
    assert scheduled == ['scheduled']


@pytest.mark.asyncio
async def test_review_runs_for_non_compare_requests(review_harness):
    memory, scheduled = review_harness
    await _run_review(memory, {'features': {'memory': True}})
    assert scheduled == ['scheduled']
