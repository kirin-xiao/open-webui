"""Focused unit tests for the budgeted context-compaction rewrite (Slice A).

``open_webui.utils.context_compaction`` imports ``models.chats`` /
``models.config`` (and therefore the DB layer), so it is not importable in the
shallow test tier. The pure helpers under test are AST-extracted from source and
``exec``'d with faithful stubs for the few globals they touch — the same
extraction pattern as ``test_cache_usage.py`` / ``test_frozen_top_content.py``.

The tests pin the invariants the middleware and system baseline depend on:

* the ceiling is the configured threshold unless model metadata lowers it;
* token estimation anchors on provider usage but counts CJK as ~1 token;
* the tail walk keeps a user-aligned window and always keeps the newest entry;
* tool outputs are truncated in compacted input and serialized tail;
* the checkpoint record round-trips as JSON and still reads legacy strings;
* the checkpoint renders as a user-role ``<conversation-checkpoint>`` message
  carrying ``<summary>`` only (the retained tail is not duplicated);
* middleware inserts that message and never injects ``[CONVERSATION SUMMARY]``
  into the system top, nor forces a baseline re-freeze.

Slice B additions:

* the summary template carries the required headings and rules;
* first-run vs incremental (update) prompt selection, plus legacy rewrite;
* at most one corrective retry before the degraded fallback.

Slice C1/C2 additions:

* the per-chat Redis lease (acquire/release/skip-on-contention) and its
  in-process fail-open fallback when Redis is unavailable;
* the ``chat.meta`` lifecycle ledger and idempotent stale ``running`` settlement;
* reactive overflow detection plus a single lower-budget recovery and one retry,
  with guards against recovery after a normal compaction or a second attempt.
"""

import ast
import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

COMPACTION_PY = Path(__file__).resolve().parents[1] / 'utils' / 'context_compaction.py'
MIDDLEWARE_PY = Path(__file__).resolve().parents[1] / 'utils' / 'middleware.py'
BASELINE_PY = Path(__file__).resolve().parents[1] / 'utils' / 'system_baseline.py'
MAIN_PY = Path(__file__).resolve().parents[1] / 'main.py'

CONSTS = (
    'SUMMARY_HEADINGS',
    'SUMMARY_SECTIONS',
    'SUMMARY_RULES',
    'SUMMARY_SHARED_INSTRUCTION',
    'DEFAULT_CONTEXT_COMPACTION_PROMPT',
    'LEGACY_SUMMARY_INSTRUCTION',
    'UPDATE_CONTEXT_COMPACTION_PROMPT',
    'CORRECTIVE_SUMMARY_PROMPT',
    'DEFAULT_SUMMARY_MAX_TOKENS',
    'CHECKPOINT_VERSION',
    'CHECKPOINT_STATUS_COMPLETED',
    'LEGACY_CHECKPOINT_VERSION',
    'DEFAULT_CONTEXT_COMPACTION_BUFFER',
    'DEFAULT_CONTEXT_COMPACTION_KEEP_TOKENS',
    'OUTPUT_TOKEN_MAX',
    'TOOL_OUTPUT_MAX_CHARS',
    'CHECKPOINT_TAG',
    'CONTEXT_COMPACTION_LOCK_KEY',
    'CONTEXT_COMPACTION_LOCK_TTL',
    'COMPACTION_IN_PROGRESS_DETAIL',
    'CONTEXT_COMPACTION_STATE_KEY',
    'STATUS_RUNNING',
    'STATUS_COMPLETED',
    'STATUS_FAILED',
    '_COMPACTION_LOCK_RELEASE_SCRIPT',
    'CONTEXT_OVERFLOW_MARKERS',
    'CONTEXT_OVERFLOW_ERROR_NAMES',
)

FUNCS = (
    '_is_cjk',
    '_estimate_tokens',
    'truncate_tool_output',
    '_output_parts_text',
    '_truncate_output_parts',
    '_truncate_tool_outputs_in_message',
    '_truncate_tool_outputs_in_messages',
    '_serialize_recent_message',
    '_serialize_recent_messages',
    '_serialize_assistant_message',
    '_serialize_output_item',
    '_find_tail_start',
    '_first_int',
    '_cache_token_count',
    '_usage_token_count',
    '_message_usage',
    '_estimate_messages_tokens',
    '_estimate_context_tokens',
    '_exceeds_token_threshold',
    '_parse_positive_int',
    '_as_bool',
    '_resolve_token_threshold',
    '_model_context_limit',
    '_resolve_prompt_ceiling',
    '_newest_is_checkpoint',
    '_strip_transient_checkpoints',
    '_parse_checkpoint',
    '_apply_latest_summary_checkpoint',
    '_serialize_checkpoint',
    '_build_checkpoint_record',
    'build_checkpoint_message',
    'insert_checkpoint_message',
    '_build_context_usage',
    'compact_messages_for_request',
    '_execute_compaction',
    '_mark_compaction_ran',
    'compact_chat_branch',
    '_emit_compaction_status',
    '_get_redis',
    'acquire_compaction_lock',
    'release_compaction_lock',
    '_record_compaction_state',
    'settle_stale_compaction',
    '_error_text',
    'is_context_overflow_error',
    'compact_for_overflow',
    'build_compaction_prompt',
    'has_summary_section',
    '_is_legacy_summary',
    '_summary_task_model_params',
    '_degraded_summary',
    '_generate_summary',
    '_generate_summary_with_retry',
)


async def _default_load_chat_meta(chat_id):
    return {}


async def _default_save_meta_key(chat_id, key, value):
    return True


def _default_get_message_list(messages_map, message_id=None):
    if isinstance(messages_map, dict):
        return list(messages_map.values())
    return list(messages_map or [])



class _FakeJSONCodec:
    @staticmethod
    def dumps(value, **kwargs):
        return json.dumps(value, **kwargs)

    @staticmethod
    def loads(value, **kwargs):
        return json.loads(value, **kwargs)


def _get_output_text(output):
    if not isinstance(output, list):
        return ''
    texts = []
    for item in output:
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        parts = item.get('content') or []
        if not isinstance(parts, list):
            continue
        text = ''.join(
            str(part.get('text')) for part in parts if isinstance(part, dict) and part.get('text') is not None
        )
        if text and not text.isspace():
            texts.append(text)
    return '\n'.join(texts)


def _get_content_from_message(message):
    content = message.get('content')
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text')
    elif content:
        return content
    return _get_output_text(message.get('output')) or (content if isinstance(content, str) else None)


def _load_ns(source=COMPACTION_PY):
    import logging
    import time
    from typing import Any

    ns = {
        'JSONCodec': _FakeJSONCodec,
        'get_content_from_message': _get_content_from_message,
        'time': time,
        'uuid': uuid,
        'Any': Any,
        'log': logging.getLogger('test.context_compaction'),
        'HTTPException': HTTPException,
        'REDIS_KEY_PREFIX': 'open-webui',
        'REDIS_TASK_TTL': 300,
        'load_chat_meta': _default_load_chat_meta,
        'save_meta_key': _default_save_meta_key,
        'get_message_list': _default_get_message_list,
        '_COMPACTION_LOCAL_LOCKS': set(),
    }
    tree = ast.parse(source.read_text())
    found = set()
    for node in tree.body:
        is_constant = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in CONSTS
        )
        if is_constant or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCS
        ):
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
            found.add(node.targets[0].id if is_constant else node.name)
    assert found == set(FUNCS) | set(CONSTS), f'missing: {(set(FUNCS) | set(CONSTS)) - found}'
    return ns


def _function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f'{name} not found')


def _call_lines(node, func_name):
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if (isinstance(func, ast.Name) and func.id == func_name) or (
                isinstance(func, ast.Attribute) and func.attr == func_name
            ):
                lines.append(child.lineno)
    return lines


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_cjk_characters_count_as_one_token():
    ns = _load_ns()
    assert ns['_estimate_tokens']('中文字符') == 4
    assert ns['_estimate_tokens']('abcd中文') == 3  # 2 CJK + 4//4
    assert ns['_estimate_tokens']('abcdefgh') == 2
    assert ns['_estimate_tokens']('') == 0
    assert ns['_estimate_tokens'](None) == 0


def test_usage_anchor_includes_cache_read_and_write():
    ns = _load_ns()
    assert (
        ns['_usage_token_count'](
            {'prompt_tokens': 1000, 'completion_tokens': 50, 'cache_read_input_tokens': 300, 'cache_write_tokens': 20}
        )
        == 1370
    )
    # Nested OpenAI-style cache counter.
    assert ns['_usage_token_count'](
        {'input_tokens': 100, 'output_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 40}}
    ) == 150


def test_context_estimate_anchors_on_newest_usage_and_adds_deltas():
    ns = _load_ns()
    messages = [
        {'role': 'user', 'content': 'x' * 400},
        {
            'role': 'assistant',
            'content': 'ok',
            'usage': {'prompt_tokens': 1000, 'completion_tokens': 50, 'cache_read_input_tokens': 300},
        },
        {'role': 'user', 'content': 'y' * 400},
    ]
    delta = ns['_estimate_messages_tokens'](messages[2:])
    assert ns['_estimate_context_tokens'](messages) == 1350 + delta
    assert ns['_exceeds_token_threshold'](messages, '', None, 1350) is True
    assert ns['_exceeds_token_threshold'](messages, '', None, 100000) is False


# ---------------------------------------------------------------------------
# Ceiling math
# ---------------------------------------------------------------------------


def test_ceiling_defaults_to_configured_threshold_without_model_metadata():
    ns = _load_ns()
    config = {'token_threshold': 80000, 'token_cap': 100000, 'buffer': 20000}
    assert ns['_model_context_limit']({'info': {'params': {}}}) is None
    assert ns['_resolve_prompt_ceiling'](config) == 80000
    # Per-request override is capped by the global cap.
    assert ns['_resolve_prompt_ceiling'](config, {'params': {'compact_token_threshold': 5000}}) == 5000
    assert ns['_resolve_prompt_ceiling'](config, {'params': {'compact_token_threshold': 200000}}) == 100000


def test_model_metadata_can_only_lower_the_ceiling():
    ns = _load_ns()
    config = {'token_threshold': 80000, 'token_cap': 200000, 'buffer': 20000}

    context_limited = {'info': {'meta': {'context_length': 100000, 'max_output_tokens': 40000}}}
    # context - max(min(output, 32000), buffer) = 100000 - 32000
    assert ns['_resolve_prompt_ceiling'](config, None, context_limited) == 68000

    input_limited = {'info': {'meta': {'max_input_tokens': 50000}}}
    assert ns['_resolve_prompt_ceiling'](config, None, input_limited) == 30000

    # Metadata that is higher than the configured threshold never raises it.
    generous = {'info': {'meta': {'context_length': 1000000}}}
    assert ns['_resolve_prompt_ceiling'](config, None, generous) == 80000


# ---------------------------------------------------------------------------
# Tail walk
# ---------------------------------------------------------------------------


def _long(role, content, **extra):
    return {'role': role, 'content': content, **extra}


def test_tail_walk_keeps_newest_entry_even_when_over_budget():
    ns = _load_ns()
    messages = [
        _long('user', 'u1' * 200),
        _long('assistant', 'a1' * 200),
        _long('user', 'u2' * 200),
        _long('assistant', 'a2' * 200),
    ]
    # Only the newest exchange fits: it is kept, the older prefix is summarized.
    assert ns['_find_tail_start'](messages, 1) == 2
    # A lone user prompt with a response has nothing older to summarize.
    assert ns['_find_tail_start']([_long('user', 'hi'), _long('assistant', 'yo')], 1) is None


def test_tail_walk_snaps_to_user_boundary_and_keeps_tool_pairs():
    ns = _load_ns()
    messages = [
        _long('user', 'u1'),
        _long('assistant', 'a1'),
        _long('user', 'u2'),
        {
            'role': 'assistant',
            'content': '',
            'output': [{'type': 'function_call', 'name': 'f', 'arguments': '{}'}],
        },
        {'role': 'tool', 'content': 'tool result', 'tool_call_id': 'c1'},
        _long('user', 'u3'),
        _long('assistant', 'a3'),
    ]
    for keep_tokens in (1, 200, 400, 2000):
        start = ns['_find_tail_start'](messages, keep_tokens)
        assert start is not None
        # The retained window never starts mid-turn.
        assert messages[start]['role'] == 'user'
        # The newest entry is always retained.
        assert start <= len(messages) - 1


def test_tail_walk_metrics_choose_the_requested_window():
    ns = _load_ns()
    # Each serialized message is ~102 tokens; 320 fits a3+u3+a2 but not u2, then
    # snaps back from a2 to the u2 turn boundary.
    messages = [
        _long('user', 'u1' * 200),
        _long('assistant', 'a1' * 200),
        _long('user', 'u2' * 200),
        _long('assistant', 'a2' * 200),
        _long('user', 'u3' * 200),
        _long('assistant', 'a3' * 200),
    ]
    assert ns['_find_tail_start'](messages, 320) == 2


def test_tail_walk_handles_degenerate_histories():
    ns = _load_ns()
    # Nothing to summarize: empty, all-system, or a single user turn.
    assert ns['_find_tail_start']([], 100) is None
    assert ns['_find_tail_start']([_long('system', 'sys')], 100) is None
    assert ns['_find_tail_start']([_long('user', 'only')], 100) is None
    # A newest-only user turn after older turns still snaps to its boundary.
    assert (
        ns['_find_tail_start']([_long('user', 'older'), _long('assistant', 'reply'), _long('user', 'newest')], 1)
        == 2
    )


def test_tail_walk_keeps_block_content_and_tool_only_turns():
    ns = _load_ns()
    messages = [
        _long('user', 'older'),
        _long('assistant', 'reply'),
        _long('user', [{'type': 'text', 'text': 'newest block content'}]),
        {
            'role': 'assistant',
            'content': '',
            'output': [{'type': 'function_call', 'name': 'f', 'arguments': '{}'}],
        },
    ]
    start = ns['_find_tail_start'](messages, 1)
    # Keep the newest user turn together with the tool call it produced.
    assert start == 2
    assert messages[start]['role'] == 'user'
    # A tool result survives serialization without a crash on list content.
    assert 'newest block content' in ns['_serialize_recent_messages'](messages[start:])


# ---------------------------------------------------------------------------
# Tool-output truncation
# ---------------------------------------------------------------------------


def test_truncate_tool_output_keeps_short_and_clips_long():
    ns = _load_ns()
    assert ns['truncate_tool_output']('short') == 'short'
    clipped = ns['truncate_tool_output']('a' * 2500)
    assert clipped.endswith('\n[truncated]')
    assert clipped[:2000] == 'a' * 2000


def test_tool_outputs_truncated_in_messages():
    ns = _load_ns()
    long_text = 'z' * 5000
    messages = [
        {'role': 'tool', 'content': long_text, 'tool_call_id': 'c1'},
        {
            'role': 'assistant',
            'content': '',
            'output': [
                {'type': 'function_call_output', 'call_id': 'c1', 'output': [{'type': 'input_text', 'text': long_text}]}
            ],
        },
    ]
    truncated = ns['_truncate_tool_outputs_in_messages'](messages)
    assert truncated[0]['content'].endswith('\n[truncated]')
    assert truncated[1]['output'][0]['output'][0]['text'].endswith('\n[truncated]')
    # The original messages are not mutated.
    assert messages[0]['content'] == long_text


# ---------------------------------------------------------------------------
# Structured checkpoint record / legacy shim
# ---------------------------------------------------------------------------


def test_structured_checkpoint_round_trips():
    ns = _load_ns()
    record = ns['_build_checkpoint_record'](
        summary='the summary',
        recent='[User]: recent',
        dropped_messages=[{'id': 'm1'}, {'id': 'm2'}],
        model='model-x',
        tokens=1234,
    )
    text = ns['_serialize_checkpoint'](record)
    assert isinstance(text, str)
    parsed = ns['_parse_checkpoint'](text)
    assert parsed['summary'] == 'the summary'
    assert parsed['recent'] == '[User]: recent'
    assert parsed['dropped_ids'] == ['m1', 'm2']
    assert parsed['model'] == 'model-x'
    assert parsed['tokens'] == 1234
    assert parsed['version'] == ns['CHECKPOINT_VERSION']
    assert parsed['status'] == 'completed'


def test_legacy_bare_string_is_accepted_as_summary():
    ns = _load_ns()
    parsed = ns['_parse_checkpoint']('a legacy summary')
    assert parsed['summary'] == 'a legacy summary'
    assert parsed['version'] == ns['LEGACY_CHECKPOINT_VERSION']
    assert parsed['legacy'] is True

    messages = [
        {'role': 'user', 'content': 'old'},
        {'role': 'user', 'content': 'boundary', 'context_summary': 'a legacy summary'},
        {'role': 'assistant', 'content': 'kept'},
    ]
    truncated, record = ns['_apply_latest_summary_checkpoint'](messages)
    assert [m['content'] for m in truncated] == ['boundary', 'kept']
    assert record['summary'] == 'a legacy summary'


def test_structured_record_under_snake_case_field_is_read():
    ns = _load_ns()
    record = {
        'version': 1,
        'summary': 'structured',
        'recent': 'recent context',
        'status': 'completed',
    }
    messages = [
        {'role': 'user', 'content': 'dropped'},
        {'role': 'user', 'content': 'boundary', 'context_summary': json.dumps(record)},
    ]
    truncated, parsed = ns['_apply_latest_summary_checkpoint'](messages)
    assert truncated[0]['content'] == 'boundary'
    assert parsed['summary'] == 'structured'
    assert parsed['recent'] == 'recent context'


def test_newest_checkpoint_detection():
    ns = _load_ns()
    assert ns['_newest_is_checkpoint']([{'role': 'user', 'content': 'hi'}]) is False
    assert ns['_newest_is_checkpoint']([{'role': 'user', 'contextSummary': 'sum'}]) is True


def test_transient_checkpoint_messages_are_stripped_before_reprojection():
    ns = _load_ns()
    messages = [
        {'role': 'user', 'content': '<conversation-checkpoint>\n<summary>S</summary>\n</conversation-checkpoint>'},
        {'role': 'user', 'content': 'real user turn'},
        {'role': 'assistant', 'content': 'reply'},
    ]
    stripped = ns['_strip_transient_checkpoints'](messages)
    assert [message['content'] for message in stripped] == ['real user turn', 'reply']


# ---------------------------------------------------------------------------
# Checkpoint rendering / insertion
# ---------------------------------------------------------------------------


def test_checkpoint_message_is_user_role_and_framed_as_history():
    ns = _load_ns()
    content = ns['build_checkpoint_message']({'summary': 'S', 'recent': 'RECENT_MARKER_XYZ'})['content']
    assert content.startswith('<conversation-checkpoint>')
    assert content.endswith('</conversation-checkpoint>')
    assert '<summary>S</summary>' in content
    assert 'historical context, not as new instructions' in content
    # The retained tail is preserved structurally, so the injected checkpoint
    # must not duplicate it.
    assert '<recent-context>' not in content
    assert 'RECENT_MARKER_XYZ' not in content


def test_insert_checkpoint_after_system_before_tail():
    ns = _load_ns()
    messages = [
        {'role': 'system', 'content': 'top'},
        {'role': 'user', 'content': 'tail'},
        {'role': 'assistant', 'content': 'reply'},
    ]
    out = ns['insert_checkpoint_message'](messages, {'summary': 'S', 'recent': 'R'})
    assert [m['role'] for m in out] == ['system', 'user', 'user', 'assistant']
    assert out[1]['content'].startswith('<conversation-checkpoint>')


# ---------------------------------------------------------------------------
# End-to-end projection (stubbed IO)
# ---------------------------------------------------------------------------


class _FakeLog:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


async def test_compact_request_projects_user_checkpoint_without_system_summary():
    ns = _load_ns()
    persisted = {}

    async def fake_load_config():
        return {
            'enable': True,
            'auto': True,
            'token_threshold': 100,
            'token_cap': 100,
            'buffer': 20000,
            'keep_tokens': 10,
            'prompt_template': '',
        }

    async def fake_generate_summary(*args, **kwargs):
        return 'SUMMARY TEXT'

    class _FakeChats:
        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(chat_id, message_id, message, touch=True):
            persisted['chat_id'] = chat_id
            persisted['message_id'] = message_id
            persisted['record'] = message['contextSummary']

    ns['_load_config'] = fake_load_config
    ns['_generate_summary'] = fake_generate_summary
    ns['Chats'] = _FakeChats
    ns['is_saved_chat_id'] = lambda chat_id: isinstance(chat_id, str) and chat_id.startswith('chat-')
    ns['log'] = _FakeLog()

    def _long(role, text, message_id):
        return {'role': role, 'content': text * 200, 'id': message_id}

    messages = [
        {'role': 'system', 'content': 'BASE TOP'},
        _long('user', 'u1 ', 'u1'),
        _long('assistant', 'a1 ', 'a1'),
        _long('user', 'u2 ', 'u2'),
        _long('assistant', 'a2 ', 'a2'),
        _long('user', 'u3 ', 'u3'),
        _long('assistant', 'a3 ', 'a3'),
    ]

    projected, record = await ns['compact_messages_for_request'](
        request=None,
        user=None,
        messages=messages,
        metadata={'chat_id': 'chat-1'},
        model_id='m',
        models={'m': {}},
        system_prompt='BASE TOP',
    )
    out = ns['insert_checkpoint_message'](projected, record)

    system_top = '\n'.join(m['content'] for m in out if m['role'] == 'system')
    assert '[CONVERSATION SUMMARY]' not in system_top
    checkpoints = [m for m in out if m['role'] == 'user' and m['content'].startswith('<conversation-checkpoint>')]
    assert len(checkpoints) == 1
    assert '<summary>SUMMARY TEXT</summary>' in checkpoints[0]['content']
    assert '<recent-context>' not in checkpoints[0]['content']
    # The dropped prefix is gone; the newest exchange survives.
    contents = [m.get('content') for m in out]
    assert messages[1]['content'] not in contents
    assert messages[5]['content'] in contents
    # Durable record written to the boundary (first retained) message as JSON.
    assert persisted['chat_id'] == 'chat-1'
    assert persisted['message_id'] == 'u3'
    assert ns['_parse_checkpoint'](persisted['record'])['summary'] == 'SUMMARY TEXT'


# ---------------------------------------------------------------------------
# Slice B — structured summary template / retry
# ---------------------------------------------------------------------------


def test_summary_template_has_required_headings_and_rules():
    ns = _load_ns()
    prompt = ns['build_compaction_prompt'](False)
    for heading in ns['SUMMARY_HEADINGS']:
        assert heading in prompt, f'missing {heading}'
    assert '### Completed' in prompt
    assert '### Active' in prompt
    assert '### Blocked' in prompt
    assert 'terse, single-line bullets' in prompt
    assert 'Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers' in prompt
    assert 'unanswered or require further action' in prompt
    assert 'uncommitted, committed, pushed,' in prompt
    assert 'under review, or merged' in prompt
    assert 'Do not mention the summary process' in prompt
    assert 'Do not continue the task or call tools' in prompt
    assert 'Return only the structured Markdown' in prompt
    assert 'at most 15' in prompt
    # Backward-compatible variable surface.
    assert '{{COMPACTED_MESSAGES}}' in prompt
    assert '{{RECENT_MESSAGES}}' in prompt


def test_first_run_versus_incremental_prompt_selection():
    ns = _load_ns()
    fresh = ns['build_compaction_prompt'](False)
    incremental = ns['build_compaction_prompt'](True)
    assert incremental != fresh
    assert 'Update and consolidate' in incremental
    assert 'Newer history always takes precedence' in incremental
    assert 'Reconcile Work State and Next Move' in incremental
    assert 'wrapper tags' in incremental
    assert '{{PREVIOUS_SUMMARY}}' in incremental
    for heading in ns['SUMMARY_HEADINGS']:
        assert heading in incremental, f'missing {heading}'
    # A custom admin template always wins, for either branch.
    assert ns['build_compaction_prompt'](False, False, 'CUSTOM') == 'CUSTOM'
    assert ns['build_compaction_prompt'](True, False, 'CUSTOM') == 'CUSTOM'
    # Whitespace-only overrides fall back to the built-in prompts.
    assert ns['build_compaction_prompt'](False, False, '   ') == fresh


def test_legacy_format_detection_triggers_rewrite():
    ns = _load_ns()
    assert ns['has_summary_section']('## Objective\n- do the thing') is True
    assert ns['has_summary_section']('free-form prose') is False
    assert ns['has_summary_section']('') is False

    assert ns['_is_legacy_summary']('free-form prose') is True
    assert ns['_is_legacy_summary']('## Work State\n### Active\n- x') is False
    assert ns['_is_legacy_summary'](None) is False
    assert ns['_is_legacy_summary']('   ') is False

    update = ns['build_compaction_prompt'](True, False)
    legacy = ns['build_compaction_prompt'](True, True)
    assert legacy != update
    assert 'earlier format' in legacy
    assert '{{PREVIOUS_SUMMARY}}' in legacy


async def test_corrective_retry_fires_once_then_falls_back():
    ns = _load_ns()
    calls = []

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return 'no headings here'

    ns['_request_summary_completion'] = fake_completion
    summary = await ns['_generate_summary_with_retry'](
        request=None,
        user=None,
        payload={'messages': [{'role': 'user', 'content': 'P'}]},
        previous_summary=None,
        compacted_messages=[{'role': 'user', 'content': 'dropped prefix'}],
    )

    # Exactly one corrective retry, no more.
    assert len(calls) == 2
    assert calls[1]['messages'][-1]['content'] == ns['CORRECTIVE_SUMMARY_PROMPT']
    # The original prompt is preserved ahead of the corrective follow-up.
    assert calls[1]['messages'][0]['content'] == 'P'
    # Second miss falls through to the degraded truncated-history dump.
    assert 'dropped prefix' in summary
    assert ns['has_summary_section'](summary) is False


async def test_corrective_retry_recovers_on_second_response():
    ns = _load_ns()
    calls = []

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return '## Objective\n- recovered' if len(calls) == 2 else 'nope'

    ns['_request_summary_completion'] = fake_completion
    summary = await ns['_generate_summary_with_retry'](None, None, {'messages': []}, None, [])
    assert summary == '## Objective\n- recovered'
    assert len(calls) == 2


async def test_valid_summary_skips_corrective_retry():
    ns = _load_ns()
    calls = []

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return '## Work State\n### Active\n- x'

    ns['_request_summary_completion'] = fake_completion
    summary = await ns['_generate_summary_with_retry'](None, None, {'messages': []}, None, [])
    assert summary == '## Work State\n### Active\n- x'
    assert len(calls) == 1


async def test_custom_template_accepts_headingless_summary():
    """An admin template owns its output shape; no built-in heading is required."""
    ns = _load_ns()
    calls = []

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return 'free-form custom summary'

    ns['_request_summary_completion'] = fake_completion
    summary = await ns['_generate_summary_with_retry'](
        None, None, {'messages': []}, None, [], require_headings=False
    )
    assert summary == 'free-form custom summary'
    assert len(calls) == 1


async def test_custom_template_empty_still_retries_once_then_falls_back():
    ns = _load_ns()
    calls = []

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return '   '

    ns['_request_summary_completion'] = fake_completion
    summary = await ns['_generate_summary_with_retry'](
        None, None, {'messages': []}, None, [], require_headings=False
    )
    assert len(calls) == 2
    assert ns['has_summary_section'](summary) is False


async def test_corrective_retry_empty_response_falls_back():
    ns = _load_ns()
    calls = []

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return ''

    ns['_request_summary_completion'] = fake_completion
    summary = await ns['_generate_summary_with_retry'](None, None, {'messages': []}, None, [])
    assert len(calls) == 2
    assert summary == ''


async def test_generate_summary_wires_custom_template_to_skip_heading_check():
    """End-to-end wiring: a custom template's heading-less output is not degraded."""
    ns = _load_ns()
    calls = []

    class _FakeConfig:
        @staticmethod
        async def get_many(*keys):
            return {'task.model.params': {}, 'chat.context_compaction.model': None}

    class _FakeRequest:
        pass

    async def fake_completion(request, payload, user):
        calls.append(payload)
        return 'CUSTOM SUMMARY without headings'

    ns['Config'] = _FakeConfig
    ns['_request_summary_completion'] = fake_completion
    # Pure passthroughs for the task-prompt helpers the extracted function calls.
    ns['replace_prompt_variable'] = lambda template, prompt: template
    ns['replace_messages_variable'] = lambda template, messages=None, variable_name='MESSAGES': (
        template.replace('{{' + variable_name + '}}', 'CONTENT')
    )
    ns['prompt_variables_template'] = lambda template, variables: template
    ns['get_last_user_message'] = lambda messages: 'last'
    ns['apply_params_to_form_data'] = lambda form_data, model, params=None: form_data

    async def fake_prompt_template(template, user=None):
        return template

    ns['prompt_template'] = fake_prompt_template

    summary = await ns['_generate_summary'](
        _FakeRequest(),
        None,
        'm',
        {'m': {'info': {'params': {}}}},
        [{'role': 'user', 'content': 'old'}],
        [{'role': 'assistant', 'content': 'new'}],
        None,
        'custom {{COMPACTED_MESSAGES}} {{RECENT_MESSAGES}}',
    )
    assert summary == 'CUSTOM SUMMARY without headings'
    assert len(calls) == 1


def test_summary_request_output_is_always_bounded():
    ns = _load_ns()
    default = ns['DEFAULT_SUMMARY_MAX_TOKENS']
    assert isinstance(default, int) and default > 0

    # An explicit task-model bound always wins.
    assert ns['_summary_task_model_params']({'max_tokens': 512}, {}) == {'max_tokens': 512}
    # Otherwise the model's own bound is used.
    assert ns['_summary_task_model_params'](
        {}, {'info': {'params': {'max_tokens': 2048}}}
    ) == {'max_tokens': 2048}
    # Otherwise a sane default is applied — never unset.
    assert ns['_summary_task_model_params']({}, {}) == {'max_tokens': default}
    assert ns['_summary_task_model_params'](None, {'info': {}})['max_tokens'] == default
    # Empty-string params are dropped, not forwarded.
    assert ns['_summary_task_model_params']({'temperature': ''}, {}) == {'max_tokens': default}


def test_degraded_summary_dumps_truncated_history():
    ns = _load_ns()
    summary = ns['_degraded_summary'](
        'previous summary',
        [
            {'role': 'user', 'content': 'q' * 1000},
            {'role': 'assistant', 'content': 'a' * 1000},
        ],
    )
    assert summary.startswith('previous summary')
    assert '- user: ' + 'q' * 500 in summary
    assert '- assistant: ' + 'a' * 500 in summary


# ---------------------------------------------------------------------------
# Middleware wiring — no system-top summary, no forced re-freeze
# ---------------------------------------------------------------------------


def test_middleware_inserts_user_checkpoint_without_system_summary():
    tree = ast.parse(MIDDLEWARE_PY.read_text())
    payload = _function_node(tree, 'process_chat_payload')

    assert _call_lines(payload, 'insert_checkpoint_message'), 'checkpoint must be inserted at request assembly'
    assert _call_lines(payload, 'compact_messages_for_request'), 'auto trigger must still run'

    # No executable string constant may carry the retired system-message marker.
    for node in ast.walk(payload):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert '[CONVERSATION SUMMARY]' not in node.value

    # The compaction path must not force a baseline re-freeze.
    for node in ast.walk(payload):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'apply_system_baseline':
            assert all(keyword.arg != 'force' for keyword in node.keywords)


def test_system_baseline_has_no_force_switch():
    tree = ast.parse(BASELINE_PY.read_text())
    fn = _function_node(tree, 'apply_system_baseline')
    assert 'force' not in [arg.arg for arg in fn.args.kwonlyargs]


def test_summary_request_strips_recursion_metadata():
    tree = ast.parse(COMPACTION_PY.read_text())
    fn = _function_node(tree, '_generate_summary')
    stripped = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.Tuple)):
            iterable = node.iter if isinstance(node, ast.For) else node
            elts = getattr(iterable, 'elts', None)
            if elts:
                for elt in elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        stripped.add(elt.value)
    assert {'chat_id', 'user_message_id'} <= stripped


def test_middleware_compaction_is_gated_on_owns_ledger():
    tree = ast.parse(MIDDLEWARE_PY.read_text())
    payload = _function_node(tree, 'process_chat_payload')
    assigned = False
    used = False
    for node in ast.walk(payload):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'owns_ledger' for target in node.targets
        ):
            assigned = True
        if isinstance(node, ast.Name) and node.id == 'owns_ledger' and isinstance(node.ctx, ast.Load):
            used = True
    assert assigned and used


# ---------------------------------------------------------------------------
# Slice C1 — per-chat lock
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeApp:
    def __init__(self, redis):
        self.state = _FakeState(redis=redis)


class _FakeRequest:
    def __init__(self, redis=None):
        self.app = _FakeApp(redis)
        self.state = _FakeState()


class _FakeRedis:
    """Minimal async Redis with ``SET NX EX`` and token-guarded ``EVAL`` release."""

    def __init__(self):
        self.values = {}
        self.set_calls = []

    async def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def eval(self, script, numkeys, key, token):
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0


class _BrokenRedis:
    async def set(self, *args, **kwargs):
        raise RuntimeError('redis unavailable')

    async def eval(self, *args, **kwargs):
        raise RuntimeError('redis unavailable')


async def test_compaction_lock_acquires_once_and_releases_by_token():
    ns = _load_ns()
    redis = _FakeRedis()
    request = _FakeRequest(redis)
    key = f'{ns["CONTEXT_COMPACTION_LOCK_KEY"]}:chat-1'

    token = await ns['acquire_compaction_lock'](request, 'chat-1')
    assert token
    assert redis.set_calls[0][2] is True  # NX
    assert redis.set_calls[0][3] == ns['CONTEXT_COMPACTION_LOCK_TTL']
    # A second caller cannot acquire while the lease is held.
    assert await ns['acquire_compaction_lock'](request, 'chat-1') is None

    await ns['release_compaction_lock'](request, 'chat-1', token)
    assert key not in redis.values
    assert await ns['acquire_compaction_lock'](request, 'chat-1')


async def test_compaction_lock_release_does_not_drop_another_holders_token():
    ns = _load_ns()
    redis = _FakeRedis()
    request = _FakeRequest(redis)
    key = f'{ns["CONTEXT_COMPACTION_LOCK_KEY"]}:chat-1'

    token = await ns['acquire_compaction_lock'](request, 'chat-1')
    # Simulate the lease expiring and another process acquiring it.
    redis.values[key] = 'other-holder'
    await ns['release_compaction_lock'](request, 'chat-1', token)
    assert redis.values[key] == 'other-holder'


async def test_compaction_lock_falls_back_in_process_without_redis():
    ns = _load_ns()
    request = _FakeRequest(None)
    assert ns['_get_redis'](request) is None

    token = await ns['acquire_compaction_lock'](request, 'chat-1')
    assert token
    assert await ns['acquire_compaction_lock'](request, 'chat-1') is None
    await ns['release_compaction_lock'](request, 'chat-1', token)
    assert await ns['acquire_compaction_lock'](request, 'chat-1')


async def test_compaction_lock_fails_open_when_redis_errors():
    ns = _load_ns()
    request = _FakeRequest(_BrokenRedis())

    token = await ns['acquire_compaction_lock'](request, 'chat-1')
    assert token, 'a Redis error must not block compaction'
    assert await ns['acquire_compaction_lock'](request, 'chat-1') is None
    # Release also tolerates the broken backend.
    await ns['release_compaction_lock'](request, 'chat-1', token)


async def test_compact_messages_skips_when_lock_is_held():
    ns = _load_ns()
    summary_calls = []

    async def fake_config():
        return {
            'enable': True,
            'auto': True,
            'token_threshold': 100,
            'token_cap': 100,
            'buffer': 20000,
            'keep_tokens': 10,
            'prompt_template': '',
        }

    async def fake_summary(*args, **kwargs):
        summary_calls.append(1)
        return 'SUMMARY'

    async def lock_held(request, chat_id):
        return None

    ns['_load_config'] = fake_config
    ns['_generate_summary'] = fake_summary
    ns['acquire_compaction_lock'] = lock_held

    def _long(role, text, message_id):
        return {'role': role, 'content': text * 200, 'id': message_id}

    messages = [
        {'role': 'system', 'content': 'TOP'},
        _long('user', 'u1 ', 'u1'),
        _long('assistant', 'a1 ', 'a1'),
        _long('user', 'u2 ', 'u2'),
        _long('assistant', 'a2 ', 'a2'),
        _long('user', 'u3 ', 'u3'),
        _long('assistant', 'a3 ', 'a3'),
    ]

    projected, record = await ns['compact_messages_for_request'](
        None, None, messages, {'chat_id': 'chat-1'}, 'm', {'m': {}}, 'TOP'
    )

    assert record is None
    assert summary_calls == []
    # Contention must skip compaction, not error: the full history is retained.
    contents = [message.get('content') for message in projected]
    assert messages[1]['content'] in contents
    assert messages[-1]['content'] in contents


class _ManualChat:
    id = 'chat-1'
    current_message_id = 'm2'
    chat = {'history': {'currentId': 'm2'}}


async def test_manual_compaction_returns_409_when_lock_is_held():
    ns = _load_ns()
    summary_calls = []

    async def fake_config():
        return {
            'enable': True,
            'auto': True,
            'token_threshold': 100,
            'token_cap': 100,
            'buffer': 20000,
            'keep_tokens': 10,
            'prompt_template': '',
        }

    class _Chats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            return {'m1': {}, 'm2': {}}

    async def lock_held(request, chat_id):
        return None

    async def fake_summary(*args, **kwargs):
        summary_calls.append(1)
        return 'SUMMARY'

    ns['_load_config'] = fake_config
    ns['Chats'] = _Chats
    ns['acquire_compaction_lock'] = lock_held
    ns['_generate_summary'] = fake_summary
    ns['get_message_list'] = lambda messages_map, current_id: [
        {'id': 'm1', 'role': 'user', 'content': 'old'},
        {'id': 'm2', 'role': 'assistant', 'content': 'new'},
    ]

    with pytest.raises(HTTPException) as excinfo:
        await ns['compact_chat_branch'](None, None, _ManualChat(), 'model', {})

    assert excinfo.value.status_code == 409
    assert 'Wait for the current response to finish' in excinfo.value.detail
    assert summary_calls == []


# ---------------------------------------------------------------------------
# Slice C1 — lifecycle ledger + stale settlement
# ---------------------------------------------------------------------------


async def _compaction_settings():
    return {
        'enable': True,
        'auto': True,
        'token_threshold': 100,
        'token_cap': 100,
        'buffer': 20000,
        'keep_tokens': 10,
        'prompt_template': '',
    }


def _long_history():
    def _long(role, text, message_id):
        return {'role': role, 'content': text * 200, 'id': message_id}

    return [
        {'role': 'system', 'content': 'TOP'},
        _long('user', 'u1 ', 'u1'),
        _long('assistant', 'a1 ', 'a1'),
        _long('user', 'u2 ', 'u2'),
        _long('assistant', 'a2 ', 'a2'),
        _long('user', 'u3 ', 'u3'),
        _long('assistant', 'a3 ', 'a3'),
    ]


async def test_compaction_lifecycle_records_running_then_completed():
    ns = _load_ns()
    writes = []

    class _Chats:
        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(chat_id, message_id, message, touch=True):
            pass

    async def fake_summary(*args, **kwargs):
        return 'SUMMARY'

    async def fake_save(chat_id, key, value):
        writes.append((chat_id, key, value))
        return True

    ns['_load_config'] = _compaction_settings
    ns['_generate_summary'] = fake_summary
    ns['Chats'] = _Chats
    ns['is_saved_chat_id'] = lambda chat_id: isinstance(chat_id, str) and chat_id.startswith('chat-')
    ns['save_meta_key'] = fake_save

    projected, record = await ns['compact_messages_for_request'](
        None, None, _long_history(), {'chat_id': 'chat-1'}, 'm', {'m': {}}, 'TOP'
    )

    assert record is not None
    statuses = [value['status'] for _, key, value in writes if key == ns['CONTEXT_COMPACTION_STATE_KEY']]
    assert statuses == [ns['STATUS_RUNNING'], ns['STATUS_COMPLETED']]


async def test_compaction_lifecycle_records_failed_and_reraises():
    ns = _load_ns()
    writes = []

    async def failing_summary(*args, **kwargs):
        raise RuntimeError('summarizer exploded')

    async def fake_save(chat_id, key, value):
        writes.append((key, value))
        return True

    ns['_load_config'] = _compaction_settings
    ns['_generate_summary'] = failing_summary
    ns['save_meta_key'] = fake_save

    with pytest.raises(RuntimeError):
        await ns['compact_messages_for_request'](
            None, None, _long_history(), {'chat_id': 'chat-1'}, 'm', {'m': {}}, 'TOP'
        )

    statuses = [value['status'] for key, value in writes if key == ns['CONTEXT_COMPACTION_STATE_KEY']]
    assert statuses == [ns['STATUS_RUNNING'], ns['STATUS_FAILED']]


async def test_settle_stale_compaction_is_idempotent():
    ns = _load_ns()
    key = ns['CONTEXT_COMPACTION_STATE_KEY']
    store = {key: {'status': ns['STATUS_RUNNING']}}
    writes = []

    async def fake_load(chat_id):
        return dict(store)

    async def fake_save(chat_id, save_key, value):
        store[save_key] = value
        writes.append((chat_id, save_key, value))
        return True

    ns['load_chat_meta'] = fake_load
    ns['save_meta_key'] = fake_save

    assert await ns['settle_stale_compaction']('chat-1') is True
    assert store[key]['status'] == ns['STATUS_FAILED']
    assert store[key]['settled'] is True
    # Idempotent: a settled (or absent) record is never rewritten.
    assert await ns['settle_stale_compaction']('chat-1') is False
    assert len(writes) == 1


async def test_settle_stale_compaction_ignores_completed():
    ns = _load_ns()
    key = ns['CONTEXT_COMPACTION_STATE_KEY']
    writes = []

    async def fake_load(chat_id):
        return {key: {'status': ns['STATUS_COMPLETED']}}

    async def fake_save(chat_id, save_key, value):
        writes.append(value)
        return True

    ns['load_chat_meta'] = fake_load
    ns['save_meta_key'] = fake_save

    assert await ns['settle_stale_compaction']('chat-1') is False
    assert writes == []


async def test_settle_stale_compaction_does_not_settle_a_live_run():
    """A concurrent attempt must not fail a sibling still inside its lease.

    ``_execute_compaction`` settles before acquiring the lock, so a fresh
    ``running`` record written by the current holder must survive.
    """
    ns = _load_ns()
    key = ns['CONTEXT_COMPACTION_STATE_KEY']
    store = {key: {'status': ns['STATUS_RUNNING'], 'updated_at': int(time.time())}}
    writes = []

    async def fake_load(chat_id):
        return dict(store)

    async def fake_save(chat_id, save_key, value):
        store[save_key] = value
        writes.append(value)
        return True

    ns['load_chat_meta'] = fake_load
    ns['save_meta_key'] = fake_save

    assert await ns['settle_stale_compaction']('chat-1') is False
    assert store[key]['status'] == ns['STATUS_RUNNING']
    assert writes == []


async def test_settle_stale_compaction_settles_a_record_past_the_lease():
    ns = _load_ns()
    key = ns['CONTEXT_COMPACTION_STATE_KEY']
    store = {
        key: {
            'status': ns['STATUS_RUNNING'],
            'updated_at': int(time.time()) - ns['CONTEXT_COMPACTION_LOCK_TTL'] - 10,
        }
    }
    writes = []

    async def fake_load(chat_id):
        return dict(store)

    async def fake_save(chat_id, save_key, value):
        store[save_key] = value
        writes.append(value)
        return True

    ns['load_chat_meta'] = fake_load
    ns['save_meta_key'] = fake_save

    assert await ns['settle_stale_compaction']('chat-1') is True
    assert store[key]['status'] == ns['STATUS_FAILED']
    assert len(writes) == 1


# ---------------------------------------------------------------------------
# Slice C2 — overflow detection + one-shot recovery
# ---------------------------------------------------------------------------


def test_context_overflow_detection_matches_provider_failures():
    ns = _load_ns()
    is_overflow = ns['is_context_overflow_error']

    assert is_overflow("This model's maximum context length is 8192 tokens")
    assert is_overflow('context_length_exceeded')
    assert is_overflow('prompt is too long: 9000 tokens > 8000 maximum')
    assert is_overflow('Please reduce the length of the messages')
    assert is_overflow('Input token count exceeds the maximum number of tokens allowed')
    assert is_overflow({'error': {'message': 'the input length exceeds the context length'}})

    # Unrelated provider failures are not treated as overflow.
    assert not is_overflow('invalid api key')
    assert not is_overflow('rate limit exceeded, retry later')
    assert not is_overflow({'error': {'message': 'model not found'}})

    class ContextOverflowError(Exception):
        pass

    assert is_overflow(ContextOverflowError('typed sdk error'))
    assert not is_overflow(ValueError('typed sdk error'))

    overflow_response = JSONResponse(status_code=400, content={'error': {'message': 'maximum context length exceeded'}})
    assert is_overflow(overflow_response)
    other_response = JSONResponse(status_code=401, content={'error': {'message': 'unauthorized'}})
    assert not is_overflow(other_response)


MIDDLEWARE_RECOVERY_FUNCS = ('is_error_response', 'recover_from_context_overflow', 'run_with_overflow_recovery')


def _load_middleware_recovery(extra=None):
    import logging

    async def default_compact_for_overflow(request, user, messages, metadata, model_id, models, system_prompt):
        return messages, {'summary': 'RECOVERED'}

    ns = {
        'JSONResponse': JSONResponse,
        'log': logging.getLogger('test.middleware.recovery'),
        'is_saved_chat_id': lambda chat_id: isinstance(chat_id, str) and chat_id.startswith('chat-'),
        'get_system_message': lambda messages: None,
        'get_content_from_message': _get_content_from_message,
        'insert_checkpoint_message': lambda messages, record: messages,
        'process_messages_with_output': lambda messages, reasoning_format=None: messages,
        'sanitize_tool_pairs': lambda messages: messages,
        'get_reasoning_format': lambda model: None,
        'compaction_models': lambda request: {},
        'is_context_overflow_error': lambda error: True,
        'compact_for_overflow': default_compact_for_overflow,
    }
    if extra:
        ns.update(extra)

    tree = ast.parse(MIDDLEWARE_PY.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in MIDDLEWARE_RECOVERY_FUNCS:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(MIDDLEWARE_PY), 'exec'), ns)
            found.add(node.name)
    assert found == set(MIDDLEWARE_RECOVERY_FUNCS), f'missing: {set(MIDDLEWARE_RECOVERY_FUNCS) - found}'
    return ns


async def test_overflow_recovery_retries_exactly_once():
    ns = _load_middleware_recovery()
    attempts = []
    recovery_calls = []

    async def completion():
        attempts.append(1)
        if len(attempts) == 1:
            return JSONResponse(status_code=400, content={'error': {'message': 'maximum context length exceeded'}})
        return {'choices': [{'message': {'content': 'ok'}}]}

    async def fake_recover(request, form_data, user, model, metadata, error):
        recovery_calls.append(error)
        return True

    ns['recover_from_context_overflow'] = fake_recover
    form_data = {'messages': [], 'model': 'm'}
    response = await ns['run_with_overflow_recovery'](_FakeRequest(), form_data, None, {}, {}, completion)

    assert response == {'choices': [{'message': {'content': 'ok'}}]}
    assert len(attempts) == 2
    assert len(recovery_calls) == 1


async def test_overflow_recovery_does_not_retry_when_guard_declines():
    ns = _load_middleware_recovery()
    attempts = []

    async def completion():
        attempts.append(1)
        return JSONResponse(status_code=400, content={'error': {'message': 'maximum context length exceeded'}})

    async def fake_recover(*args):
        return False

    ns['recover_from_context_overflow'] = fake_recover
    response = await ns['run_with_overflow_recovery'](_FakeRequest(), {'messages': []}, None, {}, {}, completion)

    assert len(attempts) == 1
    assert ns['is_error_response'](response)


async def test_overflow_recovery_reraises_original_when_not_recoverable():
    ns = _load_middleware_recovery()

    async def completion():
        raise ValueError('maximum context length exceeded')

    async def fake_recover(*args):
        return False

    ns['recover_from_context_overflow'] = fake_recover
    with pytest.raises(ValueError):
        await ns['run_with_overflow_recovery'](_FakeRequest(), {'messages': []}, None, {}, {}, completion)


async def test_overflow_recovery_leaves_successful_response_alone():
    ns = _load_middleware_recovery()
    attempts = []
    recovery_calls = []

    async def completion():
        attempts.append(1)
        return {'ok': True}

    async def fake_recover(*args):
        recovery_calls.append(1)
        return True

    ns['recover_from_context_overflow'] = fake_recover
    response = await ns['run_with_overflow_recovery'](_FakeRequest(), {'messages': []}, None, {}, {}, completion)

    assert response == {'ok': True}
    assert len(attempts) == 1
    assert recovery_calls == []


async def test_recovery_skipped_after_a_compaction_ran_this_step():
    ns = _load_middleware_recovery()
    compact_calls = []

    async def fake_compact(request, user, messages, metadata, model_id, models, system_prompt):
        compact_calls.append(1)
        return messages, {'summary': 's'}

    ns['compact_for_overflow'] = fake_compact
    request = _FakeRequest()
    request.state.context_compaction_ran = True

    result = await ns['recover_from_context_overflow'](
        request,
        {'messages': [], 'model': 'm'},
        None,
        {},
        {'chat_id': 'chat-1', 'user_message_id': 'u1'},
        'overflow',
    )

    assert result is False
    assert compact_calls == []


async def test_recovery_runs_at_most_once_per_step():
    ns = _load_middleware_recovery()
    compact_calls = []

    async def fake_compact(request, user, messages, metadata, model_id, models, system_prompt):
        compact_calls.append(1)
        return messages, {'summary': 's'}

    ns['compact_for_overflow'] = fake_compact
    request = _FakeRequest()
    form_data = {'messages': [{'role': 'user', 'content': 'a'}], 'model': 'm'}
    metadata = {'chat_id': 'chat-1', 'user_message_id': 'u1'}

    assert await ns['recover_from_context_overflow'](request, form_data, None, {}, metadata, 'overflow') is True
    assert len(compact_calls) == 1
    assert request.state.context_overflow_recovery_attempted is True

    # A second overflow in the same step must not compact again.
    assert await ns['recover_from_context_overflow'](request, form_data, None, {}, metadata, 'overflow') is False
    assert len(compact_calls) == 1


async def test_recovery_prefers_the_unprojected_source_messages():
    ns = _load_middleware_recovery()
    seen = []

    async def fake_compact(request, user, messages, metadata, model_id, models, system_prompt):
        seen.append(messages)
        return messages, {'summary': 's'}

    ns['compact_for_overflow'] = fake_compact
    request = _FakeRequest()
    source = [{'id': 'u1', 'role': 'user', 'content': 'full history'}]
    request.state.context_compaction_source_messages = source
    form_data = {'messages': [{'role': 'user', 'content': 'projected'}], 'model': 'm'}
    metadata = {'chat_id': 'chat-1', 'user_message_id': 'u2'}

    assert await ns['recover_from_context_overflow'](request, form_data, None, {}, metadata, 'overflow') is True
    assert seen == [source]


async def test_recovery_ignores_non_overflow_errors():
    ns = _load_middleware_recovery({'is_context_overflow_error': lambda error: False})
    request = _FakeRequest()

    result = await ns['recover_from_context_overflow'](
        request,
        {'messages': [], 'model': 'm'},
        None,
        {},
        {'chat_id': 'chat-1', 'user_message_id': 'u1'},
        'unauthorized',
    )

    assert result is False
    assert getattr(request.state, 'context_overflow_recovery_attempted', False) is False


async def test_recovery_returns_false_when_nothing_to_compact():
    ns = _load_middleware_recovery()

    async def fake_compact(request, user, messages, metadata, model_id, models, system_prompt):
        return messages, None

    ns['compact_for_overflow'] = fake_compact
    request = _FakeRequest()

    result = await ns['recover_from_context_overflow'](
        request,
        {'messages': [], 'model': 'm'},
        None,
        {},
        {'chat_id': 'chat-1', 'user_message_id': 'u1'},
        'overflow',
    )

    assert result is False


async def test_recovery_requires_a_saved_chat_with_a_user_message():
    ns = _load_middleware_recovery()
    compact_calls = []

    async def fake_compact(*args, **kwargs):
        compact_calls.append(1)
        return [], {'summary': 's'}

    ns['compact_for_overflow'] = fake_compact
    request = _FakeRequest()

    # Temporary chat (no persistence surface for a checkpoint).
    assert (
        await ns['recover_from_context_overflow'](
            request, {'messages': []}, None, {}, {'chat_id': 'temp-1', 'user_message_id': 'u1'}, 'overflow'
        )
        is False
    )
    # Saved chat but no anchor to attach the checkpoint to.
    assert (
        await ns['recover_from_context_overflow'](
            request, {'messages': []}, None, {}, {'chat_id': 'chat-1'}, 'overflow'
        )
        is False
    )
    assert compact_calls == []


async def test_recovery_only_primary_compare_branch_owns_the_checkpoint():
    ns = _load_middleware_recovery()
    compact_calls = []

    async def fake_compact(*args, **kwargs):
        compact_calls.append(1)
        return args[2], {'summary': 's'}

    ns['compact_for_overflow'] = fake_compact
    form_data = {'messages': [{'role': 'user', 'content': 'a'}], 'model': 'm'}

    # A non-primary compare branch must leave the shared checkpoint to the primary.
    request = _FakeRequest()
    metadata = {
        'chat_id': 'chat-1',
        'user_message_id': 'u1',
        'compare_mode': True,
        'is_primary_branch': False,
    }
    assert await ns['recover_from_context_overflow'](request, form_data, None, {}, metadata, 'overflow') is False
    assert compact_calls == []
    assert getattr(request.state, 'context_overflow_recovery_attempted', False) is False

    # The primary branch may recover.
    request = _FakeRequest()
    metadata = {
        'chat_id': 'chat-1',
        'user_message_id': 'u1',
        'compare_mode': True,
        'is_primary_branch': True,
    }
    assert await ns['recover_from_context_overflow'](request, form_data, None, {}, metadata, 'overflow') is True
    assert len(compact_calls) == 1


# ---------------------------------------------------------------------------
# Slice C1 — wiring invariants
# ---------------------------------------------------------------------------


def test_execute_compaction_locks_before_summary_and_releases_after():
    tree = ast.parse(COMPACTION_PY.read_text())
    fn = _function_node(tree, '_execute_compaction')

    settle_lines = _call_lines(fn, 'settle_stale_compaction')
    acquire_lines = _call_lines(fn, 'acquire_compaction_lock')
    summary_lines = _call_lines(fn, '_generate_summary')
    release_lines = _call_lines(fn, 'release_compaction_lock')

    assert settle_lines and acquire_lines and summary_lines and release_lines
    assert min(settle_lines) < min(acquire_lines)
    assert min(acquire_lines) < min(summary_lines)
    assert max(release_lines) > min(summary_lines)


def test_process_chat_wires_one_shot_overflow_recovery():
    tree = ast.parse(MAIN_PY.read_text())
    fn = _function_node(tree, 'process_chat')
    assert _call_lines(fn, 'run_with_overflow_recovery'), 'provider completion must be wrapped for overflow recovery'
