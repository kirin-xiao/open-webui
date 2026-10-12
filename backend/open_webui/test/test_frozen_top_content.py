"""Focused prompt-order tests for the frozen-top content layer (P0-1/P0-2).

``utils/system_baseline.py`` is light (it imports ``misc`` transitively, which is
aiohttp/mimeparse-heavy on some machines), so the pure helpers and the two new
prompt constants are AST-extracted from source and ``exec``'d with a faithful
``add_or_update_system_message`` stub — the same extraction pattern as
``test_cache_usage.py``. This keeps the tests runnable without the full app and
pins the ordering invariants that the middleware relies on:

* identity leads the frozen top and precedes the memory preamble / ``<memory_context>``;
* the memory preamble is absent entirely when memory is off (conditional assembly);
* the date line renders in the user's timezone and is appended after the frozen bytes.
"""

import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASELINE_PY = Path(__file__).resolve().parents[1] / 'utils' / 'system_baseline.py'
FUNCS = (
    'assemble_frozen_top_content',
    '_resolve_timezone',
    'render_current_date_line',
    'append_current_date_line',
)
CONSTS = ('SYSTEM_IDENTITY_PROMPT', 'MEMORY_CONTEXT_PREAMBLE')

# Mirrors ``open_webui.utils.misc.add_or_update_system_message`` exactly.
def _add_or_update_system_message(content, messages, append=False):
    if messages and messages[0].get('role') == 'system':
        if append:
            messages[0]['content'] = f'{messages[0]["content"]}\n{content}'
        else:
            messages[0]['content'] = f'{content}\n{messages[0]["content"]}'
    else:
        messages.insert(0, {'role': 'system', 'content': content})
    return messages


class _FakeLog:
    def debug(self, *args, **kwargs):
        pass


def _load_helpers():
    ns = {
        'datetime': datetime,
        'ZoneInfo': ZoneInfo,
        'add_or_update_system_message': _add_or_update_system_message,
        'log': _FakeLog(),
    }
    tree = ast.parse(BASELINE_PY.read_text())
    found: set[str] = set()
    for node in tree.body:
        is_constant = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in CONSTS
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in FUNCS):
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(BASELINE_PY), 'exec'), ns)
            found.add(node.targets[0].id if is_constant else node.name)
    assert found == set(FUNCS) | set(CONSTS), f'missing: {(set(FUNCS) | set(CONSTS)) - found}'
    return ns


# ---------------------------------------------------------------------------
# P0-1 — ordering and conditional assembly
# ---------------------------------------------------------------------------


def test_identity_precedes_memory_context_in_frozen_top():
    ns = _load_helpers()
    messages = [{'role': 'system', 'content': 'You are helpful.'}, {'role': 'user', 'content': 'hi'}]
    messages, signature = ns['assemble_frozen_top_content'](messages, memory_enabled=True)

    # Other injectors append after assembly: the memory base (memory.py) and the
    # Pyodide docs (config.py), both with append=True.
    messages = _add_or_update_system_message('<memory_context>\n- likes tea\n</memory_context>', messages, append=True)

    content = messages[0]['content']
    identity = ns['SYSTEM_IDENTITY_PROMPT']
    preamble = ns['MEMORY_CONTEXT_PREAMBLE']
    assert content.startswith(identity)  # identity leads, even before the user's own prompt
    assert content.index(identity) < content.index('You are helpful.')
    assert content.index(identity) < content.index(preamble) < content.index('<memory_context>')
    assert signature == [f'identity:{identity}', f'memory_preamble:{preamble}']


def test_identity_is_prepended_not_appended():
    ns = _load_helpers()
    messages = [{'role': 'system', 'content': 'PRIOR'}, {'role': 'user', 'content': 'hi'}]
    out, _ = ns['assemble_frozen_top_content'](messages, memory_enabled=False)
    assert out[0]['content'] == f'{ns["SYSTEM_IDENTITY_PROMPT"]}\nPRIOR'


def test_memory_disabled_has_no_preamble_vocabulary():
    """(d) conditional assembly: no ``<memory_context>`` vocabulary at all."""
    ns = _load_helpers()
    messages = [{'role': 'user', 'content': 'hi'}]
    out, signature = ns['assemble_frozen_top_content'](messages, memory_enabled=False)
    content = out[0]['content']
    assert ns['SYSTEM_IDENTITY_PROMPT'] in content
    assert '# Memory' not in content
    assert '<memory_context>' not in content
    assert '<memory_context_update>' not in content
    assert 'memory_preamble:' not in '\n'.join(signature)
    # The preamble constant is not referenced via a substring that could sneak in.
    assert ns['MEMORY_CONTEXT_PREAMBLE'] not in content


def test_enabling_memory_changes_the_injection_signature():
    ns = _load_helpers()
    off, sig_off = ns['assemble_frozen_top_content']([{'role': 'user', 'content': 'hi'}], memory_enabled=False)
    on, sig_on = ns['assemble_frozen_top_content']([{'role': 'user', 'content': 'hi'}], memory_enabled=True)
    assert sig_off != sig_on
    assert len(sig_on) == len(sig_off) + 1
    # Identity is in both; only the preamble entry differs.
    assert sig_on[0] == sig_off[0]


def test_preamble_frames_blocks_as_reference_not_instructions():
    """The deliberate v2 inversion: LOW authority, not "read and follow"."""
    ns = _load_helpers()
    preamble = ns['MEMORY_CONTEXT_PREAMBLE']
    assert 'background reference' in preamble
    assert 'not instructions to obey' in preamble
    assert 'Read and follow' not in preamble
    assert 'when they conflict, prefer the most recent' in preamble
    assert 'Never copy private data' in preamble


# ---------------------------------------------------------------------------
# P0-2 — per-request date line
# ---------------------------------------------------------------------------


def test_render_current_date_line_format():
    ns = _load_helpers()
    now = datetime(2026, 9, 24, 10, 30, tzinfo=ZoneInfo('Asia/Shanghai'))
    assert (
        ns['render_current_date_line'](timezone='Asia/Shanghai', now=now)
        == 'Current date: 2026-09-24 (Thursday), timezone Asia/Shanghai'
    )


def test_render_current_date_line_unknown_timezone_falls_back_to_utc():
    ns = _load_helpers()
    now = datetime(2026, 9, 24, 10, 30, tzinfo=ZoneInfo('UTC'))
    assert ns['render_current_date_line'](timezone='Not/AZone', now=now) == (
        'Current date: 2026-09-24 (Thursday), timezone UTC'
    )
    assert ns['render_current_date_line'](timezone=None, now=now).endswith('timezone UTC')


def test_date_line_is_appended_after_the_frozen_bytes():
    ns = _load_helpers()
    messages = [{'role': 'system', 'content': 'FROZEN_TOP'}, {'role': 'user', 'content': 'hi'}]
    now = datetime(2026, 9, 24, 10, 30, tzinfo=ZoneInfo('Asia/Shanghai'))
    out = ns['append_current_date_line'](
        messages, user={'timezone': 'Asia/Shanghai'}, metadata={}, now=now
    )
    assert out[0]['content'] == 'FROZEN_TOP\nCurrent date: 2026-09-24 (Thursday), timezone Asia/Shanghai'


def test_date_line_creates_a_system_message_when_absent():
    ns = _load_helpers()
    now = datetime(2026, 9, 24, 10, 30, tzinfo=ZoneInfo('Asia/Shanghai'))
    out = ns['append_current_date_line'](
        [{'role': 'user', 'content': 'hi'}], user=None, metadata={}, now=now
    )
    assert out[0]['role'] == 'system'
    assert out[0]['content'] == 'Current date: 2026-09-24 (Thursday), timezone UTC'


def test_date_line_is_skipped_for_internal_requests():
    ns = _load_helpers()
    now = datetime(2026, 9, 24, 10, 30, tzinfo=ZoneInfo('Asia/Shanghai'))
    out = ns['append_current_date_line'](
        [{'role': 'system', 'content': 'X'}], user=None, metadata={'internal': True}, now=now
    )
    assert out[0]['content'] == 'X'


# ---------------------------------------------------------------------------
# P0-2 wiring — every frozen-top rebuild re-appends the per-request date
# ---------------------------------------------------------------------------

MIDDLEWARE_PY = Path(__file__).resolve().parents[1] / 'utils' / 'middleware.py'


def _function_node(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {MIDDLEWARE_PY}')


def _call_lines(node: ast.AST, func_name: str) -> list[int]:
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if (isinstance(func, ast.Name) and func.id == func_name) or (
                isinstance(func, ast.Attribute) and func.attr == func_name
            ):
                lines.append(child.lineno)
    return lines


def test_streaming_tool_loop_reappends_date_after_system_restore():
    """The RAG restore in the tool-call loop rewinds the system message to
    ``metadata['system_prompt']`` (frozen bytes, no date). If the per-request
    date line is not re-appended there, continuation calls silently lose it.
    """
    tree = ast.parse(MIDDLEWARE_PY.read_text())
    handler = _function_node(tree, 'streaming_chat_response_handler')
    restores = _call_lines(handler, 'replace_system_message_content')
    appends = _call_lines(handler, 'append_current_date_line')
    assert restores, 'expected the RAG system-message restore in the streaming handler'
    assert appends, 'the streaming handler must re-append the per-request date after restoring the top'
    assert min(appends) > max(restores), f'restore@{restores} must precede append@{appends}'


def _system_prompt_assignment_lines(node: ast.AST) -> list[int]:
    lines = []
    for child in ast.walk(node):
        if not isinstance(child, (ast.Assign, ast.AugAssign)):
            continue
        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'metadata'
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == 'system_prompt'
            ):
                lines.append(child.lineno)
    return lines


def test_metadata_system_prompt_excludes_the_per_request_date():
    """P3 decision: ``metadata['system_prompt']`` is the persisted capture surface
    and deliberately stops at the frozen bytes. The per-request date is appended
    to the outgoing message *after* it, never folded back into metadata (which
    would persist per-request data). This is the invariant a future capture like
    ``temp_system_prompt_record.txt`` relies on.
    """
    tree = ast.parse(MIDDLEWARE_PY.read_text())
    payload = _function_node(tree, 'process_chat_payload')

    assigned = _system_prompt_assignment_lines(payload)
    date_calls = _call_lines(payload, 'append_current_date_line')
    assert assigned, "expected metadata['system_prompt'] assignments in process_chat_payload"
    assert date_calls, 'expected the per-request date append in process_chat_payload'
    assert max(assigned) < min(date_calls), (
        f'metadata system_prompt@{assigned} must be captured before the date append@{date_calls}'
    )

    # And the date line must never be assigned into metadata['system_prompt'].
    for child in ast.walk(payload):
        if isinstance(child, ast.Assign):
            called = getattr(child.value, 'func', None)
            func_name = getattr(called, 'id', None) or getattr(called, 'attr', None)
            if func_name != 'append_current_date_line':
                continue
            for target in child.targets:
                assert not (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'metadata'
                ), 'append_current_date_line must not be written into metadata'