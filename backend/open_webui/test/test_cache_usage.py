"""Focused unit tests for provider prompt-cache usage extraction (P2-1).

``open_webui.utils.response`` imports ``open_webui.utils.misc`` (aiohttp,
mimeparse, ...), so it is not importable in the shallow test tier. The two
helpers under test are pure, so we AST-extract them from source and stub the
few globals they touch — the same extraction pattern the middleware tests use.

The point of the guardrail is that the log line is stable and parseable, so the
tests pin both the extracted counters and the emitted ``cache_usage`` record.
"""

import ast
import json
from numbers import Number
from pathlib import Path

RESPONSE_PY = Path(__file__).resolve().parents[1] / 'utils' / 'response.py'
HELPERS = ('extract_cache_usage', 'log_cache_usage')
CACHE_CONSTANTS = ('CACHE_USAGE_KEYS', 'CACHE_USAGE_DETAIL_KEYS', 'CACHE_USAGE_DETAIL_CONTAINERS')


class _FakeLog:
    def __init__(self):
        self.records = []

    def info(self, fmt, *args):
        self.records.append(fmt % args)


class _FakeJSONCodec:
    @staticmethod
    def dumps(value, **kwargs):
        return json.dumps(value, **kwargs)


def _load_helpers():
    ns = {
        'Number': Number,
        'log': _FakeLog(),
        'JSONCodec': _FakeJSONCodec,
    }
    tree = ast.parse(RESPONSE_PY.read_text())
    found = set()
    for node in tree.body:
        is_constant = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in CACHE_CONSTANTS
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in HELPERS):
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(RESPONSE_PY), 'exec'), ns)
            found.add(node.targets[0].id if is_constant else node.name)
    assert found == set(HELPERS) | set(CACHE_CONSTANTS), f'missing: {(set(HELPERS) | set(CACHE_CONSTANTS)) - found}'
    return ns


def test_anthropic_top_level_counters():
    extract = _load_helpers()['extract_cache_usage']
    assert extract(
        {
            'input_tokens': 10,
            'cache_read_input_tokens': 2048,
            'cache_creation_input_tokens': 512,
        }
    ) == {
        'cache_read_input_tokens': 2048,
        'cache_creation_input_tokens': 512,
    }


def test_openai_nested_prompt_tokens_details():
    extract = _load_helpers()['extract_cache_usage']
    assert extract(
        {
            'prompt_tokens': 1000,
            'prompt_tokens_details': {'cached_tokens': 768, 'cache_write_tokens': 128},
        }
    ) == {
        'cached_tokens': 768,
        'cache_write_tokens': 128,
    }


def test_gemini_total_cached_tokens():
    extract = _load_helpers()['extract_cache_usage']
    assert extract({'total_cached_tokens': 4096}) == {'total_cached_tokens': 4096}


def test_top_level_and_nested_are_merged():
    extract = _load_helpers()['extract_cache_usage']
    assert extract(
        {
            'cache_read_input_tokens': 1024,
            'prompt_tokens_details': {'cached_tokens': 512},
        }
    ) == {
        'cache_read_input_tokens': 1024,
        'cached_tokens': 512,
    }


def test_missing_or_empty_returns_empty():
    extract = _load_helpers()['extract_cache_usage']
    assert extract(None) == {}
    assert extract({}) == {}
    assert extract({'input_tokens': 5, 'output_tokens': 2}) == {}


def test_non_numeric_and_bool_are_ignored():
    extract = _load_helpers()['extract_cache_usage']
    assert extract({'cached_tokens': '768', 'cache_read_input_tokens': True}) == {}
    assert extract({'cached_tokens': None}) == {}


def test_non_finite_and_complex_are_ignored_without_raising():
    extract = _load_helpers()['extract_cache_usage']
    # ``Number`` covers float('nan'/'inf') and complex, whose int() raises.
    assert extract({'cached_tokens': float('nan')}) == {}
    assert extract({'cached_tokens': float('inf')}) == {}
    assert extract({'cache_read_input_tokens': complex(1, 2)}) == {}
    assert extract({'prompt_tokens_details': {'cached_tokens': float('-inf')}}) == {}
    # A malformed counter must not hide a valid sibling.
    assert extract({'cached_tokens': float('nan'), 'cache_read_input_tokens': 7}) == {
        'cache_read_input_tokens': 7
    }


def test_non_dict_detail_container_is_ignored():
    extract = _load_helpers()['extract_cache_usage']
    assert extract({'prompt_tokens_details': None}) == {}
    assert extract({'prompt_tokens_details': 'nope'}) == {}


def test_log_cache_usage_is_parseable_and_returns_counters():
    ns = _load_helpers()
    counters = ns['log_cache_usage'](
        {'cache_read_input_tokens': 1024}, source='stream.chat', model='claude'
    )
    assert counters == {'cache_read_input_tokens': 1024}
    (record,) = ns['log'].records
    assert record.startswith('cache_usage source=stream.chat model=claude ')
    assert json.loads(record.split(' ', 3)[3]) == {'cache_read_input_tokens': 1024}


def test_log_cache_usage_is_silent_without_counters():
    ns = _load_helpers()
    assert ns['log_cache_usage']({'input_tokens': 3}) == {}
    assert ns['log'].records == []
