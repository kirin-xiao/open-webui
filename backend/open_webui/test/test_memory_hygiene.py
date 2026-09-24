"""Focused tests for memory-store hygiene (plan item P1-1).

Three mechanisms are pinned here:

1. Timestamp + attribution are stamped at write time (``memory_write_meta``)
   and rendered by ``memory_label`` as ``[YYYY-MM-DD][user|assistant]``, with a
   backward-compatible fallback for legacy rows that carry no ``meta``.
2. The background-review prompt/op set carries an explicit consolidation duty.
3. Attribution defaults: human writes are ``user``; tool/review writes are
   ``assistant`` unless the operation states otherwise, so assistant
   inferences (已修正-style self-corrections) are distinguishable from
   user-stated facts.

These are pure-helper tests plus the pydantic operation model; no DB or model
call is involved.
"""

import re
import time

from open_webui.routers.memories import MemoryOperationModel, UpdateMemoriesForm
from open_webui.utils import memory


class _Row:
    """Minimal stand-in for a MemoryModel row."""

    def __init__(self, content, path=None, meta=None, memory_id='m1', memory_type='context'):
        self.id = memory_id
        self.content = content
        self.path = path
        self.meta = meta
        self.type = memory_type
        self.created_at = 0
        self.updated_at = 0


class _LegacyRow:
    """A row shape from before ``meta`` existed at all (no attribute)."""

    def __init__(self, content, path=None):
        self.id = 'legacy'
        self.content = content
        self.path = path
        self.type = 'context'
        self.created_at = 0
        self.updated_at = 0


class _Results:
    """Duck-typed vector search result."""

    def __init__(self, ids, documents, distances):
        self.ids = [ids]
        self.documents = [documents]
        self.metadatas = [[{'path': None} for _ in ids]]
        self.distances = [distances]


# ---------------------------------------------------------------------------
# Write-time timestamp + attribution
# ---------------------------------------------------------------------------


def test_memory_date_is_machine_readable_and_localtime():
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', memory.memory_date())
    timestamp = 1700000000
    assert memory.memory_date(timestamp) == time.strftime('%Y-%m-%d', time.localtime(timestamp))


def test_write_meta_stamps_date_and_attribution_defaults():
    manual = memory.memory_write_meta(source='manual', timestamp=1700000000)
    assert manual['date'] == memory.memory_date(1700000000)
    assert manual['created_by'] == 'manual'
    assert manual['attribution'] == 'user'

    # A tool/review write never claims user provenance without evidence.
    review = memory.memory_write_meta(source='background_review')
    assert review['attribution'] == 'assistant'
    tool = memory.memory_write_meta(source='tool')
    assert tool['attribution'] == 'assistant'


def test_write_meta_explicit_attribution_wins():
    assert memory.memory_write_meta(source='background_review', attribution='user')['attribution'] == 'user'
    assert memory.memory_write_meta(source='tool', attribution='assistant')['attribution'] == 'assistant'
    # An invalid value is dropped and falls back by source rather than being
    # stored verbatim (or aborting the write).
    assert memory.memory_write_meta(source='manual', attribution='model')['attribution'] == 'user'
    assert memory.memory_write_meta(source='tool', attribution='model')['attribution'] == 'assistant'


# ---------------------------------------------------------------------------
# memory_label rendering + legacy fallback
# ---------------------------------------------------------------------------


def test_memory_label_renders_date_and_attribution():
    row = _Row('Likes tea', path='profile/drinks', meta={'date': '2026-09-24', 'attribution': 'user'})
    assert memory.memory_label(row) == '[2026-09-24][user] profile/drinks: Likes tea'


def test_memory_label_date_only_and_attribution_only():
    assert memory.memory_label(_Row('fact', meta={'date': '2026-09-24'})) == '[2026-09-24] fact'
    assert memory.memory_label(_Row('fact', meta={'attribution': 'assistant'})) == '[assistant] fact'


def test_memory_label_legacy_rows_unchanged():
    # No meta at all.
    assert memory.memory_label(_Row('plain')) == 'plain'
    assert memory.memory_label(_Row('with path', path='a/b')) == 'a/b: with path'
    # Pre-change rows that already carried a created_by but no date/attribution.
    assert memory.memory_label(_Row('old manual', meta={'created_by': 'manual'})) == 'old manual'
    # A row object predating the meta column entirely.
    assert memory.memory_label(_LegacyRow('very old', path='x')) == 'x: very old'
    # An out-of-shape date must not leak into the label.
    assert memory.memory_label(_Row('bad date', meta={'date': 'yesterday'})) == 'bad date'
    assert memory.memory_label(_Row('bad attr', meta={'attribution': 'robot'})) == 'bad attr'


def test_memory_label_distinguishes_assistant_inference_from_user_fact():
    user_fact = memory.memory_label(_Row('Position is 100 shares', meta={'date': '2026-09-24', 'attribution': 'user'}))
    inference = memory.memory_label(
        _Row('Position is 100 shares', meta={'date': '2026-09-24', 'attribution': 'assistant'})
    )
    assert user_fact != inference
    assert '[user]' in user_fact and '[assistant]' in inference


def test_collect_context_entries_render_meta_from_the_row():
    rows = [_Row('ETF position', meta={'date': '2026-09-24', 'attribution': 'assistant'})]
    results = _Results(['m1'], ['ETF position'], [0.9])
    entries = memory.collect_memory_entries(rows, results, [], 0.0)
    assert [entry.label for entry in entries] == ['[2026-09-24][assistant] ETF position']


# ---------------------------------------------------------------------------
# Operation model carries attribution through validation
# ---------------------------------------------------------------------------


def test_operation_attribution_survives_validation():
    form = UpdateMemoriesForm(
        operations=[
            {'action': 'add', 'content': 'new fact', 'attribution': 'user'},
            {'action': 'replace', 'id': 'm1', 'content': 'corrected', 'attribution': 'assistant'},
        ],
        source='background_review',
    )
    operations = memory.validate_memory_operations(form)
    assert [op['attribution'] for op in operations] == ['user', 'assistant']


def test_operation_attribution_defaults_to_none_and_tolerates_junk():
    model = MemoryOperationModel(action='add', content='x')
    assert model.model_dump()['attribution'] is None
    # Malformed provenance is tolerated at the model boundary and dropped
    # downstream, so one bad field cannot abort a whole review batch.
    junk = MemoryOperationModel(action='add', content='x', attribution='model')
    assert junk.model_dump()['attribution'] == 'model'
    assert memory.sanitize_memory_attribution(junk.attribution) is None
    fallback = memory.memory_write_meta(source='background_review', attribution=junk.attribution)
    assert fallback['attribution'] == 'assistant'


# ---------------------------------------------------------------------------
# Background-review prompt / op set carries the consolidation duty
# ---------------------------------------------------------------------------


def test_review_prompt_contains_consolidation_duty():
    prompt = memory.memory_review_prompt(
        existing_text='- id=m1 date=2026-09-20 attribution=user content=zero fill',
        transcript='user: actually it filled',
        today='2026-09-24',
    )
    assert memory.MEMORY_REVIEW_CONSOLIDATION_DUTY in prompt
    # Wrapping is cosmetic; check the duty's substance on normalized whitespace.
    flat = ' '.join(prompt.split())
    for needle in (
        'one active value per slot',
        'Recency wins',
        'delete the superseded text',
        'drop narration',
        'Never leave two entries that contradict each other',
        'Never label an assistant inference as "user"',
    ):
        assert needle in flat, needle
    # Attribution is part of the operation shape the reviewer must emit.
    assert '"attribution":"user|assistant"' in prompt
    assert "Today's date is 2026-09-24" in prompt


def test_review_prompt_includes_dates_for_recency():
    prompt = memory.memory_review_prompt(
        existing_text='- id=m1 date=2026-09-20 attribution=user content=old',
        transcript='user: new',
        today='2026-09-24',
    )
    # The prompt must surface the dates it needs to order conflicts.
    assert 'date=2026-09-20 attribution=user' in prompt


# ---------------------------------------------------------------------------
# Interactive tool schemas declare provenance (the durable attribution fix)
# ---------------------------------------------------------------------------


def _builtin_tool_function(name: str):
    """AST-extract one function definition from ``tools/builtin.py``.

    Parsed rather than imported: importing the module pulls the whole app and
    wipes ``backend/open_webui/static``.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / 'tools' / 'builtin.py').read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in tools/builtin.py')


def test_memory_tools_declare_attribution():
    """``add_memory`` / ``replace_memory_content`` carry a model-declared
    ``attribution`` parameter, and ``update_memory`` advertises per-op
    attribution in its docstring — so an assistant inference can be marked
    ``assistant`` and a relayed user fact ``user`` instead of being guessed
    from the transport.
    """
    for name in ('add_memory', 'replace_memory_content'):
        func = _builtin_tool_function(name)
        args = [arg.arg for arg in func.args.args]
        assert 'attribution' in args, name

    update = _builtin_tool_function('update_memory')
    doc = update.body[0].value.value  # module docstring expression
    assert '"attribution": "user"|"assistant"' in doc
    assert 'your own inference' in doc
