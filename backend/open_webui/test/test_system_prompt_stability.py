"""Tests for cache-stable chat injections (#30239).

The system prompt is regenerated every turn and the prompt-cache contract only
holds if the whole request prefix is byte-stable. Retrieval context and memory
deltas used to be spliced into the current user turn and never persisted, so the
next turn replayed the raw user message and diverged inside ``u_N``.

These tests pin the mitigation: the rendered text is frozen in ``chat.meta``
against the message id it augmented and replayed byte-identically at the same
position. They are focused unit tests over the pure ledger helpers plus the
anchor/dedup rules in ``add_memory_context``.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from open_webui.utils import chat_injections, memory
from open_webui.utils.chat_injections import (
    add_injection,
    apply_injections,
    collect_injections,
    normalize_injections,
    prune_injections,
    upsert_injection,
)

# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------


def test_normalize_drops_malformed_entries():
    raw = [
        {'anchor': 'a', 'text': '<memory_context>x</memory_context>'},
        {'anchor': 'b'},  # no text
        {'text': 'orphan'},  # no anchor
        'nonsense',
        None,
    ]
    assert normalize_injections(raw) == [
        {
            'source': '',
            'anchor': 'a',
            'position': 'prepend',
            'text': '<memory_context>x</memory_context>',
            'ids': [],
        }
    ]
    # A legacy/absent shape must never crash.
    assert normalize_injections(None) == []
    assert normalize_injections('delta') == []
    # A producer's default source is applied to legacy entries that predate it.
    assert normalize_injections(raw, source='memory')[0]['source'] == 'memory'


def test_upsert_is_one_per_anchor():
    ledger = upsert_injection([], anchor='u1', text='first')
    again = upsert_injection(ledger, anchor='u1', text='second')
    # Same anchor: the original frozen text wins, no duplicate.
    assert again == ledger
    assert len(again) == 1

    third = upsert_injection(again, anchor='u2', text='other')
    assert [entry['anchor'] for entry in third] == ['u1', 'u2']


def test_prune_removes_orphaned_anchors():
    ledger = [{'anchor': 'live', 'position': 'prepend', 'text': 'x', 'ids': []}]
    assert prune_injections(ledger, {'live', 'other'}) == ledger
    assert prune_injections(ledger, {'other'}) == []
    # Unknown chain must not wipe the ledger.
    assert prune_injections(ledger, set()) == ledger


# ---------------------------------------------------------------------------
# Replay fidelity — the cache criterion
# ---------------------------------------------------------------------------


def _user(message_id: str, content: str) -> dict:
    return {'id': message_id, 'role': 'user', 'content': content}


def test_replay_is_byte_identical_and_idempotent():
    frozen = 'CONTEXT_FROM_TURN_N'
    ledger = upsert_injection([], anchor='u1', text=frozen)

    turn_n = apply_injections([_user('u1', 'hello')], ledger)
    first = turn_n[0]['content']

    # Replaying the same ledger again (retry / continuation) must not double it.
    again = apply_injections([_user('u1', 'hello')], ledger)
    assert again[0]['content'] == first
    assert first == f'{frozen}\nhello'


def test_replay_uses_frozen_text_not_live_render():
    """A later edit/deletion/re-score cannot change what an earlier turn sent."""
    ledger = upsert_injection([], anchor='u1', text='OLD_LABEL')
    # Even if a producer would now render something different, replay is frozen.
    replayed = apply_injections([_user('u1', 'q')], ledger)
    assert replayed[0]['content'] == 'OLD_LABEL\nq'


def test_turn_n_is_a_prefix_of_turn_n_plus_one():
    """The actual prompt-cache contract: turn N's request is a strict prefix of
    turn N+1's (up to newly generated content)."""
    ledger_n = upsert_injection([], anchor='u1', text='D1')
    turn_n = apply_injections([_user('u1', 'q1')], ledger_n)
    request_n = [*turn_n, {'role': 'assistant', 'content': 'a1'}]

    ledger_n1 = upsert_injection(ledger_n, anchor='u2', text='D2')
    turn_n1 = apply_injections(
        [_user('u1', 'q1'), {'role': 'assistant', 'content': 'a1'}, _user('u2', 'q2')],
        ledger_n1,
    )
    request_n1 = [*turn_n1, {'role': 'assistant', 'content': 'a2'}]

    # Everything turn N sent is still there, elementwise equal.
    assert request_n1[: len(request_n)] == request_n


def test_collect_injections_order_is_memory_then_rag():
    """Replay order must match live order: memory is prepended first, then RAG,
    so RAG ends up outermost in the final text."""
    meta = {
        'memory_context': {'base': ['m1'], 'injections': [{'anchor': 'u1', 'text': 'mem'}]},
        'rag_injections': [{'anchor': 'u1', 'text': 'rag'}],
    }
    collected = collect_injections(meta)
    assert [entry['text'] for entry in collected] == ['mem', 'rag']

    messages = apply_injections([_user('u1', 'q')], collected)
    # memory prepends -> 'mem\nq'; rag prepends -> 'rag\nmem\nq'
    assert messages[0]['content'] == 'rag\nmem\nq'


def test_append_position():
    ledger = upsert_injection([], anchor='u1', text='tail', position='append')
    messages = apply_injections([_user('u1', 'head')], ledger)
    assert messages[0]['content'] == 'head\ntail'


# ---------------------------------------------------------------------------
# Memory ledger: selection + anchor dedup
# ---------------------------------------------------------------------------


class _MemoryRow:
    def __init__(self, memory_id, content, path=None, memory_type='context'):
        self.id = memory_id
        self.content = content
        self.path = path
        self.type = memory_type
        self.created_at = 0
        self.updated_at = 0


def _entry(memory_id, label='fact', score=1.0):
    return memory.MemoryEntry(id=memory_id, section='context', label=label, score=score, created_at=0)


def test_select_excludes_already_injected_ids():
    recovered = [_entry('m1'), _entry('m2'), _entry('m3')]
    state = {
        'base': ['m1'],
        'injections': [{'anchor': 'u1', 'position': 'prepend', 'text': 'd', 'ids': ['m2']}],
    }
    base_ids, new_ids = memory._select_memory_ids(recovered, state, {'m1': 1, 'm2': 1, 'm3': 1}, 'chat-1')
    assert base_ids == ['m1']
    # Only never-injected ids flow to a new tail injection.
    assert new_ids == ['m3']


def test_select_first_turn_puts_everything_in_base():
    recovered = [_entry('m1'), _entry('m2')]
    base_ids, new_ids = memory._select_memory_ids(
        recovered, {'base': [], 'injections': []}, {'m1': 1, 'm2': 1}, 'chat-1'
    )
    assert base_ids == ['m1', 'm2']
    assert new_ids == []


def test_select_unsaved_chat_has_no_ledger():
    recovered = [_entry('m1')]
    base_ids, new_ids = memory._select_memory_ids(recovered, {'base': [], 'injections': []}, {'m1': 1}, 'temporary:abc')
    assert base_ids == ['m1']
    assert new_ids == []


def test_legacy_delta_is_ignored():
    """Old chats carry a 'delta' key with no frozen text; it converges away."""
    state = {'base': ['m1'], 'injections': []}
    base_ids, new_ids = memory._select_memory_ids(
        [_entry('m1'), _entry('m2')],
        state,
        {'m1': 1, 'm2': 1},
        'chat-1',
    )
    assert base_ids == ['m1']
    assert new_ids == ['m2']  # previously-delta'd m2 is simply re-selected once


# ---------------------------------------------------------------------------
# add_memory_context wiring (no DB)
# ---------------------------------------------------------------------------


class _Config:
    @staticmethod
    async def get_many(*keys):
        return {}

    @staticmethod
    async def get(key, default=None):
        return default


@pytest.fixture
def memory_harness(monkeypatch):
    """Drive `add_memory_context` with stubbed config, retrieval and persistence."""
    saved: dict = {}

    rows = [_MemoryRow('m1', 'likes tea'), _MemoryRow('m2', 'speaks German')]

    class _Memories:
        @staticmethod
        async def get_memories_by_user_id(user_id):
            return rows

    async def _query_memory(request, form, user):
        return None

    async def _load_state(chat_id):
        return dict(saved) if saved else {'base': [], 'injections': [], 'chain_ids': {'u1', 'u2'}}

    async def _save_state(chat_id, state):
        saved.clear()
        saved.update(state)

    monkeypatch.setattr(memory, 'Config', _Config)
    monkeypatch.setattr(memory, 'Memories', _Memories)
    monkeypatch.setattr(memory, '_load_memory_state', _load_state)
    monkeypatch.setattr(memory, '_save_memory_state', _save_state)
    monkeypatch.setattr(memory, 'collect_memory_entries', lambda *a, **k: [_entry('m1'), _entry('m2')])

    return memory, saved


def _form_data(messages):
    return {'messages': messages}


async def _rows_with(*rows):
    return list(rows)


def _metadata(chat_id='chat-1', user_message_id='u1'):
    return {'chat_id': chat_id, 'user_message_id': user_message_id}


@pytest.mark.asyncio
async def test_first_turn_freezes_base_and_saves_no_injection(memory_harness):
    module, saved = memory_harness
    form_data = _form_data([{'id': 'u1', 'role': 'user', 'content': 'hi'}])
    result = await module.add_memory_context(None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata())

    # Base block goes to the system prompt, nothing to the tail.
    assert result['messages'][0]['role'] == 'system'
    assert saved['base'] == ['m1', 'm2']
    assert saved['injections'] == []
    assert len(result['messages']) == 2  # system + user


@pytest.mark.asyncio
async def test_continuation_does_not_emit_second_injection(memory_harness, monkeypatch):
    """A continuation whose last message is not a user turn must not attach a new
    block even when a newly-retrieved memory has never been injected."""
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [
                {'anchor': 'u1', 'position': 'prepend', 'text': '<memory_context>old</memory_context>', 'ids': ['m2']}
            ],
            'chain_ids': {'u1', 'a1'},
        }
    )
    # A third memory becomes relevant, but the trailing message is not a user
    # turn: the block must wait for the next user turn.
    third = _MemoryRow('m3', 'lives in Berlin')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(third))})
    )
    monkeypatch.setattr(module, 'collect_memory_entries', lambda *a, **k: [_entry('m1'), _entry('m2'), _entry('m3')])

    form_data = _form_data(
        [
            {'id': 'u1', 'role': 'user', 'content': 'hi'},
            {'id': 'a1', 'role': 'assistant', 'content': 'hello'},
        ]
    )
    result = await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u1')
    )
    assert len(saved['injections']) == 1
    # No new tail block: the assistant turn stays last and is untouched.
    assert result['messages'][-1] == {'id': 'a1', 'role': 'assistant', 'content': 'hello'}
    assert all('m3' not in entry['ids'] for entry in saved['injections'])


@pytest.mark.asyncio
async def test_new_turn_emits_one_frozen_injection(memory_harness):
    module, saved = memory_harness
    saved.update({'base': ['m1'], 'injections': []})
    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    result = await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u2')
    )

    assert len(saved['injections']) == 1
    injection = saved['injections'][0]
    assert injection['anchor'] == 'u2'
    assert injection['ids'] == ['m2']
    # The current turn carries the frozen text.
    assert injection['text'] in result['messages'][-1]['content']


@pytest.mark.asyncio
async def test_orphaned_injection_is_pruned_on_next_write(memory_harness):
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [
                {'anchor': 'deleted', 'position': 'prepend', 'text': 'old', 'ids': ['m9']},
                {'anchor': 'u1', 'position': 'prepend', 'text': 'kept', 'ids': ['m2']},
            ],
            'chain_ids': {'u1', 'u2'},  # 'deleted' is off the active chain
        }
    )
    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u2')
    )
    assert [injection['anchor'] for injection in saved['injections']] == ['u1']


@pytest.mark.asyncio
async def test_new_turn_is_not_pruned_by_stale_chain(memory_harness):
    """The current anchor is not in stored history yet; pruning must run before
    recording the turn or the freshly emitted injection is dropped immediately."""
    module, saved = memory_harness
    saved.update({'base': ['m1'], 'injections': [], 'chain_ids': {'u1'}})
    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u2')
    )
    assert [injection['anchor'] for injection in saved['injections']] == ['u2']


@pytest.mark.asyncio
async def test_load_memory_state_reads_legacy_delta_without_replaying_it(monkeypatch):
    """A pre-#30239 chat has only a 'delta' list; it must load as empty injections
    (nothing to replay) without crashing, and keep its base."""
    from open_webui.models.chat_messages import ChatMessages
    from open_webui.models.chats import Chats

    class _Chat:
        meta = {'memory_context': {'base': ['m1'], 'delta': ['m2']}}

    async def _get_chat(chat_id):
        return _Chat()

    async def _messages_map(chat_id):
        return {'u1': {'id': 'u1', 'role': 'user'}, 'a1': {'id': 'a1', 'role': 'assistant'}}

    monkeypatch.setattr(Chats, 'get_chat_by_id', _get_chat)
    monkeypatch.setattr(ChatMessages, 'get_messages_map_by_chat_id', _messages_map)

    state = await memory._load_memory_state('chat-1')
    assert state['base'] == ['m1']
    assert state['injections'] == []  # legacy delta converges away
    assert state['chain_ids'] == {'u1', 'a1'}


@pytest.mark.asyncio
async def test_load_memory_state_prunes_from_chat_message_rows(monkeypatch):
    """chain_ids must come from the same source replay reads (chat_message rows),
    so a stale JSON blob cannot make a live anchor look orphaned."""
    from open_webui.models.chat_messages import ChatMessages
    from open_webui.models.chats import Chats

    class _Chat:
        meta = {
            'memory_context': {
                'base': [],
                'injections': [
                    {'anchor': 'live', 'position': 'prepend', 'text': 'x', 'ids': []},
                    {'anchor': 'gone', 'position': 'prepend', 'text': 'y', 'ids': []},
                ],
            }
        }

    async def _get_chat(chat_id):
        return _Chat()

    async def _messages_map(chat_id):
        return {'live': {'id': 'live', 'role': 'user'}}

    monkeypatch.setattr(Chats, 'get_chat_by_id', _get_chat)
    monkeypatch.setattr(ChatMessages, 'get_messages_map_by_chat_id', _messages_map)

    state = await memory._load_memory_state('chat-1')
    assert [entry['anchor'] for entry in state['injections']] == ['live', 'gone']
    assert state['chain_ids'] == {'live'}


@pytest.mark.asyncio
async def test_unsaved_chat_writes_no_meta(memory_harness):
    module, saved = memory_harness
    form_data = _form_data([{'id': 'u1', 'role': 'user', 'content': 'hi'}])
    result = await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(chat_id='temporary:xyz')
    )
    # Legacy behavior: everything lands in the system prompt, nothing persisted.
    assert saved == {}
    assert result['messages'][0]['role'] == 'system'
    assert len(result['messages']) == 2


@pytest.mark.asyncio
async def test_legacy_delta_state_does_not_crash(memory_harness, monkeypatch):
    module, saved = memory_harness
    saved.update({'delta': ['m2']})  # pre-#30239 shape, no base/injections

    async def _load_legacy(chat_id):
        return {'base': ['m1'], 'injections': [], 'chain_ids': set()}

    monkeypatch.setattr(module, '_load_memory_state', _load_legacy)
    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    result = await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u2')
    )
    assert any(message['role'] == 'system' for message in result['messages'])
    # m2 was never frozen; it is re-selected and emitted once.
    assert saved['injections'][0]['ids'] == ['m2']


# ---------------------------------------------------------------------------
# Idempotent replay covers the live-injection double-apply case
# ---------------------------------------------------------------------------


def test_live_injection_is_not_doubled_by_replay():
    """The current turn is written live and recorded; replaying history plus the
    live ledger must not splice the same block twice."""
    ledger = upsert_injection([], anchor='u1', text='BLOCK')
    live = apply_injections([_user('u1', 'q')], ledger)  # applied live at emit time
    replayed = apply_injections(live, ledger)  # same request replays its own ledger
    assert replayed[0]['content'].count('BLOCK') == 1


# ---------------------------------------------------------------------------
# Anchor-authoritative live placement and fit-driven emitted ids
# ---------------------------------------------------------------------------


def test_attach_injection_targets_the_anchor_not_the_tail():
    """A guided-regeneration prompt (transient user turn, no id) sits last; the
    block must land on the recorded anchor instead."""
    messages = [
        _user('u1', 'original'),
        {'role': 'assistant', 'content': 'prior answer'},
        {'role': 'user', 'content': 'regeneration prompt'},  # no id
    ]
    chat_injections.attach_injection(messages, 'u1', 'BLOCK')
    assert messages[0]['content'] == 'BLOCK\noriginal'
    assert messages[-1]['content'] == 'regeneration prompt'
    assert messages[-1].get('id') is None


def test_attach_injection_is_idempotent():
    messages = [_user('u1', 'q')]
    chat_injections.attach_injection(messages, 'u1', 'BLOCK')
    chat_injections.attach_injection(messages, 'u1', 'BLOCK')
    assert messages[0]['content'].count('BLOCK') == 1


def test_fit_reports_truncated_entry_as_emitted():
    """An oversized lone entry is truncated but still emitted; its id must be
    recorded so it is not re-selected and re-emitted every turn."""
    entry = _entry('m1', label='x' * 500)
    text, dropped, emitted = memory.render_memory_context([entry], {'user': 250, 'context': 250})
    assert text
    assert emitted == ['m1']
    assert dropped['context'] == 0


def test_fit_does_not_mark_dropped_entry_as_emitted():
    entries = [_entry('m1', label='a' * 200), _entry('m2', label='b' * 200)]
    text, dropped, emitted = memory.render_memory_context(entries, {'user': 250, 'context': 250})
    # First entry fits (truncated), second is dropped and must stay re-selectable.
    assert emitted == ['m1']
    assert dropped['context'] == 1
    assert 'm2' not in emitted


# ---------------------------------------------------------------------------
# RAG ledger: same-anchor dedup and cross-turn replay
# ---------------------------------------------------------------------------


def _rag_meta(anchor='u1', text='RAG_BLOCK'):
    return {'rag_injections': [{'anchor': anchor, 'position': 'prepend', 'text': text, 'ids': []}]}


def test_rag_injection_replays_at_original_position():
    turn_n = apply_injections([_user('u1', 'q1')], collect_injections(_rag_meta()))
    assert turn_n[0]['content'] == 'RAG_BLOCK\nq1'

    # Turn N+1: raw DB history for u1 (no RAG), replayed from the same ledger.
    turn_n1 = apply_injections(
        [_user('u1', 'q1'), {'role': 'assistant', 'content': 'a1'}, _user('u2', 'q2')],
        collect_injections(_rag_meta()),
    )
    assert turn_n1[0]['content'] == turn_n[0]['content']
    assert turn_n1[-1]['content'] == 'q2'  # current turn untouched by the old entry


@pytest.mark.asyncio
async def test_rag_record_dedups_per_anchor(monkeypatch):
    """A retried turn must not record (or splice) a second RAG block."""
    module = chat_injections
    saved = {'rag_injections': _rag_meta()['rag_injections']}

    async def _load_meta(chat_id):
        return saved

    async def _save(chat_id, key, value):
        saved[key] = value
        return True

    monkeypatch.setattr(module, 'load_chat_meta', _load_meta)
    monkeypatch.setattr(module, 'save_meta_key', _save)

    _, recorded = await module.record_rag_injection('chat-1', anchor='u1', text='RAG_BLOCK_NEW')
    assert recorded is False  # already frozen; caller must not splice again
    assert len(saved['rag_injections']) == 1
    assert saved['rag_injections'][0]['text'] == 'RAG_BLOCK'

    _, recorded = await module.record_rag_injection('chat-1', anchor='u2', text='SECOND')
    assert recorded is True
    assert [entry['anchor'] for entry in saved['rag_injections']] == ['u1', 'u2']


@pytest.mark.asyncio
async def test_rag_record_noop_for_unsaved_chat(monkeypatch):
    module = chat_injections

    async def _save(chat_id, key, value):  # pragma: no cover - must not run
        raise AssertionError('unsaved chats must not persist meta')

    monkeypatch.setattr(module, 'save_meta_key', _save)
    await module.record_rag_injection('temporary:xyz', anchor='u1', text='x')


# ---------------------------------------------------------------------------
# Phase 2 — system-prompt baseline freeze
# ---------------------------------------------------------------------------


from open_webui.utils import system_baseline  # noqa: E402


class _BaselineStore:
    """In-memory stand-in for a chat's meta, shared by load/save."""

    def __init__(self):
        self.meta: dict = {}
        self.saves = 0

    def install(self, monkeypatch):
        async def _load(chat_id):
            return system_baseline.normalize_baseline(self.meta.get(system_baseline.SYSTEM_BASELINE_KEY))

        async def _save(chat_id, baseline):
            self.meta[system_baseline.SYSTEM_BASELINE_KEY] = baseline
            self.saves += 1

        monkeypatch.setattr(system_baseline, '_load_baseline', _load)
        monkeypatch.setattr(system_baseline, '_save_baseline', _save)


def _inputs_hash(**overrides):
    values = dict(
        model_id='m',
        model_system=None,
        chat_system='You are helpful.',
        folder_system=None,
        features={},
        tool_ids=[],
        skill_ids=[],
        terminal_id=None,
        tool_servers=[],
        direct=False,
    )
    values.update(overrides)
    return system_baseline.compute_inputs_hash(**values)


def _system(content):
    return [{'role': 'system', 'content': content}, _user('u1', 'hi')]


@pytest.mark.asyncio
async def test_baseline_freezes_once_and_replays_bytes(monkeypatch):
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()

    # Turn 1 renders and freezes.
    turn1 = _system('You are helpful.\nToday is 2026-09-22.')
    frozen = await system_baseline.apply_system_baseline('chat-1', turn1, {}, inputs_hash=inputs)
    assert frozen == 'You are helpful.\nToday is 2026-09-22.'
    assert store.saves == 1

    # Turn 2 renders a drifted top (date roll-over); the frozen bytes win.
    turn2 = _system('You are helpful.\nToday is 2026-09-23.')
    replayed = await system_baseline.apply_system_baseline('chat-1', turn2, {}, inputs_hash=inputs)
    assert replayed == frozen
    assert turn2[0]['content'] == frozen
    assert store.saves == 1  # no re-baseline


@pytest.mark.asyncio
async def test_baseline_rebaselines_on_changed_inputs(monkeypatch):
    store = _BaselineStore()
    store.install(monkeypatch)

    await system_baseline.apply_system_baseline(
        'chat-1', _system('Old prompt'), {}, inputs_hash=_inputs_hash(chat_system='Old prompt')
    )
    new_inputs = _inputs_hash(chat_system='New prompt')
    replayed = await system_baseline.apply_system_baseline('chat-1', _system('New prompt'), {}, inputs_hash=new_inputs)
    assert replayed == 'New prompt'
    assert store.saves == 2
    assert store.meta[system_baseline.SYSTEM_BASELINE_KEY]['inputs_hash'] == new_inputs


@pytest.mark.asyncio
async def test_baseline_migrates_legacy_conversation_summary(monkeypatch):
    """A pre-rewrite baseline frozen with the old summary marker is dropped once.

    Compaction no longer touches the system top, so the stale summary must not
    shadow the marker-free live render (and the new boundary checkpoint) forever.
    """
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()

    legacy_content = 'base\n[CONVERSATION SUMMARY]\nold summary'
    store.meta[system_baseline.SYSTEM_BASELINE_KEY] = {
        'inputs_hash': inputs,
        'hash': system_baseline._sha256(legacy_content),
        'content': legacy_content,
    }

    replayed = await system_baseline.apply_system_baseline('chat-1', _system('base'), {}, inputs_hash=inputs)
    assert replayed == 'base'
    assert system_baseline.LEGACY_SUMMARY_MARKER not in replayed
    assert store.saves == 1  # migrated once, not replayed forever

    # The migrated baseline now freezes normally: same inputs replay the bytes.
    replayed_again = await system_baseline.apply_system_baseline('chat-1', _system('base'), {}, inputs_hash=inputs)
    assert replayed_again == 'base'
    assert store.saves == 1


@pytest.mark.asyncio
async def test_baseline_migration_clears_legacy_when_live_top_empty(monkeypatch):
    """A legacy marker baseline is cleared even when there is nothing to re-freeze."""
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()
    legacy_content = '[CONVERSATION SUMMARY]\nonly summary'
    store.meta[system_baseline.SYSTEM_BASELINE_KEY] = {
        'inputs_hash': inputs,
        'hash': system_baseline._sha256(legacy_content),
        'content': legacy_content,
    }

    # No system message at all -> nothing to freeze, but the stale baseline must go.
    assert (
        await system_baseline.apply_system_baseline(
            'chat-1', [{'role': 'user', 'content': 'hi'}], {}, inputs_hash=inputs
        )
        is None
    )
    assert store.meta[system_baseline.SYSTEM_BASELINE_KEY] is None
    assert store.saves == 1

    # The migration does not repeat on the next request.
    await system_baseline.apply_system_baseline(
        'chat-1', [{'role': 'user', 'content': 'hi'}], {}, inputs_hash=inputs
    )
    assert store.saves == 1


def test_strip_legacy_summary_marker_removes_trailing_block():
    assert system_baseline.strip_legacy_summary_marker('base') == 'base'
    assert (
        system_baseline.strip_legacy_summary_marker('base\n[CONVERSATION SUMMARY]\nsummary') == 'base'
    )
    assert system_baseline.strip_legacy_summary_marker('[CONVERSATION SUMMARY]\nonly') == ''


@pytest.mark.asyncio
async def test_baseline_skips_unsaved_internal_and_compare(monkeypatch):
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()

    assert await system_baseline.apply_system_baseline('temporary:x', _system('s'), {}, inputs_hash=inputs) is None
    assert (
        await system_baseline.apply_system_baseline('chat-1', _system('s'), {'internal': True}, inputs_hash=inputs)
        is None
    )
    assert (
        await system_baseline.apply_system_baseline('chat-1', _system('s'), {'compare_mode': True}, inputs_hash=inputs)
        is None
    )
    assert store.saves == 0


@pytest.mark.asyncio
async def test_baseline_normalize_rejects_junk():
    assert system_baseline.normalize_baseline(None) is None
    assert system_baseline.normalize_baseline({'content': 'x'}) is None
    assert system_baseline.normalize_baseline({'inputs_hash': 'h', 'content': ''}) is None
    good = system_baseline.normalize_baseline({'inputs_hash': 'h', 'content': 'body'})
    assert good['content'] == 'body'
    assert good['hash'] == system_baseline._sha256('body')


def test_inputs_hash_ignores_rendered_drift():
    """Silent drift must not move the hash; user actions must."""
    base = _inputs_hash()
    assert base == _inputs_hash()
    assert base != _inputs_hash(chat_system='changed')
    assert base != _inputs_hash(model_id='other')
    assert base != _inputs_hash(tool_ids=['t1'])
    assert base != _inputs_hash(features={'memory': True})
    # Ordering of id lists is not a user action.
    assert _inputs_hash(tool_ids=['a', 'b']) == _inputs_hash(tool_ids=['b', 'a'])


def test_inputs_hash_covers_every_raw_source():
    base = _inputs_hash()
    assert base != _inputs_hash(model_system='model default')
    assert base != _inputs_hash(folder_system='folder prompt')
    assert base != _inputs_hash(skill_ids=['s1'])
    assert base != _inputs_hash(terminal_id='tty-1')
    assert base != _inputs_hash(tool_servers=[{'id': 'srv'}])
    assert base != _inputs_hash(direct=True)
    # Phase 3: the remaining user-action sources that reach the frozen top.
    assert base != _inputs_hash(use_builtin_tools=True)
    assert base != _inputs_hash(is_note_chat=True)
    assert base != _inputs_hash(function_calling='legacy')
    assert base != _inputs_hash(attached_knowledge=[{'type': 'collection', 'id': 'c1'}])
    assert base != _inputs_hash(system_injection_signature='skill:s1:body')
    # Ordering of attached knowledge / tool servers is not a user action.
    assert _inputs_hash(attached_knowledge=[{'type': 'a', 'id': '1'}, {'type': 'b', 'id': '2'}]) == _inputs_hash(
        attached_knowledge=[{'type': 'b', 'id': '2'}, {'type': 'a', 'id': '1'}]
    )


@pytest.mark.asyncio
async def test_baseline_replays_even_when_live_render_drops_the_system_message(monkeypatch):
    """The memory base can empty between turns, leaving no live system message;
    the frozen top must still be replayed or the prefix diverges."""
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()

    turn1 = _system('<memory_context>\n- likes tea\n</memory_context>')
    frozen = await system_baseline.apply_system_baseline('chat-1', turn1, {}, inputs_hash=inputs)

    # Turn 2 renders no system message at all (base emptied).
    turn2 = [_user('u1', 'hi'), {'role': 'assistant', 'content': 'a'}]
    replayed = await system_baseline.apply_system_baseline('chat-1', turn2, {}, inputs_hash=inputs)
    assert replayed == frozen
    assert turn2[0] == {'role': 'system', 'content': frozen}
    assert store.saves == 1


@pytest.mark.asyncio
async def test_empty_rerender_clears_a_stale_baseline(monkeypatch):
    store = _BaselineStore()
    store.install(monkeypatch)

    await system_baseline.apply_system_baseline(
        'chat-1', _system('old top'), {}, inputs_hash=_inputs_hash(chat_system='old top')
    )
    assert store.meta[system_baseline.SYSTEM_BASELINE_KEY] is not None

    # Inputs changed and the live render legitimately produced nothing: clear the
    # stale baseline so it cannot resurface later under the new inputs.
    empty = [_user('u1', 'hi')]
    assert (
        await system_baseline.apply_system_baseline('chat-1', empty, {}, inputs_hash=_inputs_hash(chat_system='new'))
        is None
    )
    assert store.meta[system_baseline.SYSTEM_BASELINE_KEY] is None


# ---------------------------------------------------------------------------
# Phase 2 — memory-edit / deletion tail updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_edit_emits_tail_update_only_once(memory_harness, monkeypatch):
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'u2'},
        }
    )
    # The row was edited: same id, new label.
    edited = _MemoryRow('m1', 'likes coffee')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(edited))})
    )
    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    result = await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u2')
    )
    blocks = [entry['text'] for entry in saved['injections'] if 'memory_context_update' in entry['text']]
    assert blocks and 'likes coffee' in blocks[0]
    assert result['messages'][-1]['content'].endswith(blocks[0])
    # The label map advances, so the next turn does not repeat it.
    assert saved['base_labels']['m1'] == 'likes coffee'


@pytest.mark.asyncio
async def test_base_deletion_emits_removed_notice_but_keeps_top(memory_harness, monkeypatch):
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'u2'},
        }
    )
    # m1 is gone from the store; only an unrelated row remains.
    row = _MemoryRow('m2', 'speaks German')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(row))})
    )
    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u2')
    )
    blocks = [entry['text'] for entry in saved['injections'] if 'memory_context_update' in entry['text']]
    assert blocks and 'likes tea no longer applies. Disregard it.' in blocks[0]
    assert saved['base'] == []  # converged out of base, top left frozen


# ---------------------------------------------------------------------------
# P1-2 — supersession narration for base updates
# ---------------------------------------------------------------------------


def test_base_edit_narration_supersedes_the_earlier_entry():
    """The subject names the *frozen* entry's slot (path), not its content."""
    row = _MemoryRow('m1', 'holds 10.3万')
    row.path = '投资/资金结构'
    row.meta = {'date': '2026-09-24', 'attribution': 'user'}
    old = '[2026-09-20][user] 投资/资金结构: holds 5万'
    updates, labels = memory._collect_base_updates({'base': ['m1'], 'base_labels': {'m1': old}}, {'m1': row}, ['m1'])
    assert updates == [
        '- Updated (supersedes the earlier entry about 投资/资金结构): '
        '[2026-09-24][user] 投资/资金结构: holds 10.3万'
    ]
    assert labels == {'m1': '[2026-09-24][user] 投资/资金结构: holds 10.3万'}


def test_base_deletion_narration_says_no_longer_applies():
    old = '[2026-09-20][user] 投资/资金结构: holds 5万'
    updates, labels = memory._collect_base_updates({'base': ['m1'], 'base_labels': {'m1': old}}, {}, ['m1'])
    assert updates == [f'- {old} no longer applies. Disregard it.']
    assert labels == {}


def test_supersession_subject_is_short_and_path_preferred():
    subject = memory.memory_supersession_subject('[2026-09-20][user] 投资/资金结构: holds 5万')
    assert subject == '投资/资金结构'
    # No content leaks when a path names the entry.
    assert 'holds 5万' not in subject
    # A pathless row falls back to its content head and is truncated, not echoed whole.
    subject = memory.memory_supersession_subject('x' * 200)
    assert len(subject) == memory._BASE_UPDATE_SUBJECT_MAX
    assert subject.endswith('…')


@pytest.mark.asyncio
async def test_base_update_narration_replays_byte_identically(memory_harness, monkeypatch):
    """The supersession text is frozen in the ledger and replays byte-for-byte."""
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'u2'},
        }
    )
    edited = _MemoryRow('m1', 'likes coffee')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(edited))})
    )
    await module.add_memory_context(
        None,
        _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}]),
        type('U', (), {'id': 'u'})(),
        {'id': 'm'},
        _metadata(user_message_id='u2'),
    )
    block = [entry['text'] for entry in saved['injections'] if 'memory_context_update' in entry['text']][0]
    assert 'Updated (supersedes the earlier entry about likes tea): likes coffee' in block

    ledger = normalize_injections(saved['injections'])
    first = apply_injections([_user('u2', 'next')], ledger)[0]['content']
    second = apply_injections([_user('u2', 'next')], ledger)[0]['content']
    assert first == second
    assert block in first


@pytest.mark.asyncio
async def test_base_update_defers_on_continuation(memory_harness, monkeypatch):
    """No trailing user turn: the diff stays pending (labels not advanced)."""
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'a1'},
        }
    )
    edited = _MemoryRow('m1', 'likes coffee')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(edited))})
    )
    form_data = _form_data(
        [
            {'id': 'u1', 'role': 'user', 'content': 'hi'},
            {'id': 'a1', 'role': 'assistant', 'content': 'hello'},
        ]
    )
    await module.add_memory_context(
        None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id='u1')
    )
    assert not any('memory_context_update' in entry['text'] for entry in saved['injections'])
    assert saved['base_labels']['m1'] == 'likes tea'  # still pending


@pytest.mark.asyncio
async def test_phase1_base_emptied_with_injections_is_not_reestablished(monkeypatch):
    """A Phase-1 chat (no ``established`` marker) whose base was emptied by
    deletions must not re-bake ids that already replay at the tail — that would
    duplicate context. ``injections`` present counts as established."""
    from open_webui.models.chat_messages import ChatMessages
    from open_webui.models.chats import Chats

    class _Chat:
        meta = {
            'memory_context': {
                'base': [],
                'injections': [
                    {'anchor': 'u1', 'position': 'prepend', 'text': '<memory_context>x</memory_context>', 'ids': ['m1']}
                ],
            }
        }

    async def _get_chat(chat_id):
        return _Chat()

    async def _messages_map(chat_id):
        return {'u1': {'id': 'u1', 'role': 'user'}}

    monkeypatch.setattr(Chats, 'get_chat_by_id', _get_chat)
    monkeypatch.setattr(ChatMessages, 'get_messages_map_by_chat_id', _messages_map)

    state = await memory._load_memory_state('chat-1')
    assert state['established'] is True

    # m1 already replays at the tail; it must not become a new base row.
    base_ids, new_ids = memory._select_memory_ids([_entry('m1'), _entry('m2')], state, {'m1': 1, 'm2': 1}, 'chat-1')
    assert base_ids == []
    assert new_ids == ['m2']


@pytest.mark.asyncio
async def test_two_turn_system_bytes_stable_while_memory_edited(memory_harness, monkeypatch):
    """The Phase-2 cache criterion end to end: an edit to a frozen base row leaves
    the *frozen* system bytes untouched and delivers the change once at the tail."""
    module, saved = memory_harness
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()

    saved.update(
        {
            'base': ['m1'],
            'injections': [],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'u2', 'u3'},
        }
    )

    async def _run(user_message_id, content_label):
        row = _MemoryRow('m1', content_label)
        monkeypatch.setattr(
            module,
            'Memories',
            type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(row))}),
        )
        monkeypatch.setattr(module, 'collect_memory_entries', lambda *a, **k: [_entry('m1')])
        form_data = _form_data([{'id': user_message_id, 'role': 'user', 'content': 'q'}])
        form_data = await module.add_memory_context(
            None, form_data, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata(user_message_id=user_message_id)
        )
        frozen = await system_baseline.apply_system_baseline('chat-1', form_data['messages'], {}, inputs_hash=inputs)
        return frozen

    # Turn 2 establishes the baseline (frozen from the then-current render).
    turn2 = await _run('u2', 'likes coffee')
    assert '- likes coffee' in turn2
    assert store.saves == 1

    # Turn 3: the row is edited again, but the frozen top must not change.
    turn3 = await _run('u3', 'likes espresso')
    assert turn3 == turn2
    assert store.saves == 1

    # Two edits, each delivered exactly once at the tail, in chronological order.
    update_blocks = [entry for entry in saved['injections'] if 'memory_context_update' in entry['text']]
    assert [entry['anchor'] for entry in update_blocks] == ['u2', 'u3']
    assert 'likes coffee' in update_blocks[0]['text']
    assert 'likes espresso' in update_blocks[1]['text']


def test_add_injection_allows_multiple_per_anchor():
    ledger = chat_injections.add_injection([], anchor='u1', text='delta')
    ledger = chat_injections.add_injection(ledger, anchor='u1', text='edit')
    assert [entry['text'] for entry in ledger] == ['delta', 'edit']
    # Retry/duplicate text is a no-op.
    assert chat_injections.add_injection(ledger, anchor='u1', text='edit') == ledger


# ---------------------------------------------------------------------------
# Phase 3 — generalized tail channel
# ---------------------------------------------------------------------------


def test_collect_injections_order_is_memory_rag_then_plugins():
    """Replay order must match live order: memory, then RAG, then plugin blocks."""
    meta = {
        'memory_context': {'base': [], 'injections': [{'anchor': 'u1', 'text': 'mem'}]},
        'rag_injections': [{'anchor': 'u1', 'text': 'rag'}],
        'message_injections': [{'anchor': 'u1', 'text': 'plugin', 'source': 'my_filter'}],
    }
    assert [entry['text'] for entry in collect_injections(meta)] == ['mem', 'rag', 'plugin']
    messages = apply_injections([_user('u1', 'q')], collect_injections(meta))
    # Each prepends, so the last applied (plugin) is outermost.
    assert messages[0]['content'] == 'plugin\nrag\nmem\nq'


def test_source_tags_are_applied_per_producer():
    """A producer's entries carry its label, including legacy entries with none."""
    meta = {
        'memory_context': {'base': [], 'injections': [{'anchor': 'u1', 'text': 'm'}]},
        'rag_injections': [{'anchor': 'u1', 'text': 'r'}],
    }
    assert [entry['source'] for entry in chat_injections.memory_injections(meta)] == ['memory']
    assert [entry['source'] for entry in chat_injections.rag_injections(meta)] == ['rag']
    # An explicitly stored source wins over the producer default.
    tagged = {'rag_injections': [{'anchor': 'u1', 'text': 'r', 'source': 'custom'}]}
    assert chat_injections.rag_injections(tagged)[0]['source'] == 'custom'


def test_plugin_message_injection_replays_at_original_position():
    meta = {'message_injections': [{'anchor': 'u1', 'position': 'append', 'text': 'PLUGIN', 'source': 'f'}]}
    turn_n = apply_injections([_user('u1', 'q1')], collect_injections(meta))
    assert turn_n[0]['content'] == 'q1\nPLUGIN'

    turn_n1 = apply_injections(
        [_user('u1', 'q1'), {'role': 'assistant', 'content': 'a1'}, _user('u2', 'q2')],
        collect_injections(meta),
    )
    assert turn_n1[0]['content'] == turn_n[0]['content']  # byte-stable prefix
    assert turn_n1[-1]['content'] == 'q2'


@pytest.mark.asyncio
async def test_record_message_injection_dedups_by_anchor_and_text(monkeypatch):
    module = chat_injections
    saved: dict = {}

    async def _load_meta(chat_id):
        return dict(saved)

    async def _save(chat_id, key, value):
        saved[key] = value
        return True

    monkeypatch.setattr(module, 'load_chat_meta', _load_meta)
    monkeypatch.setattr(module, 'save_meta_key', _save)

    _, recorded = await module.record_message_injection(
        'chat-1', anchor='u1', text='BLOCK', source='my_filter'
    )
    assert recorded is True
    assert saved[chat_injections.MESSAGE_INJECTIONS_KEY][0]['source'] == 'my_filter'

    # Retrying the same block is a no-op (caller must not splice twice).
    _, recorded = await module.record_message_injection('chat-1', anchor='u1', text='BLOCK')
    assert recorded is False
    assert len(saved[chat_injections.MESSAGE_INJECTIONS_KEY]) == 1

    # A different block on the same anchor is allowed.
    _, recorded = await module.record_message_injection('chat-1', anchor='u1', text='SECOND')
    assert recorded is True
    assert [entry['text'] for entry in saved[chat_injections.MESSAGE_INJECTIONS_KEY]] == ['BLOCK', 'SECOND']


@pytest.mark.asyncio
async def test_record_message_injection_noop_for_unsaved_chat(monkeypatch):
    module = chat_injections

    async def _save(chat_id, key, value):  # pragma: no cover - must not run
        raise AssertionError('unsaved chats must not persist meta')

    monkeypatch.setattr(module, 'save_meta_key', _save)
    _, recorded = await module.record_message_injection('temporary:xyz', anchor='u1', text='x')
    assert recorded is False


@pytest.mark.asyncio
async def test_record_message_injection_replace_supersedes_same_source(monkeypatch):
    """A re-run turn that produced a different outcome supersedes the old one
    (e.g. image generation failed after a retry), instead of accumulating a stale
    block that replays alongside the new one."""
    module = chat_injections
    saved: dict = {}

    async def _load_meta(chat_id):
        return dict(saved)

    async def _save(chat_id, key, value):
        saved[key] = value
        return True

    monkeypatch.setattr(module, 'load_chat_meta', _load_meta)
    monkeypatch.setattr(module, 'save_meta_key', _save)

    await module.record_message_injection(
        'chat-1', anchor='u1', text='created', source='image_generation', replace=True
    )
    await module.record_message_injection(
        'chat-1', anchor='u1', text='failed', source='image_generation', replace=True
    )
    entries = saved[module.MESSAGE_INJECTIONS_KEY]
    assert [entry['text'] for entry in entries] == ['failed']

    # A different source on the same anchor is untouched by the replace.
    await module.record_message_injection('chat-1', anchor='u1', text='note', source='other', replace=True)
    assert sorted(entry['text'] for entry in saved[module.MESSAGE_INJECTIONS_KEY]) == ['failed', 'note']


@pytest.mark.asyncio
async def test_record_rag_injection_replace_follows_the_final_block(monkeypatch):
    """The approved-tool-call path resets the user turn and re-splices a combined
    file+tool block; the ledger must follow it or replay diverges."""
    module = chat_injections
    saved: dict = {}

    async def _load_meta(chat_id):
        return dict(saved)

    async def _save(chat_id, key, value):
        saved[key] = value
        return True

    monkeypatch.setattr(module, 'load_chat_meta', _load_meta)
    monkeypatch.setattr(module, 'save_meta_key', _save)

    await module.record_rag_injection('chat-1', anchor='u1', text='FILE_ONLY')
    _, recorded = await module.record_rag_injection('chat-1', anchor='u1', text='FILE_AND_TOOL', replace=True)
    assert recorded is True
    assert [entry['text'] for entry in saved[chat_injections.RAG_INJECTIONS_KEY]] == ['FILE_AND_TOOL']

    # Replay of the anchor now matches the final block (one entry, no stale copy).
    replayed = apply_injections([_user('u1', 'q')], collect_injections(saved))
    assert replayed[0]['content'] == 'FILE_AND_TOOL\nq'


@pytest.mark.asyncio
async def test_record_prunes_orphaned_anchors(monkeypatch):
    """RAG/plugin entries whose anchor left the active chain are dropped in the
    same write that records the new turn (no unbounded meta growth)."""
    module = chat_injections
    saved: dict = {
        'rag_injections': [{'anchor': 'gone', 'position': 'prepend', 'text': 'old', 'ids': []}],
        'message_injections': [{'anchor': 'gone', 'position': 'prepend', 'text': 'old', 'ids': []}],
    }

    async def _load_meta(chat_id):
        return dict(saved)

    async def _save(chat_id, key, value):
        saved[key] = value
        return True

    monkeypatch.setattr(module, 'load_chat_meta', _load_meta)
    monkeypatch.setattr(module, 'save_meta_key', _save)

    await module.record_rag_injection('chat-1', anchor='u2', text='R', chain_ids={'u2'})
    assert [entry['anchor'] for entry in saved['rag_injections']] == ['u2']

    await module.record_message_injection('chat-1', anchor='u2', text='P', chain_ids={'u2'})
    assert [entry['anchor'] for entry in saved['message_injections']] == ['u2']

    # With no known chain, nothing is pruned (an empty set must not wipe the ledger).
    await module.record_rag_injection('chat-1', anchor='u3', text='R2', chain_ids=set())
    assert sorted(entry['anchor'] for entry in saved['rag_injections']) == ['u2', 'u3']


@pytest.mark.asyncio
async def test_apply_source_context_routes_to_tail_not_system(monkeypatch):
    """Producer-level check for Phase 3: retrieval must never touch a system
    message and must be recorded against the user-turn anchor."""
    from open_webui.utils import middleware as mw

    recorded: dict = {}

    async def _rag_template(template, context, query):
        return f'RAG[{context}]'

    async def _record(chat_id, *, anchor, text, **kwargs):
        recorded['chat_id'] = chat_id
        recorded['anchor'] = anchor
        recorded['text'] = text
        return [], True

    class _Config:
        @staticmethod
        async def get(key, default=None):
            return default

    monkeypatch.setattr(mw, 'rag_template', _rag_template)
    monkeypatch.setattr(mw, 'record_rag_injection', _record)
    monkeypatch.setattr(mw, 'Config', _Config)

    messages = [{'id': 'u1', 'role': 'user', 'content': 'q'}]
    out = await mw.apply_source_context_to_messages(
        None,
        messages,
        [{'source': {'id': 's1', 'name': 'S'}, 'document': ['doc'], 'metadata': [{}]}],
        'q',
        metadata={'chat_id': 'chat-1', 'user_message_id': 'u1'},
    )

    assert not any(message.get('role') == 'system' for message in out)
    assert recorded['anchor'] == 'u1'
    assert recorded['text'].startswith('RAG[')
    assert out[0]['content'].startswith('RAG[')
    assert out[0]['content'].endswith('q')


@pytest.mark.asyncio
async def test_apply_source_context_emits_retrieval_status(monkeypatch):
    """A freshly frozen retrieval block surfaces its kind and chunk count."""
    from open_webui.utils import middleware as mw

    async def _rag_template(template, context, query):
        return f'RAG[{context}]'

    async def _record(chat_id, *, anchor, text, **kwargs):
        return [], True

    class _Config:
        @staticmethod
        async def get(key, default=None):
            return default

    monkeypatch.setattr(mw, 'rag_template', _rag_template)
    monkeypatch.setattr(mw, 'record_rag_injection', _record)
    monkeypatch.setattr(mw, 'Config', _Config)

    events = []

    async def _emit(event):
        events.append(event)

    sources = [{'source': {'id': 's1', 'name': 'S'}, 'document': ['doc'], 'metadata': [{}]}]
    await mw.apply_source_context_to_messages(
        None,
        [{'id': 'u1', 'role': 'user', 'content': 'q'}],
        sources,
        'q',
        metadata={'chat_id': 'chat-1', 'user_message_id': 'u1'},
        event_emitter=_emit,
    )

    assert [event['data']['action'] for event in events] == [chat_injections.STATUS_KNOWLEDGE_RETRIEVED]
    assert events[0]['data']['count'] == 1


@pytest.mark.asyncio
async def test_apply_source_context_replay_does_not_reannounce(monkeypatch):
    """`recorded=False` means the block is replaying; no second status is emitted."""
    from open_webui.utils import middleware as mw

    async def _rag_template(template, context, query):
        return f'RAG[{context}]'

    async def _record(chat_id, *, anchor, text, **kwargs):
        return [], False

    class _Config:
        @staticmethod
        async def get(key, default=None):
            return default

    monkeypatch.setattr(mw, 'rag_template', _rag_template)
    monkeypatch.setattr(mw, 'record_rag_injection', _record)
    monkeypatch.setattr(mw, 'Config', _Config)

    events = []

    async def _emit(event):
        events.append(event)

    await mw.apply_source_context_to_messages(
        None,
        [{'id': 'u1', 'role': 'user', 'content': 'q'}],
        [{'source': {'id': 's1', 'name': 'S'}, 'document': ['doc'], 'metadata': [{}]}],
        'q',
        metadata={'chat_id': 'chat-1', 'user_message_id': 'u1'},
        event_emitter=_emit,
    )

    assert events == []


def test_multiple_append_sources_replay_in_live_order():
    """Two append-side producers on one anchor (memory edit, image-gen verdict)
    replay in the order they were recorded, preserving the live text."""
    ledger = add_injection([], anchor='u1', text='mem_update', position='append', source='memory')
    ledger = add_injection(ledger, anchor='u1', text='image_ok', position='append', source='image_generation')
    messages = apply_injections([_user('u1', 'q')], ledger)
    assert messages[0]['content'] == 'q\nmem_update\nimage_ok'


def test_baseline_applies_is_not_gated_on_rag_mode():
    """Phase 3 removed the ``RAG_SYSTEM_CONTEXT`` exclusion: retrieval is a tail
    injection now, so the top is freezable in that mode too."""
    assert system_baseline._baseline_applies({}) is True
    assert system_baseline._baseline_applies({'compare_mode': True}) is False
    assert system_baseline._baseline_applies({'internal': True}) is False


def test_base_update_and_delta_replay_in_order():
    """The live turn carries delta then edit; replay restores both, in order."""
    ledger = chat_injections.upsert_injection([], anchor='u1', text='<memory_context>d</memory_context>')
    ledger = chat_injections.add_injection(
        ledger, anchor='u1', text='<memory_context_update>\n- x\n</memory_context_update>'
    )
    replayed = apply_injections([_user('u1', 'q')], ledger)
    content = replayed[0]['content']
    assert content.index('<memory_context>') < content.index('q') < content.index('<memory_context_update>')
    assert content.endswith('</memory_context_update>')


# ---------------------------------------------------------------------------
# P0-1 — identity block + memory preamble inside the frozen top
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identity_and_memory_preamble_precede_memory_context(memory_harness):
    """(a) The identity block leads; the memory preamble frames the
    ``<memory_context>`` base that ``add_memory_context`` appends after it."""
    module, saved = memory_harness
    messages = [
        {'role': 'system', 'content': 'USER_PROMPT'},
        {'id': 'u1', 'role': 'user', 'content': 'hi'},
    ]
    messages, signature = system_baseline.assemble_frozen_top_content(messages, memory_enabled=True)
    result = await module.add_memory_context(
        None, {'messages': messages}, type('U', (), {'id': 'u'})(), {'id': 'm'}, _metadata()
    )

    content = result['messages'][0]['content']
    # The injected base block starts with the tag + newline; the preamble's inline
    # mention is followed by a space, so target the real block unambiguously.
    base_block_at = content.index('<memory_context>\n[Memory Context]')
    assert content.index(system_baseline.SYSTEM_IDENTITY_PROMPT) < content.index('USER_PROMPT')
    assert (
        content.index(system_baseline.SYSTEM_IDENTITY_PROMPT)
        < content.index(system_baseline.MEMORY_CONTEXT_PREAMBLE)
        < base_block_at
    )
    # The preamble survived `_strip_memory_context` (which runs before the base
    # is appended); its literal `<memory_context>` reference is intact.
    assert content.count('# Memory\n') == 1
    assert signature == [
        f'identity:{system_baseline.SYSTEM_IDENTITY_PROMPT}',
        f'memory_preamble:{system_baseline.MEMORY_CONTEXT_PREAMBLE}',
    ]
    # (c) The preamble is top content only; the tail ledger stays untouched.
    assert saved['injections'] == []


def test_strip_memory_context_leaves_the_preamble_reference_intact():
    """P0-1: the preamble mentions ``<memory_context>`` inline; the strip helper
    must remove only a real injected block, not the preamble that references it."""
    preamble = '# Memory\n- <memory_context> and <memory_context_update> are reference\n'
    base = '<memory_context>\n- likes tea\n</memory_context>'
    messages = [{'role': 'system', 'content': f'{preamble}{base}'}]
    memory._strip_memory_context(messages)
    assert base not in messages[0]['content']
    assert '<memory_context> and <memory_context_update>' in messages[0]['content']


def test_memory_toggle_changes_the_inputs_hash():
    """(e) Enabling memory folds the preamble into the signature, so it
    re-baselines the chat as a user action."""
    _, sig_off = system_baseline.assemble_frozen_top_content(
        [{'role': 'user', 'content': 'hi'}], memory_enabled=False
    )
    _, sig_on = system_baseline.assemble_frozen_top_content(
        [{'role': 'user', 'content': 'hi'}], memory_enabled=True
    )
    assert _inputs_hash(system_injection_signature='\n'.join(sig_off)) != _inputs_hash(
        system_injection_signature='\n'.join(sig_on)
    )
    # Identity is always in the signature; opting out of memory drops only the preamble.
    assert 'memory_preamble:' not in '\n'.join(sig_off)
    assert 'identity:' in '\n'.join(sig_off)


# ---------------------------------------------------------------------------
# P0-2 — date line appended outside the freeze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_date_line_is_appended_outside_the_frozen_baseline(monkeypatch):
    """(b) The date appears after the frozen bytes and is never persisted."""
    store = _BaselineStore()
    store.install(monkeypatch)
    inputs = _inputs_hash()

    turn1 = _system('You are helpful.\n# Memory\n- ref')
    frozen = await system_baseline.apply_system_baseline('chat-1', turn1, {}, inputs_hash=inputs)
    stored = store.meta[system_baseline.SYSTEM_BASELINE_KEY]
    assert stored['content'] == frozen
    assert 'Current date:' not in stored['content']

    now = datetime(2026, 9, 24, 10, 30, tzinfo=ZoneInfo('Asia/Shanghai'))
    messages = [{'role': 'system', 'content': frozen}, _user('u1', 'hi')]
    out = system_baseline.append_current_date_line(
        messages, user={'timezone': 'Asia/Shanghai'}, metadata={}, now=now
    )
    assert out[0]['content'] == (
        f'{frozen}\nCurrent date: 2026-09-24 (Thursday), timezone Asia/Shanghai'
    )
    # The persisted baseline must remain date-free after the append.
    assert 'Current date:' not in store.meta[system_baseline.SYSTEM_BASELINE_KEY]['content']


# ---------------------------------------------------------------------------
# P1-3: harness-injection visibility (status events)
# ---------------------------------------------------------------------------


def test_injection_status_event_carries_only_kind_and_count():
    """The UI event must stay minimal — source kind and count, never the text."""
    event = chat_injections.injection_status_event('memory_context_updated', 2)
    assert event == {
        'type': 'status',
        'data': {'action': 'memory_context_updated', 'count': 2, 'done': True},
    }
    # Junk counts clamp to zero rather than reaching the UI negative.
    assert chat_injections.injection_status_event('rag', -1)['data']['count'] == 0


@pytest.mark.asyncio
async def test_emit_injection_status_is_best_effort():
    assert await chat_injections.emit_injection_status(None, 'memory_context_updated', 1) is None

    async def _boom(event):
        raise RuntimeError('emitter down')

    # An emitter failure must never propagate into the request path.
    assert await chat_injections.emit_injection_status(_boom, 'memory_context_updated', 1) is None


@pytest.mark.asyncio
async def test_new_turn_emits_memory_status_once(memory_harness):
    """A freshly frozen delta surfaces one status; replaying the same anchor does not."""
    module, saved = memory_harness
    saved.update({'base': ['m1'], 'injections': []})
    events = []

    async def _emit(event):
        events.append(event)

    form_data = _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}])
    await module.add_memory_context(
        None,
        form_data,
        type('U', (), {'id': 'u'})(),
        {'id': 'm'},
        _metadata(user_message_id='u2'),
        event_emitter=_emit,
    )
    assert [event['data']['action'] for event in events] == [chat_injections.STATUS_MEMORY_CONTEXT_UPDATED]
    assert events[0]['data']['count'] == 1

    # Second call for the same anchor: ledger already has it, so nothing is announced.
    events.clear()
    await module.add_memory_context(
        None,
        _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}]),
        type('U', (), {'id': 'u'})(),
        {'id': 'm'},
        _metadata(user_message_id='u2'),
        event_emitter=_emit,
    )
    assert events == []


@pytest.mark.asyncio
async def test_base_edit_emits_memory_update_status(memory_harness, monkeypatch):
    """A frozen-base edit travels as a tail update and surfaces its own status."""
    module, saved = memory_harness
    saved.update(
        {
            'base': ['m1'],
            'injections': [],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'u2'},
        }
    )
    edited = _MemoryRow('m1', 'likes coffee')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(edited))})
    )
    events = []

    async def _emit(event):
        events.append(event)

    await module.add_memory_context(
        None,
        _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}]),
        type('U', (), {'id': 'u'})(),
        {'id': 'm'},
        _metadata(user_message_id='u2'),
        event_emitter=_emit,
    )
    update_events = [
        event for event in events if event['data']['action'] == chat_injections.STATUS_MEMORY_CONTEXT_UPDATE
    ]
    assert len(update_events) == 1
    assert update_events[0]['data']['count'] == 1


@pytest.mark.asyncio
async def test_base_update_dedup_does_not_reannounce(memory_harness, monkeypatch):
    """A reload that re-derives an already-frozen update block must not re-announce.

    ``add_injection`` dedupes on ``(anchor, text)``; when the persisted labels lag
    behind the ledger (hand-edited or partially migrated meta), the diff is
    recomputed but the block is already frozen — so no second status may fire.
    """
    module, saved = memory_harness
    edited = _MemoryRow('m1', 'likes coffee')
    monkeypatch.setattr(
        module, 'Memories', type('M', (), {'get_memories_by_user_id': staticmethod(lambda *a, **k: _rows_with(edited))})
    )
    block = (
        '<memory_context_update>\n'
        '- Updated (supersedes the earlier entry about likes tea): likes coffee\n'
        '</memory_context_update>'
    )
    saved.update(
        {
            'base': ['m1'],
            'injections': [
                {'source': 'memory', 'anchor': 'u2', 'position': 'append', 'text': block, 'ids': []}
            ],
            'base_labels': {'m1': 'likes tea'},
            'established': True,
            'chain_ids': {'u1', 'u2'},
        }
    )
    events = []

    async def _emit(event):
        events.append(event)

    await module.add_memory_context(
        None,
        _form_data([{'id': 'u2', 'role': 'user', 'content': 'next'}]),
        type('U', (), {'id': 'u'})(),
        {'id': 'm'},
        _metadata(user_message_id='u2'),
        event_emitter=_emit,
    )
    assert events == []
    # The block was deduped, not duplicated.
    assert [entry['text'] for entry in saved['injections']] == [block]
