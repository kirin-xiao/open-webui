from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from open_webui.models.config import Config
from open_webui.models.memories import Memories
from open_webui.utils.access_control import has_permission
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.chat_injections import (
    MEMORY_CONTEXT_KEY,
    STATUS_MEMORY_CONTEXT_UPDATE,
    STATUS_MEMORY_CONTEXT_UPDATED,
    add_injection,
    attach_injection,
    emit_injection_status,
    normalize_injections,
    prune_injections,
    upsert_injection,
)
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import (
    add_or_update_system_message,
    get_content_from_message,
)

log = logging.getLogger(__name__)

MEMORY_CONTEXT_OPEN = '<memory_context>'
MEMORY_CONTEXT_CLOSE = '</memory_context>'
MEMORY_CONTEXT_UPDATE_OPEN = '<memory_context_update>'
MEMORY_CONTEXT_UPDATE_CLOSE = '</memory_context_update>'


def clean_memory_content(content: str | None) -> str:
    value = (content or '').strip()
    if not value:
        raise HTTPException(status_code=400, detail='Memory content cannot be empty')
    return value


def clean_memory_path(path: str | None) -> str | None:
    value = re.sub(r'/+', '/', (path or '').strip().strip('/'))
    if not value:
        return None
    parts = value.split('/')
    if any(part in {'', '.', '..'} for part in parts) or any(ord(char) < 32 for char in value):
        raise HTTPException(status_code=400, detail='Invalid memory path')
    return value


def memory_vector_text(content: str, path: str | None = None) -> str:
    path = clean_memory_path(path)
    return f'{path}\n{content}' if path else content


def memory_date(timestamp: int | None = None) -> str:
    """Machine-readable write date (``YYYY-MM-DD``), server-local time."""
    if timestamp is None:
        timestamp = int(time.time())
    return time.strftime('%Y-%m-%d', time.localtime(timestamp))


def memory_row_meta(memory) -> dict:
    """The memory row's ``meta`` dict, tolerating legacy rows without one."""
    meta = getattr(memory, 'meta', None)
    return meta if isinstance(meta, dict) else {}


def memory_row_date(memory) -> str | None:
    value = memory_row_meta(memory).get('date')
    return value if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) else None


def sanitize_memory_attribution(value) -> str | None:
    """Coerce a declared attribution to ``'user'``/``'assistant'``/``None``.

    Provenance is model-declared on the interactive tool schemas. A malformed
    value (e.g. ``'model'``) is dropped here rather than stored verbatim or
    allowed to abort a batch as a validation error.
    """
    return value if value in {'user', 'assistant'} else None


def memory_row_attribution(memory) -> str | None:
    return sanitize_memory_attribution(memory_row_meta(memory).get('attribution'))


def memory_write_meta(
    *,
    source: str,
    attribution: str | None = None,
    chat_id: str | None = None,
    message_id: str | None = None,
    model: str | None = None,
    timestamp: int | None = None,
) -> dict:
    """The ``meta`` stamped on a memory at write time.

    ``date`` gives every row a machine-readable recency key so consolidation can
    order conflicting entries; ``attribution`` records whether the value came
    from the user (``"user"``) or is an assistant inference
    (``"assistant"``, e.g. a self-corrected 已修正 conclusion).

    Model-declared attribution wins when present. Otherwise provenance follows
    the call path, not the transport: a human UI write is ``"user"``; an
    assistant-initiated tool or background-review write is ``"assistant"``
    (never claim user provenance without evidence).
    """
    attribution = sanitize_memory_attribution(attribution)
    if attribution is None:
        attribution = 'user' if source == 'manual' else 'assistant'
    return {
        'created_by': source,
        'attribution': attribution,
        'date': memory_date(timestamp),
        'chat_id': chat_id,
        'message_id': message_id,
        'model': model,
    }


def memory_label(memory) -> str:
    """Render a memory row as ``[date][attribution] path: content``.

    Legacy rows written before the timestamp/attribution change carry no
    ``meta`` (or a ``meta`` without ``date``/``attribution``) and render exactly
    as before — ``path: content`` — so old stores stay byte-stable until a row
    is rewritten.
    """
    label = f'{memory.path}: {memory.content}' if memory.path else memory.content
    date = memory_row_date(memory)
    attribution = memory_row_attribution(memory)
    prefix = f'[{date}]' if date else ''
    if attribution:
        prefix += f'[{attribution}]'
    return f'{prefix} {label}' if prefix else label


def _path_parts(path: str | None) -> list[str]:
    return [part for part in (path or '').split('/') if part]


def _parent_path(path: str | None) -> str | None:
    parts = _path_parts(path)
    return '/'.join(parts[:-1]) if len(parts) > 1 else None


def _path_rank(memory_path: str | None, lookup_path: str | None) -> tuple | None:
    if not lookup_path:
        return None

    memory_path = clean_memory_path(memory_path)
    lookup_path = clean_memory_path(lookup_path)
    if not memory_path or not lookup_path:
        return None

    if memory_path == lookup_path:
        return (0, 0)
    if memory_path.startswith(f'{lookup_path}/'):
        return (1, len(_path_parts(memory_path)) - len(_path_parts(lookup_path)))
    if lookup_path.startswith(f'{memory_path}/'):
        return (2, len(_path_parts(lookup_path)) - len(_path_parts(memory_path)))
    if _parent_path(memory_path) and _parent_path(memory_path) == _parent_path(lookup_path):
        return (3, 0)

    memory_parts = set(_path_parts(memory_path))
    lookup_parts = set(_path_parts(lookup_path))
    shared = len(memory_parts & lookup_parts)
    if shared:
        return (4, -shared)
    if _path_parts(memory_path)[-1:] == _path_parts(lookup_path)[-1:]:
        return (5, 0)

    return None


def _memory_matches_query(memory, query: str) -> bool:
    value = query.strip().lower()
    if not value:
        return True
    return value in (memory.content or '').lower() or value in (memory.path or '').lower()


def search_memory_rows(
    memories: list,
    *,
    query: str | None = None,
    path: str | None = None,
    memory_id: str | None = None,
    memory_type: str = 'all',
    limit: int = 20,
) -> list:
    rows = list(memories or [])
    if memory_id:
        rows = [memory for memory in rows if memory.id == memory_id]
    if memory_type != 'all':
        rows = [memory for memory in rows if memory.type == memory_type]

    query = (query or '').strip()
    lookup_path = clean_memory_path(path)
    if lookup_path:
        basename = _path_parts(lookup_path)[-1] if _path_parts(lookup_path) else lookup_path

        def related(memory) -> bool:
            rank = _path_rank(memory.path, lookup_path)
            if rank is not None:
                return True
            haystack = f'{memory.path or ""}\n{memory.content or ""}'.lower()
            return lookup_path.lower() in haystack or basename.lower() in haystack

        rows = [memory for memory in rows if related(memory)]

    if query:
        rows = [memory for memory in rows if _memory_matches_query(memory, query)]

    def sort_key(memory):
        rank = _path_rank(memory.path, lookup_path) if lookup_path else None
        return rank if rank is not None else (9, 0), -(memory.updated_at or 0), memory.id or ''

    return sorted(rows, key=sort_key)[: max(1, min(limit or 20, 100))]


def list_memory_path_groups(
    memories: list,
    *,
    query: str = '',
    memory_type: str = 'all',
    limit: int = 100,
) -> dict:
    rows = [
        memory
        for memory in (memories or [])
        if (memory_type == 'all' or memory.type == memory_type) and _memory_matches_query(memory, query)
    ]
    grouped: dict[tuple[str | None, str], dict] = {}
    for memory in rows:
        key = (memory.path, memory.type)
        group = grouped.setdefault(
            key,
            {
                'path': memory.path,
                'type': memory.type,
                'count': 0,
                'updated_at': 0,
                'children': [],
            },
        )
        group['count'] += 1
        group['updated_at'] = max(group['updated_at'], memory.updated_at or 0)

    paths = [path for path, _ in grouped if path]
    for group in grouped.values():
        path = group['path']
        if not path:
            continue
        prefix = f'{path}/'
        children = []
        for candidate in paths:
            if not candidate.startswith(prefix):
                continue
            remainder = candidate[len(prefix) :]
            child = f'{prefix}{remainder.split("/", 1)[0]}'
            if child not in children:
                children.append(child)
        group['children'] = children[:20]

    groups = sorted(grouped.values(), key=lambda item: item['updated_at'], reverse=True)
    return {'paths': groups[: max(1, min(limit or 100, 500))], 'count': len(groups)}


def read_memory_path_rows(
    memories: list,
    *,
    path: str,
    memory_type: str = 'all',
    include_children: bool = True,
    limit: int = 50,
) -> dict:
    lookup_path = clean_memory_path(path)
    if not lookup_path:
        raise HTTPException(status_code=400, detail='Memory path is required')

    rows = [memory for memory in (memories or []) if memory_type == 'all' or memory.type == memory_type]
    path_set = {memory.path for memory in rows if memory.path}
    parents = [
        '/'.join(_path_parts(lookup_path)[:idx])
        for idx in range(1, len(_path_parts(lookup_path)))
        if '/'.join(_path_parts(lookup_path)[:idx]) in path_set
    ]
    children = sorted(
        {
            f'{lookup_path}/{memory.path[len(lookup_path) + 1 :].split("/", 1)[0]}'
            for memory in rows
            if memory.path and memory.path.startswith(f'{lookup_path}/')
        }
    )

    def selected(memory) -> bool:
        if memory.path == lookup_path:
            return True
        if memory.path in parents:
            return True
        return bool(include_children and memory.path and memory.path.startswith(f'{lookup_path}/'))

    selected_rows = [memory for memory in rows if selected(memory)]

    def sort_key(memory):
        if memory.path == lookup_path:
            return (0, 0, -(memory.updated_at or 0), memory.id or '')
        if memory.path and memory.path.startswith(f'{lookup_path}/'):
            return (1, len(_path_parts(memory.path)), -(memory.updated_at or 0), memory.id or '')
        return (2, -len(_path_parts(memory.path)), -(memory.updated_at or 0), memory.id or '')

    return {
        'path': lookup_path,
        'parents': parents,
        'children': children[:50],
        'memories': sorted(selected_rows, key=sort_key)[: max(1, min(limit or 50, 100))],
    }


# Scripts whose words are not delimited by spaces. A two-character segment is a
# complete word in these scripts (工作, 生活, 健康), so the >= 3-character
# heuristic that suits English path segments makes path hints inert here.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def contains_cjk(value: str) -> bool:
    return any(any(start <= ord(char) <= end for start, end in _CJK_RANGES) for char in value)


def _hint_segment_matches(part: str, lowered_query: str) -> bool:
    """Whether a path segment is specific enough to hint from the query text."""
    part = part.lower()
    if not part or part not in lowered_query:
        return False
    return len(part) >= 2 if contains_cjk(part) else len(part) >= 3


def memory_path_hints(query: str, memories: list, limit: int = 6) -> list[str]:
    lowered = (query or '').lower()
    if not lowered:
        return []

    hints: list[str] = []
    for memory in sorted(memories or [], key=lambda item: (item.path or '', item.content or '', item.id or '')):
        path = memory.path
        if not path or path in hints:
            continue
        parts = _path_parts(path)
        last = parts[-1] if parts else path
        if path.lower() in lowered or last.lower() in lowered:
            hints.append(path)
        elif any(_hint_segment_matches(part, lowered) for part in parts):
            hints.append(path)
        if len(hints) >= limit:
            break
    return hints


def validate_memory_operations(form_data) -> list[dict]:
    if not form_data.operations:
        raise HTTPException(status_code=400, detail='No memory operations provided')

    operations = []
    for operation in form_data.operations:
        op = operation.model_dump()
        action = op.get('action')

        if action == 'add':
            op['content'] = clean_memory_content(op.get('content'))
            op['type'] = Memories.normalize_memory_type(op.get('type'))
            op['path'] = clean_memory_path(op.get('path'))
        elif action == 'replace':
            if not op.get('id'):
                raise HTTPException(status_code=400, detail='Memory id is required for replace')
            op['content'] = clean_memory_content(op.get('content'))
            if op.get('type') is not None:
                op['type'] = Memories.normalize_memory_type(op.get('type'))
            op['path'] = clean_memory_path(op.get('path'))
        elif action == 'move':
            if not op.get('id'):
                raise HTTPException(status_code=400, detail='Memory id is required for move')
            op['path'] = clean_memory_path(op.get('path'))
        elif action == 'remove':
            if not op.get('id'):
                raise HTTPException(status_code=400, detail='Memory id is required for remove')
        else:
            raise HTTPException(status_code=400, detail=f'Unsupported memory operation: {action}')

        operations.append(op)

    return operations


def model_allows_memory(model: dict | None) -> bool:
    return ((model or {}).get('info', {}).get('meta', {}).get('capabilities') or {}).get('memory', True)


MEMORY_DEFAULT_USER_CHAR_LIMIT = 2000
MEMORY_DEFAULT_CONTEXT_CHAR_LIMIT = 2000
# Path-hint rows are a lexical-recall channel: a literal path-segment match is
# strong evidence, so they enter at a floor score instead of being dropped by
# the vector relevance threshold.
MEMORY_CONTEXT_FLOOR_SCORE = -1.0

SECTION_TITLES = {'user': 'User Memory', 'context': 'Memory Context'}


def _char_limit(value, default: int) -> int:
    try:
        return max(250, int(value or default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    section: str
    label: str
    score: float
    created_at: int = 0

    @property
    def priority(self) -> tuple:
        # Relevance first, then newest, then id — never casefold order.
        return (-self.score, -self.created_at, self.id)


def _hit_id(results, index: int) -> str | None:
    ids = getattr(results, 'ids', None)
    return ids[0][index] if ids and ids[0] and len(ids[0]) > index else None


def _hit_score(results, index: int) -> float:
    distances = getattr(results, 'distances', None)
    if distances and distances[0] and len(distances[0]) > index:
        try:
            return float(distances[0][index])
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def collect_memory_entries(all_memories, results, hints, relevance_threshold: float) -> list[MemoryEntry]:
    """Build the deduped set of memories to inject.

    ``type='user'`` rows are durable facts about the user, so they are injected
    unconditionally and documented as intentionally global. ``type='context'``
    rows must clear ``memories.relevance_threshold``; rows matched by a path hint
    join at a floor score so the lexical channel is not gated by vector distance.
    """
    entries: list[MemoryEntry] = []
    seen: set[str] = set()
    rows_by_id = {memory.id: memory for memory in (all_memories or [])}

    for memory in sorted(
        (memory for memory in (all_memories or []) if memory.type == 'user'),
        key=lambda memory: (-(memory.created_at or 0), memory.id or ''),
    ):
        seen.add(memory.id)
        entries.append(
            MemoryEntry(
                id=memory.id,
                section='user',
                label=memory_label(memory),
                score=1.0,
                created_at=memory.created_at or 0,
            )
        )

    documents = getattr(results, 'documents', None) if results else None
    for index, doc in enumerate(documents[0] if documents and documents[0] else []):
        if not doc:
            continue
        memory_id = _hit_id(results, index)
        if not memory_id or memory_id in seen:
            continue
        memory = rows_by_id.get(memory_id)
        if memory is None or memory.type != 'context':
            continue
        score = _hit_score(results, index)
        if relevance_threshold > 0.0 and score < relevance_threshold:
            continue
        seen.add(memory_id)
        entries.append(
            MemoryEntry(
                id=memory_id,
                section='context',
                label=memory_label(memory),
                score=score,
                created_at=memory.created_at or 0,
            )
        )

    for hint in hints:
        for memory in search_memory_rows(all_memories, path=hint, memory_type='context', limit=4):
            if memory.id in seen:
                continue
            seen.add(memory.id)
            entries.append(
                MemoryEntry(
                    id=memory.id,
                    section='context',
                    label=memory_label(memory),
                    score=MEMORY_CONTEXT_FLOOR_SCORE,
                    created_at=memory.created_at or 0,
                )
            )

    return entries


def _fit_section(entries: list[MemoryEntry], limit: int) -> tuple[list[str], list[str], int]:
    """Fit whole entries into a section budget; never cut an entry mid-text.

    Returns the rendered lines, the ids actually emitted, and how many entries
    were dropped. The emitted ids are authoritative — the caller must not infer
    them by matching label text back into the output.
    """
    lines: list[str] = []
    emitted: list[str] = []
    used = 0
    for entry in entries:
        line = f'- {entry.label}'
        if used + len(line) + 1 > limit:
            if not lines:
                # A single entry larger than the whole budget: truncate it rather
                # than emit a section header with nothing under it.
                lines.append(line[:limit])
                emitted.append(entry.id)
            return lines, emitted, len(entries) - len(emitted)
        lines.append(line)
        emitted.append(entry.id)
        used += len(line) + 1
    return lines, emitted, 0


def render_memory_context(entries: list[MemoryEntry], limits: dict[str, int]) -> tuple[str, dict[str, int], list[str]]:
    """Render entries (already in final display order) into titled sections."""
    parts: list[str] = []
    dropped = {'user': 0, 'context': 0}
    emitted: list[str] = []
    for section in ('user', 'context'):
        section_entries = [entry for entry in entries if entry.section == section]
        lines, section_emitted, section_dropped = _fit_section(section_entries, limits[section])
        dropped[section] = section_dropped
        emitted.extend(section_emitted)
        if lines:
            parts.append(f'[{SECTION_TITLES[section]}]\n' + '\n'.join(lines))
    return '\n\n'.join(parts).strip(), dropped, emitted


def _memory_query(messages: list) -> str:
    user_messages = []
    for message in reversed(messages or []):
        if message.get('role') != 'user':
            continue
        content = get_content_from_message(message)
        if isinstance(content, str) and content.strip():
            user_messages.append(content.strip())
        if len(user_messages) >= 7:
            break
    return '\n\n'.join(reversed(user_messages))[-4000:]


def _strip_memory_context(messages: list) -> None:
    if not messages or messages[0].get('role') != 'system':
        return
    content = messages[0].get('content', '')
    # Match the actual injected block (``<memory_context>\n…``), never the bare
    # tag. The P0-1 memory preamble mentions ``<memory_context>`` inline
    # (followed by a space, not a newline); matching the bare open tag here would
    # mistake that reference for a block and strip the preamble along with it.
    marker = f'{MEMORY_CONTEXT_OPEN}\n'
    if not isinstance(content, str) or marker not in content:
        return
    start = content.find(marker)
    end = content.find(MEMORY_CONTEXT_CLOSE, start)
    if end != -1:
        messages[0]['content'] = (content[:start] + content[end + len(MEMORY_CONTEXT_CLOSE) :]).strip()


def _log_dropped(dropped: dict[str, int], limits: dict[str, int]) -> None:
    for section, count in dropped.items():
        if count:
            log.warning(
                'Memory context: dropped %d %s entr%s over the %d char cap',
                count,
                section,
                'y' if count == 1 else 'ies',
                limits[section],
            )


async def _load_memory_state(chat_id: str | None) -> dict:
    """Frozen injection state: ``base`` ids live in the system prompt (injected
    once), ``injections`` are the anchor-keyed tail blocks already emitted.

    ``base_labels`` freezes the *rendered* label of each base row so a later edit
    can be detected and delivered as a tail update instead of rewriting the top;
    ``established`` records that the base was captured, so an emptied base is not
    mistaken for a first injection and refilled."""
    if not is_saved_chat_id(chat_id):
        return {'base': [], 'injections': [], 'base_labels': {}, 'established': False, 'chain_ids': set()}
    from open_webui.models.chat_messages import ChatMessages
    from open_webui.models.chats import Chats

    chat = await Chats.get_chat_by_id(chat_id)
    if not chat:
        return {'base': [], 'injections': [], 'base_labels': {}, 'established': False, 'chain_ids': set()}

    state = (chat.meta or {}).get(MEMORY_CONTEXT_KEY) or {}
    base = [str(memory_id) for memory_id in state.get('base') or []]
    base_labels = {
        str(memory_id): str(label) for memory_id, label in (state.get('base_labels') or {}).items() if label is not None
    }
    # A legacy ``delta`` list (pre-#30239) is treated as already-converged: its
    # rendered text was never persisted, so there is nothing to replay. A chat
    # left without a base re-selects on the next turn.
    injections = normalize_injections(state.get('injections'), source='memory')

    # Prune anchors only when we are certain they left the store. Derive the
    # known ids from the same source replay reads (``chat_message`` rows), and
    # never prune on a partial/unknown view: an over-eager prune would delete a
    # valid entry from the very request that had already replayed it.
    chain_ids: set[str] = set()
    try:
        messages_map = await ChatMessages.get_messages_map_by_chat_id(chat_id) or {}
        for key, message in messages_map.items():
            chain_ids.add(str(key))
            if isinstance(message, dict) and message.get('id'):
                chain_ids.add(str(message['id']))
    except Exception:
        log.debug('Memory ledger: could not read the message map for pruning', exc_info=True)
        chain_ids = set()

    # A base counts as established once anything was injected (ids, labels, or
    # tail injections) or once the state carries the marker explicitly. The
    # injections term matters for a Phase-1 chat whose base was emptied by
    # deletions: re-treating it as a first establishment would re-bake ids that
    # already replay at the tail, duplicating context.
    established = bool(state.get('established')) or bool(base) or bool(base_labels) or bool(injections)

    return {
        'base': base,
        'injections': injections,
        'base_labels': base_labels,
        'established': established,
        'chain_ids': chain_ids,
    }


async def _save_memory_state(chat_id: str | None, state: dict) -> None:
    if not is_saved_chat_id(chat_id):
        return
    from open_webui.models.chats import Chats

    try:
        await Chats.update_chat_meta_by_id(chat_id, {MEMORY_CONTEXT_KEY: state})
    except Exception as e:
        log.debug('Failed to persist memory context state: %s', e)


def _entries_for_ids(ids: list[str], rows_by_id: dict) -> list[MemoryEntry]:
    """Render frozen ids from their stored rows, so the base block is byte-stable
    across turns regardless of what this turn's retrieval returned.

    Scores are derived from the frozen position, not from the live vector result:
    a frozen block must not reorder when a later query scores a row differently.
    """
    total = max(1, len(ids))
    entries = []
    for index, memory_id in enumerate(ids):
        memory = rows_by_id.get(memory_id)
        if memory is None:
            continue
        entries.append(
            MemoryEntry(
                id=memory.id,
                section='user' if memory.type == 'user' else 'context',
                label=memory_label(memory),
                score=1.0 - index / (total + 1),
                created_at=memory.created_at or 0,
            )
        )
    return entries


def _select_memory_ids(
    recovered: list[MemoryEntry], state: dict, rows_by_id: dict, chat_id
) -> tuple[list[str], list[str]]:
    """Split the injected set into a frozen system-prompt base and newly-relevant tail ids."""
    # Drop ids whose memory has since been deleted, so frozen state converges.
    base_ids = [memory_id for memory_id in state['base'] if memory_id in rows_by_id]
    injected_ids = {memory_id for injection in state.get('injections', []) for memory_id in injection.get('ids', [])}
    ranked = [entry.id for entry in sorted(recovered, key=lambda entry: entry.priority)]

    if not is_saved_chat_id(chat_id):
        # No persistence surface: inject the current selection in the system
        # prompt, as before. Nothing can be appended across turns.
        return ranked, []
    # ``established`` records that the base was captured. States predating the
    # marker (or built by tests) fall back to "base or injections already exist".
    established = state.get('established')
    if established is None:
        established = bool(base_ids) or bool(injected_ids) or bool(state.get('base_labels'))
    if not established:
        # First injection: everything relevant goes into the system prompt once.
        return ranked, []
    # Later turns only append memories that have never been injected anywhere.
    # Already-emitted ids are replayed from the ledger, never re-rendered.
    known_ids = set(base_ids) | injected_ids
    return base_ids, [memory_id for memory_id in ranked if memory_id not in known_ids]


# The narration that turns a bare content swap into an explicit supersession
# (plan P1-2). Deterministic bytes only: the subject and content come from the
# persisted label / live row, never a clock, so the frozen ledger replays
# byte-identically.
_BASE_UPDATE_SUBJECT_MAX = 60
_LABEL_META_PREFIX = re.compile(r'^(?:\[[^\]]*\]\s*)+')


def memory_supersession_subject(old_label: str | None) -> str:
    """The short subject naming the earlier entry a base update supersedes.

    Derived from the *frozen* label — the exact text the model saw in the top —
    rather than the live row, so an edit that also moves or renames the path still
    names the entry being replaced. The ``[date][attribution]`` prefixes and the
    content are dropped, leaving the path (the slot the entry occupies); a
    pathless entry falls back to its content head. Kept short and non-leaky, and
    purely a function of the frozen label, so it is byte-stable across replays.
    """
    value = ' '.join((old_label or '').split())
    value = _LABEL_META_PREFIX.sub('', value)
    subject = value.split(': ', 1)[0].strip() or value
    if len(subject) > _BASE_UPDATE_SUBJECT_MAX:
        subject = subject[: _BASE_UPDATE_SUBJECT_MAX - 1].rstrip() + '…'
    return subject


def _collect_base_updates(state: dict, rows_by_id: dict, base_ids: list[str]) -> tuple[list[str], dict[str, str]]:
    """Diff the frozen base labels against the live rows.

    Returns the rendered update lines and the label map to persist. An edit is
    narrated as ``Updated (supersedes the earlier entry about <subject>): <new>``
    and a deletion as ``<old label> no longer applies. Disregard it.`` — the
    earlier entry's text sits in the immutable frozen top, so the tail must say
    which way the new value supersedes it rather than swapping content in
    silence. An id that predates label-freezing adopts its current label
    silently: its text already sits in the frozen top, so echoing it would be a
    spurious update.

    Deletion is a *correction*, not a convergence: the frozen system top is
    immutable by design (#30239), so a deleted row's original text remains in the
    prefix and a ``no longer applies`` notice is delivered at the tail so the
    model learns the fact no longer holds. Rewriting the top to drop it would
    invalidate the cache this whole mechanism exists to preserve.

    ``base_ids`` is the resolved base for this turn, so a freshly established base
    records its labels immediately — otherwise the very first edit would go
    unnoticed.
    """
    updates: list[str] = []
    labels: dict[str, str] = {}
    stored = state.get('base_labels') or {}
    ordered_ids = list(dict.fromkeys([*base_ids, *state.get('base', []), *stored.keys()]))

    for memory_id in ordered_ids:
        old_label = stored.get(memory_id)
        memory = rows_by_id.get(memory_id)
        in_base = memory_id in base_ids
        if memory is None or not in_base:
            if old_label:
                updates.append(f'- {old_label} no longer applies. Disregard it.')
            continue
        current = memory_label(memory)
        labels[memory_id] = current
        if old_label is None:
            continue
        if current != old_label:
            subject = memory_supersession_subject(old_label)
            updates.append(f'- Updated (supersedes the earlier entry about {subject}): {current}')

    return updates, labels


async def _emit_tail_injection(
    *,
    form_data: dict,
    messages: list,
    anchor: str | None,
    block: str | None,
    rendered_ids: list[str],
    injections: list[dict],
    event_emitter=None,
) -> list[dict]:
    """Splice the freshly rendered tail block onto its anchor and record it.

    The block is frozen against ``anchor`` (not merely the trailing message: a
    guided-regeneration prompt must not receive a block the ledger records
    against ``u_N``). Idempotent — an already-recorded anchor is not re-spliced.
    When a *new* block is recorded, publish the injection-visibility status.
    """
    if not block or not anchor:
        return injections

    form_data['messages'] = attach_injection(messages, anchor, block, position='prepend')
    if any(injection['anchor'] == str(anchor) for injection in injections):
        return injections
    injections = upsert_injection(
        injections, anchor=anchor, text=block, position='prepend', ids=rendered_ids, source='memory'
    )
    await emit_injection_status(event_emitter, STATUS_MEMORY_CONTEXT_UPDATED, len(rendered_ids))
    return injections


async def _emit_base_updates(
    *,
    form_data: dict,
    state: dict,
    rows_by_id: dict,
    base_ids: list[str],
    anchor: str | None,
    injections: list[dict],
    event_emitter=None,
) -> tuple[list[dict], dict[str, str]]:
    """Deliver edits/deletions to frozen base rows as a tail update.

    Returns the updated ledger and the label map to persist. A continuation (no
    trailing user turn) leaves the diff pending: the labels are not advanced, so
    the next user turn still emits it exactly once.
    """
    updates, base_labels = _collect_base_updates(state, rows_by_id, base_ids)
    if not updates:
        return injections, base_labels
    if not anchor:
        return injections, state.get('base_labels') or {}

    update_block = f'{MEMORY_CONTEXT_UPDATE_OPEN}\n' + '\n'.join(updates) + f'\n{MEMORY_CONTEXT_UPDATE_CLOSE}'
    previous = injections
    injections = add_injection(injections, anchor=anchor, text=update_block, position='append', source='memory')
    # Splice the same frozen block onto the live turn; replay restores it later.
    form_data['messages'] = attach_injection(form_data['messages'], anchor, update_block, position='append')
    # ``add_injection`` dedupes on ``(anchor, text)``: a replay or a stale-label
    # reload that re-derives the same block must not re-announce it. Mirrors the
    # explicit anchor guard in ``_emit_tail_injection``.
    if injections != previous:
        await emit_injection_status(event_emitter, STATUS_MEMORY_CONTEXT_UPDATE, len(updates))
    return injections, base_labels


async def add_memory_context(
    request,
    form_data: dict,
    user,
    model: dict | None = None,
    metadata: dict | None = None,
    event_emitter=None,
):
    if not model_allows_memory(model):
        return form_data

    query = _memory_query(form_data.get('messages', []))
    if not query:
        return form_data

    all_memories = await Memories.get_memories_by_user_id(user.id)
    rows_by_id = {memory.id: memory for memory in (all_memories or [])}
    if not rows_by_id:
        return form_data

    results = None
    try:
        from open_webui.routers.memories import QueryMemoryForm, query_memory

        results = await query_memory(request, QueryMemoryForm(content=query, k=8), user)
    except Exception as e:
        log.debug(e)

    config = await Config.get_many(
        'memories.user_char_limit',
        'memories.context_char_limit',
        'memories.relevance_threshold',
    )
    try:
        threshold = float(config.get('memories.relevance_threshold') or 0.0)
    except (TypeError, ValueError):
        threshold = 0.0

    limits = {
        'user': _char_limit(config.get('memories.user_char_limit'), MEMORY_DEFAULT_USER_CHAR_LIMIT),
        'context': _char_limit(config.get('memories.context_char_limit'), MEMORY_DEFAULT_CONTEXT_CHAR_LIMIT),
    }

    recovered = collect_memory_entries(all_memories, results, memory_path_hints(query, all_memories), threshold)

    chat_id = (metadata or {}).get('chat_id')
    state = await _load_memory_state(chat_id)
    base_ids, new_ids = _select_memory_ids(recovered, state, rows_by_id, chat_id)

    messages = form_data.get('messages', [])
    _strip_memory_context(messages)

    base_entries = _entries_for_ids(base_ids, rows_by_id)
    base_text, base_dropped, _ = render_memory_context(base_entries, limits)
    _log_dropped(base_dropped, limits)
    if base_text:
        form_data['messages'] = add_or_update_system_message(
            f'{MEMORY_CONTEXT_OPEN}\n{base_text}\n{MEMORY_CONTEXT_CLOSE}',
            messages,
            append=True,
        )
        messages = form_data['messages']

    # Memories discovered after the first injection are appended to the bottom of
    # the conversation, so the cached system-prompt prefix is never rewritten.
    # The rendered block is frozen in the ledger against the user turn it
    # augmented, so later turns replay it byte-identically instead of
    # re-rendering it against a mutated memory store (#30239).
    # Prune orphans left by a deleted message or a switched branch *before*
    # recording this turn: the current anchor is not in stored history yet, so
    # pruning after the upsert would drop the block we just emitted.
    injections = prune_injections(state.get('injections', []), state.get('chain_ids') or set())

    anchor = None
    if is_saved_chat_id(chat_id) and messages and messages[-1].get('role') == 'user':
        anchor = messages[-1].get('id') or (metadata or {}).get('user_message_id')

    delta_text, delta_dropped, rendered_ids = render_memory_context(_entries_for_ids(new_ids, rows_by_id), limits)
    _log_dropped(delta_dropped, limits)

    block = f'{MEMORY_CONTEXT_OPEN}\n{delta_text}\n{MEMORY_CONTEXT_CLOSE}' if delta_text else None
    injections = await _emit_tail_injection(
        form_data=form_data,
        messages=messages,
        anchor=anchor,
        block=block,
        rendered_ids=rendered_ids,
        injections=injections,
        event_emitter=event_emitter,
    )

    # Edits and deletions to rows already frozen into the system-prompt base are
    # delivered as a tail update on the current user turn, never by rewriting the
    # frozen top.
    injections, base_labels = await _emit_base_updates(
        form_data=form_data,
        state=state,
        rows_by_id=rows_by_id,
        base_ids=base_ids,
        anchor=anchor,
        injections=injections,
        event_emitter=event_emitter,
    )

    # Compare mode fans out one request per branch against a shared meta key;
    # only the primary branch writes it so branches do not clobber each other.
    owns_ledger = not (metadata or {}).get('compare_mode') or (metadata or {}).get('is_primary_branch')
    new_state = {
        'base': base_ids,
        'injections': injections,
        'base_labels': base_labels,
        'established': True,
    }
    if is_saved_chat_id(chat_id) and owns_ledger and new_state != _readable_state(state):
        await _save_memory_state(chat_id, new_state)

    return form_data


def _readable_state(state: dict) -> dict:
    """The persisted subset of a loaded state, for a change comparison."""
    return {
        'base': state.get('base', []),
        'injections': state.get('injections', []),
        'base_labels': state.get('base_labels') or {},
        'established': bool(state.get('established')),
    }


async def review_memory_after_turn(
    *,
    request,
    user,
    model: dict | None,
    metadata: dict,
    form_data: dict,
    assistant_message: dict,
    messages: list[dict],
) -> None:
    if not model_allows_memory(model):
        return

    features = metadata.get('features') or {}
    if not features.get('memory'):
        return

    # Compare mode runs the review per branch, and the review writes memories;
    # keep it on the primary branch only so branches do not race (#30238).
    if metadata.get('compare_mode') and not metadata.get('is_primary_branch'):
        return

    assistant_content = get_content_from_message(assistant_message)
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        return

    config = await Config.get_many(
        'memories.enable',
        'memories.background_review.enable',
        'memories.review_interval_turns',
        'user.permissions',
    )
    if not config.get('memories.enable') or not config.get('memories.background_review.enable'):
        return

    try:
        interval = max(1, int(config.get('memories.review_interval_turns', 10)))
    except Exception:
        interval = 10

    user_turns = len([message for message in messages if message.get('role') == 'user'])
    if user_turns == 0 or user_turns % interval != 0:
        return

    # features is client-supplied; re-check the permission the memory routes enforce.
    if user.role != 'admin' and not await has_permission(user.id, 'features.memories', config.get('user.permissions')):
        return

    task = asyncio.create_task(
        _review_memory(
            request=request,
            user=user,
            model=model,
            metadata=metadata,
            form_data=form_data,
            assistant_message=assistant_message,
            messages=messages,
        )
    )

    def log_failure(done_task):
        try:
            done_task.result()
        except Exception as e:
            log.debug('Memory review failed: %s', e)

    task.add_done_callback(log_failure)


async def _review_memory(
    *,
    request,
    user,
    model: dict | None,
    metadata: dict,
    form_data: dict,
    assistant_message: dict,
    messages: list[dict],
) -> None:
    existing_memories = await Memories.get_memories_by_user_id(user.id)
    existing_lines = [
        f'- id={memory.id} type={memory.type} path={memory.path or ""} '
        f'date={memory_row_date(memory) or "unknown"} attribution={memory_row_attribution(memory) or "unknown"} '
        f'content={memory.content}'
        for memory in (existing_memories or [])[:80]
    ]

    assistant_content = get_content_from_message(assistant_message)
    if not isinstance(assistant_content, str):
        assistant_content = ''

    transcript_lines = []
    for message in messages[-16:]:
        role = message.get('role', '')
        content = message.get('content', '')
        if not isinstance(content, str):
            content = get_content_from_message(message)
        content = content.strip()
        if role not in {'user', 'assistant'} or not content:
            continue
        if len(content) > 1600:
            content = f'{content[:1000]}\n...(truncated)...\n{content[-400:]}'
        transcript_lines.append(f'{role}: {content}')

    if assistant_content.strip():
        assistant_final = assistant_content.strip()
        if len(assistant_final) > 1600:
            assistant_final = f'{assistant_final[:1000]}\n...(truncated)...\n{assistant_final[-400:]}'
        transcript_lines.append(f'assistant_final: {assistant_final}')

    model_id = model.get('id') if isinstance(model, dict) else form_data.get('model')
    operations = await _generate_memory_operations(
        request=request,
        user=user,
        model_id=model_id,
        metadata=metadata,
        existing_text='\n'.join(existing_lines) if existing_lines else '(none)',
        transcript='\n\n'.join(transcript_lines),
    )
    if operations:
        from open_webui.routers.memories import UpdateMemoriesForm, update_memories

        await update_memories(request, UpdateMemoriesForm(operations=operations, source='background_review'), user)


MEMORY_REVIEW_CONSOLIDATION_DUTY = """Consolidation duty (every review):
- Keep exactly one active value per slot. When several memories describe the
  same fact, preference, project, or decision, merge them into a single entry
  holding the current value.
- Recency wins. When entries conflict, the most recently stated or confirmed
  value is authoritative; fold the correction into the surviving entry and
  delete the superseded text.
- Never leave two entries that contradict each other without an explicit
  recency ordering. Superseded text is deleted, not stored beside the current
  value.
- Keep durable decision rules and standing preferences; drop narration,
  chronology, and one-off events that merely led to the rule.
- Prefer one merged entry over several overlapping entries. Use replace to fold
  a correction into the surviving entry, and remove for entries fully absorbed
  by another.
- Attribution: set "user" when the user stated the fact or preference, and
  "assistant" when it is your own inference, correction, or judgment (for
  example a self-corrected 已修正 conclusion). Never label an assistant
  inference as "user".
"""


def memory_review_prompt(*, existing_text: str, transcript: str, today: str | None = None) -> str:
    """The background-review prompt: decide memory ops, consolidate, attribute."""
    today = today or memory_date()
    return f"""Review the completed conversation turn and decide whether long-term memory should change.

Today's date is {today}.

Memory types:
- user: durable facts, preferences, or instructions about the user.
- context: other durable context that may help future chats for this user account.

Rules:
- Save enduring details that can improve future conversations.
- Do not save one-off activity, meals, temporary mood, routine daily events, or other short-lived details unless the user explicitly asks to remember them.
- Do not save secrets, credentials, transient task steps, or unsupported guesses.
- Use path when there is a clear path for the memory.
- Leave path empty when there is no clear place for the memory.
- Prefer replace/move/remove over duplicate add when an existing memory should change.
- Do not invent type, status, trait, score, importance, or stability schemas.

{MEMORY_REVIEW_CONSOLIDATION_DUTY}
- Return only JSON in this shape:
  {{"operations":[
    {{"action":"add","type":"user|context","path":"...","content":"...","attribution":"user|assistant"}},
    {{"action":"replace","id":"...","type":"user|context","path":"...","content":"...","attribution":"user|assistant"}},
    {{"action":"move","id":"...","path":"..."}},
    {{"action":"remove","id":"..."}}
  ]}}
- Use an empty operations array if nothing should be remembered.

Existing memories:
{existing_text}

Conversation:
{transcript}
"""


async def _generate_memory_operations(
    *,
    request,
    user,
    model_id: str,
    metadata: dict,
    existing_text: str,
    transcript: str,
) -> list[dict[str, Any]]:
    from open_webui.utils.chat import generate_chat_completion

    review_prompt = memory_review_prompt(existing_text=existing_text, transcript=transcript)

    response = await generate_chat_completion(
        request,
        form_data={
            'model': model_id,
            'messages': [
                {
                    'role': 'system',
                    # LICENSE covers this Open WebUI system identifier.
                    # Do not alter, remove, obscure, or replace it except as LICENSE permits:
                    # https://docs.openwebui.com/license.
                    'content': "You are Open WebUI's private memory reviewer. Return only valid JSON.",
                },
                {'role': 'user', 'content': review_prompt},
            ],
            'stream': False,
            'metadata': {
                'task': 'memory_review',
                'chat_id': metadata.get('chat_id'),
                'message_id': metadata.get('message_id'),
            },
        },
        user=user,
    )

    if not isinstance(response, dict) or not response.get('choices'):
        return []

    response_message = response.get('choices', [{}])[0].get('message', {})
    content = response_message.get('content') or response_message.get('reasoning_content') or ''
    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1 or end < start:
        return []

    try:
        parsed = JSONCodec.loads(content[start : end + 1])
    except Exception:
        return []

    operations = parsed.get('operations') if isinstance(parsed, dict) else None
    return operations if isinstance(operations, list) else []
