"""Frozen, anchor-keyed message injections (#30239).

Mid-chat context (memory deltas, retrieval context) must not be spliced into
already-sent turns: that rewrites the request prefix and invalidates the
provider's prompt cache. Instead the *rendered text* is frozen at emit time and
recorded in ``chat.meta`` against the message id it augmented, then replayed
byte-identically at the same chronological position on every later turn.

Three producers write the same entry shape through the same helpers:

- ``memory_context.injections`` (see ``utils/memory.py``) — newly-relevant
  memories attached to the turn that surfaced them.
- ``rag_injections`` — retrieval context attached to the user turn.
- ``message_injections`` — the public channel for plugins/filters that need to
  deliver mid-chat context without rewriting the system prompt.

Every entry carries its ``source`` label (``'memory'``/``'rag'``/caller-chosen)
alongside the anchor::

    {'source': 'memory', 'anchor': '<message id>',
     'position': 'prepend'|'append', 'text': '<frozen text>',
     'ids': ['<id>', ...]}

Replay only ever splices frozen ``text``; it never re-renders from ids, so a
later edit or delete cannot change what an earlier turn sent.

The three producers keep *separate top-level meta keys*. ``Chats.update_chat_meta_by_id``
is a read-merge-write that replaces a whole top-level key from a stale read, so a
single shared key would let a memory write and a retrieval write clobber each
other; splitting by producer makes concurrent writers lose at most a same-key
write, and ledger writers are further gated to the primary branch.

**Plugins/filters:** the frozen system prompt overrides any post-freeze mutation
of the system message (``utils/system_baseline.py``). A filter that wants
mid-chat delivery must call ``record_message_injection`` and splice with
``attach_injection`` instead of editing the system message per turn.
"""

from __future__ import annotations

import logging

from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.misc import add_or_update_user_message, message_contains_text, update_message_content

log = logging.getLogger(__name__)

MEMORY_CONTEXT_KEY = 'memory_context'
RAG_INJECTIONS_KEY = 'rag_injections'
MESSAGE_INJECTIONS_KEY = 'message_injections'

_POSITIONS = ('prepend', 'append')

# UI status actions for harness-injection visibility (P1-3). When a turn emits a
# tail injection the frontend shows a collapsed one-line indicator under the
# message. The payload carries only the *kind* and a count — never the frozen
# ``text`` — so surfacing the event stays privacy-positive.
STATUS_MEMORY_CONTEXT_UPDATED = 'memory_context_updated'
STATUS_MEMORY_CONTEXT_UPDATE = 'memory_context_update'
STATUS_KNOWLEDGE_RETRIEVED = 'knowledge_retrieved'


def injection_status_event(action: str, count: int) -> dict:
    """Build the lightweight ``status`` event for a harness injection.

    Pure so it can be unit-tested. The event deliberately excludes the frozen
    injection text: the user learns *that* context was added and roughly how
    much, not the bytes.
    """
    return {
        'type': 'status',
        'data': {
            'action': action,
            'count': max(0, int(count)),
            'done': True,
        },
    }


async def emit_injection_status(event_emitter, action: str, count: int) -> None:
    """Best-effort publication of an injection status; never fails the turn."""
    if event_emitter is None or count <= 0:
        return
    try:
        await event_emitter(injection_status_event(action, count))
    except Exception as e:
        log.debug('Failed to emit injection status %s: %s', action, e)


def normalize_injections(raw, source: str | None = None) -> list[dict]:
    """Coerce stored ledger entries into the canonical shape, dropping junk.

    Legacy or hand-edited meta must never crash the request path, so anything
    that is not a well-formed entry is silently ignored. ``source`` supplies the
    label for entries stored before the field existed.
    """
    if not isinstance(raw, list):
        return []

    injections: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        anchor = entry.get('anchor')
        text = entry.get('text')
        if not anchor or not isinstance(text, str) or not text:
            continue
        position = entry.get('position')
        injections.append(
            {
                'source': str(entry.get('source') or source or ''),
                'anchor': str(anchor),
                'position': position if position in _POSITIONS else 'prepend',
                'text': text,
                'ids': [str(memory_id) for memory_id in entry.get('ids') or []],
            }
        )
    return injections


def upsert_injection(
    injections: list[dict],
    *,
    anchor: str | None,
    text: str | None,
    position: str = 'prepend',
    ids: list[str] | None = None,
    source: str | None = None,
) -> list[dict]:
    """Record one injection per anchor.

    A continuation request (whose last message is not a user turn) must not emit
    a second injection for an anchor it already augmented: replay keys off the
    anchor, so a duplicate would splice the same text twice.
    """
    if not anchor or not text:
        return injections

    anchor = str(anchor)
    if any(injection['anchor'] == anchor for injection in injections):
        return injections

    return [
        *injections,
        {
            'source': str(source or ''),
            'anchor': anchor,
            'position': position if position in _POSITIONS else 'prepend',
            'text': text,
            'ids': [str(memory_id) for memory_id in ids or []],
        },
    ]


def add_injection(
    injections: list[dict],
    *,
    anchor: str | None,
    text: str | None,
    position: str = 'append',
    ids: list[str] | None = None,
    source: str | None = None,
) -> list[dict]:
    """Record an additional injection for an anchor, deduped by exact text.

    Unlike ``upsert_injection`` this allows several blocks per anchor — one user
    turn may carry both a memory delta and a memory-edit notice, and both must
    replay at their original position. Retry safety comes from deduping on
    ``(anchor, text)``: replaying or re-emitting the same block is a no-op.
    """
    if not anchor or not text:
        return injections

    anchor = str(anchor)
    if any(injection['anchor'] == anchor and injection['text'] == text for injection in injections):
        return injections

    return [
        *injections,
        {
            'source': str(source or ''),
            'anchor': anchor,
            'position': position if position in _POSITIONS else 'append',
            'text': text,
            'ids': [str(memory_id) for memory_id in ids or []],
        },
    ]


def prune_injections(injections: list[dict], chain_ids: set[str]) -> list[dict]:
    """Drop entries whose anchor is no longer in the active message chain.

    A deleted message or a switched branch leaves an orphaned anchor; the entry
    is pruned lazily on the next write. Without a known chain nothing is pruned
    (an empty ``chain_ids`` must not wipe the ledger).
    """
    if not chain_ids:
        return injections
    return [injection for injection in injections if injection['anchor'] in chain_ids]


def apply_injections(messages: list[dict], injections: list[dict]) -> list[dict]:
    """Splice frozen injection text onto its anchored messages, in place.

    Idempotent: an injection whose text is already present in the message is not
    re-spliced. That covers retries and continuations, where the ledger entry for
    the current turn already exists but the raw DB message does not carry it.
    """
    if not messages or not injections:
        return messages

    by_anchor: dict[str, list[dict]] = {}
    for injection in injections:
        anchor = injection.get('anchor')
        if anchor and injection.get('text'):
            by_anchor.setdefault(anchor, []).append(injection)

    for message in messages:
        for injection in by_anchor.get(message.get('id'), []):
            text = injection['text']
            if not message_contains_text(message, text):
                update_message_content(message, text, append=injection.get('position') == 'append')

    return messages


def memory_injections(meta: dict | None) -> list[dict]:
    return normalize_injections(((meta or {}).get(MEMORY_CONTEXT_KEY) or {}).get('injections'), source='memory')


def rag_injections(meta: dict | None) -> list[dict]:
    return normalize_injections((meta or {}).get(RAG_INJECTIONS_KEY), source='rag')


def message_injections(meta: dict | None) -> list[dict]:
    return normalize_injections((meta or {}).get(MESSAGE_INJECTIONS_KEY))


def collect_injections(meta: dict | None) -> list[dict]:
    return [*memory_injections(meta), *rag_injections(meta), *message_injections(meta)]


def apply_chat_injections(messages: list[dict], meta: dict | None) -> list[dict]:
    return apply_injections(messages, collect_injections(meta))


def attach_injection(messages: list[dict], anchor: str | None, text: str, position: str = 'prepend') -> list[dict]:
    """Splice ``text`` onto the message ``anchor`` identifies, in place.

    The anchor is authoritative: a guided-regeneration prompt (a transient
    extra user turn with no id) or any other trailing message must not receive a
    block that the ledger records against ``u_N``. Falls back to the last user
    message when the anchor is not present.
    """
    if not messages or not text:
        return messages

    target = next((message for message in messages if anchor and message.get('id') == anchor), None)
    if target is None:
        return add_or_update_user_message(text, messages, append=position == 'append')
    if not message_contains_text(target, text):
        update_message_content(target, text, append=position == 'append')
    return messages


async def load_chat_meta(chat_id: str | None) -> dict:
    """Read a saved chat's meta, or ``{}`` when there is no persistence surface."""
    if not is_saved_chat_id(chat_id):
        return {}

    from open_webui.models.chats import Chats

    chat = await Chats.get_chat_by_id(chat_id)
    return (chat.meta or {}) if chat else {}


async def save_meta_key(chat_id: str | None, key: str, value) -> bool:
    if not is_saved_chat_id(chat_id):
        return False

    from open_webui.models.chats import Chats

    try:
        return await Chats.update_chat_meta_by_id(chat_id, {key: value})
    except Exception as e:
        log.debug('Failed to persist %s ledger: %s', key, e)
        return False


async def record_rag_injection(
    chat_id: str | None,
    *,
    anchor: str | None,
    text: str,
    meta: dict | None = None,
    chain_ids: set[str] | None = None,
    replace: bool = False,
) -> tuple[list[dict], bool]:
    """Freeze the retrieval text onto its user turn and persist the ledger.

    Returns ``(ledger, recorded)``. ``recorded`` is False when the anchor already
    had a frozen block — the caller must then not splice a second copy.

    Pass the ``meta`` and ``chain_ids`` the caller already has so the check, the
    prune, and the write share one view; when ``chain_ids`` is given, entries
    whose anchor left the active chain are dropped in the same write.

    ``replace=True`` makes this the anchor's new sole retrieval block. The
    approved-tool-call path uses it: after tool sources arrive it restores the
    user turn and re-splices a *combined* file+tool block, so the ledger must
    follow the final block or the next turn replays the stale file-only one.
    """
    if not is_saved_chat_id(chat_id) or not anchor or not text:
        return (rag_injections(meta) if meta is not None else []), False

    current = rag_injections(meta) if meta is not None else rag_injections(await load_chat_meta(chat_id))
    pruned = prune_injections(current, chain_ids or set())
    base = [entry for entry in pruned if entry['anchor'] != str(anchor)] if replace else pruned
    if not replace and any(injection['anchor'] == str(anchor) for injection in pruned):
        if pruned != current:
            await save_meta_key(chat_id, RAG_INJECTIONS_KEY, pruned)
        return pruned, False

    updated = upsert_injection(base, anchor=anchor, text=text, position='prepend', source='rag')
    if updated != current:
        await save_meta_key(chat_id, RAG_INJECTIONS_KEY, updated)
    return updated, True


async def record_message_injection(
    chat_id: str | None,
    *,
    anchor: str | None,
    text: str,
    position: str = 'prepend',
    source: str = 'plugin',
    meta: dict | None = None,
    chain_ids: set[str] | None = None,
    replace: bool = False,
) -> tuple[list[dict], bool]:
    """Record a frozen tail injection from a plugin/filter (public channel).

    The supported way for filters to deliver mid-chat context: the frozen system
    prompt overrides post-freeze system-message mutations, so per-turn content
    must go through this ledger and be spliced with ``attach_injection``.

    Returns ``(ledger, recorded)``; ``recorded`` is False when ``(anchor, text)``
    was already frozen, and the caller must then not splice a second copy. When
    ``chain_ids`` is given, orphaned anchors are pruned in the same write.

    ``replace=True`` makes the entry the anchor's sole entry for this ``source``
    (a re-run turn that produced a different outcome supersedes the old one,
    rather than accumulating a stale duplicate).
    """
    if not is_saved_chat_id(chat_id) or not anchor or not text:
        return (message_injections(meta) if meta is not None else []), False

    current = message_injections(meta) if meta is not None else message_injections(await load_chat_meta(chat_id))
    pruned = prune_injections(current, chain_ids or set())
    base = (
        [entry for entry in pruned if not (entry['anchor'] == str(anchor) and entry['source'] == source)]
        if replace
        else pruned
    )
    updated = add_injection(base, anchor=anchor, text=text, position=position, source=source)
    recorded = updated != current
    if recorded:
        await save_meta_key(chat_id, MESSAGE_INJECTIONS_KEY, updated)
    return updated, recorded
