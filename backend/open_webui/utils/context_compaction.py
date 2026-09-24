from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from open_webui.env import REDIS_KEY_PREFIX, REDIS_TASK_TTL
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.chat_injections import load_chat_meta, save_meta_key
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import get_content_from_message, get_last_user_message, get_message_list
from open_webui.utils.payload import apply_params_to_form_data
from open_webui.utils.task import (
    prompt_template,
    prompt_variables_template,
    replace_messages_variable,
    replace_prompt_variable,
)

log = logging.getLogger(__name__)

# Structured summary template (Slice B). Sections are omitted when not
# applicable; ``has_summary_section`` accepts any one of the ``##`` headings so
# a partial-but-structured response still passes validation.
SUMMARY_HEADINGS = (
    '## Objective',
    '## Requirements',
    '## Decisions',
    '## Work State',
    '## Next Move',
    '## Relevant Files',
    '## Important Context',
)

SUMMARY_SECTIONS = """Use this format; omit sections that do not apply. Do not include the <template> tags.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]

## Decisions
- [decisions already made and why, or "(none)"]

## Work State
Break the objective into smaller goals and report which are completed, which are being worked on, and which are blocked.
### Completed
- [goals that have been completed; otherwise "(none)"]

### Active
- [goals currently being worked on; otherwise "(none)"]

### Blocked
- [anything blocking progress, and why; otherwise "(none)"]

## Next Move
1. [ordered list of next actions, or "(none)"]

## Relevant Files
List files and directories another agent would need to continue this work. Include at most 15, most
important first. Do not list every file that was read or changed. If none, write "(none)".
- `[file or directory path]`: [brief reason it matters]

## Important Context
- [facts the next agent cannot continue without and cannot easily find on its own; or "(none)"]
</template>"""

SUMMARY_RULES = """Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers.
- Carry forward only user questions or requests that remain unanswered or require further action. Do not
  repeat ones that newer history has answered or resolved. Preserve exact wording when carrying one forward.
- Preserve consequential workflow state, including whether changes are uncommitted, committed, pushed,
  under review, or merged.
- Do not mention the summary process or that context was compacted.
- Do not continue the task or call tools.
- Return only the structured Markdown. Do not include a preamble, explanation, or other commentary."""

SUMMARY_SHARED_INSTRUCTION = (
    'Summarize only what the user and the assistant said and did; leave out instructions and setup the assistant '
    'was given rather than told by the user.'
)

DEFAULT_CONTEXT_COMPACTION_PROMPT = (
    'You MUST summarize the conversation history that will be compacted out of the active chat context into a '
    'structured summary that another agent will use to resume the work. '
    + SUMMARY_SHARED_INSTRUCTION
    + '\n\n'
    + SUMMARY_SECTIONS
    + '\n\n'
    + SUMMARY_RULES
    + '\n\n### Messages Being Compacted:\n{{COMPACTED_MESSAGES}}'
    + '\n\n### Recent Messages Kept In Context:\n{{RECENT_MESSAGES}}'
)

LEGACY_SUMMARY_INSTRUCTION = (
    'The existing checkpoint was written with an earlier format that recorded far more detail than this one asks '
    'for. Rewrite it at the level of detail described below rather than carrying its detail forward. Keep its '
    'requirements, decisions, and open questions; they came from earlier conversation with the user.'
)

UPDATE_CONTEXT_COMPACTION_PROMPT = (
    'Update and consolidate the existing checkpoint below with the newer history into one consolidated summary.'
    '\n\n'
    'Newer history always takes precedence over the existing checkpoint. Preserve previous information unless newer '
    'history clearly contradicts, supersedes, resolves, or makes it stale. If something is no longer relevant to '
    'continuing the work, you may remove it.\n\n'
    'Incorporate newer requirements, decisions, progress, and context. Reconcile Work State and Next Move: move '
    'completed work out of Active, remove resolved blockers and answered questions, and preserve unresolved or '
    'pending work.\n\n'
    'Return only the updated Markdown sections. Do not reproduce the `<conversation-checkpoint>`, `<summary>`, or '
    '`<recent-context>` wrapper tags from the previous checkpoint.\n\n'
    + SUMMARY_SHARED_INSTRUCTION
    + '\n\n'
    + SUMMARY_SECTIONS
    + '\n\n'
    + SUMMARY_RULES
    + '\n\n### Existing Checkpoint:\n{{PREVIOUS_SUMMARY}}'
    + '\n\n### Newer Messages Being Compacted:\n{{COMPACTED_MESSAGES}}'
    + '\n\n### Recent Messages Kept In Context:\n{{RECENT_MESSAGES}}'
)

CORRECTIVE_SUMMARY_PROMPT = (
    'The previous response did not fill in the required summary template. Do not call tools. Return the summary as '
    'text using the exact section headings from the template.'
)

DEFAULT_SUMMARY_MAX_TOKENS = 4096

# Structured checkpoint record stored as JSON in the existing ``context_summary``
# column / ``contextSummary`` map field. The boundary message carries it; the
# history before it is projected out at request-assembly time.
CHECKPOINT_VERSION = 1
CHECKPOINT_STATUS_COMPLETED = 'completed'
LEGACY_CHECKPOINT_VERSION = 0

DEFAULT_CONTEXT_COMPACTION_BUFFER = 20000
DEFAULT_CONTEXT_COMPACTION_KEEP_TOKENS = 15000
OUTPUT_TOKEN_MAX = 32000
TOOL_OUTPUT_MAX_CHARS = 2000

CHECKPOINT_TAG = 'conversation-checkpoint'

# ---------------------------------------------------------------------------
# C1 — per-chat compaction lock + lifecycle ledger
# ---------------------------------------------------------------------------

# Mirrors the Redis task lease pattern in ``open_webui.tasks`` (``SET NX EX``
# plus a token-guarded release) so the auto (middleware) and manual (router)
# paths cannot summarize the same chat at once. Redis being unavailable does not
# block chat: the lock degrades to an in-process guard and fails *open* across
# processes. The durable checkpoint stays correct regardless, because a
# compaction is idempotent for a chat and the stale ``running`` record is
# settled on the next attempt.
CONTEXT_COMPACTION_LOCK_KEY = f'{REDIS_KEY_PREFIX}:context_compaction:lock'
CONTEXT_COMPACTION_LOCK_TTL = max(REDIS_TASK_TTL, 600)
COMPACTION_IN_PROGRESS_DETAIL = 'Wait for the current response to finish before compacting.'

# Lifecycle ledger stored in ``chat.meta`` (never on the message); a compaction
# interrupted by process death leaves ``running`` behind and the next attempt
# (or a host-driven startup drain) settles it to ``failed``.
CONTEXT_COMPACTION_STATE_KEY = 'context_compaction'
STATUS_RUNNING = 'running'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'

_COMPACTION_LOCK_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

# In-process fallback used only when Redis is unavailable (see the lock's
# documented fail-open behavior). Keyed by chat id; token values are not needed
# because a single process cannot contend with itself on the same event loop.
_COMPACTION_LOCAL_LOCKS: set[str] = set()

# Conservative provider context-overflow signatures. Checked only after a
# provider call has already failed, so a broad phrase match cannot reject a
# valid request; the exception-name check catches typed SDK errors.
CONTEXT_OVERFLOW_MARKERS = (
    'context_length_exceeded',
    'maximum context length',
    'max context length',
    'context window',
    'context length',
    'too many tokens',
    'too many input tokens',
    'reduce the length',
    'prompt is too long',
    'input is too long',
    'input length exceeds',
    'exceeds the maximum number of tokens',
    'exceed the maximum number of tokens',
    'input token count exceeds',
    'request too large',
)
CONTEXT_OVERFLOW_ERROR_NAMES = ('contextoverflow', 'contextlengthexceeded', 'tokenlimit', 'toomanytokens')


async def compact_messages_for_request(
    request,
    user,
    messages: list[dict],
    metadata: dict,
    model_id: str,
    models: dict,
    system_prompt: str = '',
    *,
    force: bool = False,
    keep_tokens_override: int | None = None,
) -> tuple[list[dict], dict | None]:
    """Project the chat history onto a budgeted tail plus a durable checkpoint.

    Returns ``(messages, checkpoint_record)``. ``messages`` are the system
    messages followed by the retained tail; the caller renders
    ``checkpoint_record`` into a transient user-role ``<conversation-checkpoint>``
    message (see ``build_checkpoint_message``). ``checkpoint_record`` is the
    existing record when no compaction runs this turn, so an already-compacted
    chat keeps projecting its checkpoint.

    ``force`` bypasses the auto/threshold gates and ``keep_tokens_override``
    lowers the retained-tail budget; together they back the one-shot reactive
    overflow recovery (Slice C2). The lock/lifecycle handling is unchanged, so a
    concurrent manual compaction still wins and this call skips compaction.
    """
    config = await _load_config()
    if not config['enable']:
        return messages, None

    system_messages = [messages[0]] if messages and messages[0].get('role') == 'system' else []
    messages = messages[1:] if system_messages else messages
    # Drop any transient checkpoint rendered by an earlier pass in this same
    # request (reactive overflow recovery reuses this function); the checkpoint
    # is re-derived below from the durable record.
    messages = _strip_transient_checkpoints(messages)

    # Read shim: drop everything before the newest boundary message carrying a
    # checkpoint (structured record or legacy bare string).
    newest_is_checkpoint = _newest_is_checkpoint(messages)
    messages, checkpoint = _apply_latest_summary_checkpoint(messages)
    previous_summary = checkpoint.get('summary') if checkpoint else None

    if not force and (not config['auto'] or newest_is_checkpoint):
        return [*system_messages, *messages], checkpoint

    model_info = models.get(model_id) if isinstance(models, dict) else None
    ceiling = _resolve_prompt_ceiling(config, metadata, model_info)
    if not force and (
        not _exceeds_token_threshold(messages, system_prompt, previous_summary, ceiling) or len(messages) <= 3
    ):
        return [*system_messages, *messages], checkpoint

    keep_tokens = keep_tokens_override or config['keep_tokens']
    tail_start = _find_tail_start(messages, keep_tokens)
    if tail_start is None or tail_start <= 0:
        return [*system_messages, *messages], checkpoint

    compacted_messages = messages[:tail_start]
    recent_messages = messages[tail_start:]
    if not compacted_messages or not recent_messages:
        return [*system_messages, *messages], checkpoint

    record = await _execute_compaction(
        request,
        user,
        model_id,
        models,
        metadata,
        compacted_messages,
        recent_messages,
        previous_summary,
        config,
        ceiling,
    )
    if record is None:
        # Another compaction holds the per-chat lock; continue this turn with the
        # full (or already-checkpointed) history rather than erroring.
        return [*system_messages, *messages], checkpoint

    return [*system_messages, *recent_messages], record


async def _execute_compaction(
    request,
    user,
    model_id: str,
    models: dict,
    metadata: dict,
    compacted_messages: list[dict],
    recent_messages: list[dict],
    previous_summary: str | None,
    config: dict,
    ceiling: int,
) -> dict | None:
    """Summarize, persist the checkpoint, and drive the lifecycle ledger.

    Returns the checkpoint record, or ``None`` when the per-chat lock is held by
    a concurrent compaction. The lifecycle ledger goes to ``running`` first and
    settles to ``completed``/``failed`` afterwards; a crashed run is settled by
    the next attempt (``settle_stale_compaction``).
    """
    chat_id = metadata.get('chat_id')
    await settle_stale_compaction(chat_id)
    lock_token = await acquire_compaction_lock(request, chat_id)
    if chat_id and lock_token is None:
        log.info('Context compaction already in progress for chat=%s; skipping', chat_id)
        return None

    event_emitter = None
    try:
        if chat_id and metadata.get('message_id'):
            from open_webui.socket.main import get_event_emitter

            event_emitter = await get_event_emitter(metadata)

        await _record_compaction_state(chat_id, STATUS_RUNNING, model=model_id)
        await _emit_compaction_status(event_emitter, 'Compacting context', done=False)
        try:
            summary = await _generate_summary(
                request,
                user,
                model_id,
                models,
                _truncate_tool_outputs_in_messages(compacted_messages),
                recent_messages,
                previous_summary,
                config['prompt_template'],
            )
        except Exception:
            await _emit_compaction_status(event_emitter, 'Context compaction failed', done=True, error=True)
            raise

        checkpoint_message_id = (
            recent_messages[0].get('id') or metadata.get('user_message_id') or metadata.get('message_id')
        )
        record = _build_checkpoint_record(
            summary=summary,
            recent=_serialize_recent_messages(recent_messages),
            dropped_messages=compacted_messages,
            model=model_id,
            tokens=ceiling,
        )
        if is_saved_chat_id(chat_id) and checkpoint_message_id:
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                checkpoint_message_id,
                {'contextSummary': _serialize_checkpoint(record)},
                touch=False,
            )

        log.info(
            'Compacted chat context for chat=%s checkpoint=%s response=%s dropped=%d kept=%d summary_chars=%d',
            chat_id,
            checkpoint_message_id,
            metadata.get('message_id'),
            len(compacted_messages),
            len(recent_messages),
            len(summary),
        )

        await _emit_compaction_status(event_emitter, 'Context compacted', done=True)
    except Exception as exc:
        await _record_compaction_state(chat_id, STATUS_FAILED, model=model_id, error=str(exc)[:500])
        raise
    else:
        # Publish ``completed`` while the lease is still held so a concurrent
        # attempt cannot acquire the lock and see a stale ``running`` record
        # (mirrors ``compact_chat_branch``'s else/finally ordering).
        await _record_compaction_state(chat_id, STATUS_COMPLETED, model=model_id)
        _mark_compaction_ran(request)
    finally:
        await release_compaction_lock(request, chat_id, lock_token)

    return record


def _mark_compaction_ran(request) -> None:
    """Flag this logical step so reactive overflow recovery never follows it."""
    state = getattr(request, 'state', None)
    if state is not None:
        setattr(state, 'context_compaction_ran', True)


async def _emit_compaction_status(event_emitter, description: str, *, done: bool, error: bool = False) -> None:
    if not event_emitter:
        return
    data = {'action': 'context_compaction', 'description': description, 'done': done}
    if error:
        data['error'] = True
    await event_emitter({'type': 'context_compaction', 'data': data})


def _get_redis(request):
    """The app's shared async Redis client, or ``None`` when Redis is disabled."""
    app = getattr(request, 'app', None)
    state = getattr(app, 'state', None)
    return getattr(state, 'redis', None)


async def acquire_compaction_lock(request, chat_id: str | None) -> str | None:
    """Acquire the per-chat compaction lease; ``None`` means another run holds it.

    ``SET NX EX`` mirrors the task lease in ``open_webui.tasks``. When Redis is
    unavailable the lock degrades to an in-process guard and **fails open**
    across processes: a Redis outage must never block chat, and the checkpoint
    write stays idempotent. This matches the permissive local fallback of
    ``has_active_tasks``.
    """
    token = uuid.uuid4().hex
    if not chat_id:
        return token

    redis = _get_redis(request)
    if redis is None:
        if chat_id in _COMPACTION_LOCAL_LOCKS:
            return None
        _COMPACTION_LOCAL_LOCKS.add(chat_id)
        return token

    try:
        acquired = await redis.set(
            f'{CONTEXT_COMPACTION_LOCK_KEY}:{chat_id}',
            token,
            nx=True,
            ex=CONTEXT_COMPACTION_LOCK_TTL,
        )
    except Exception as e:
        log.warning('Compaction lock acquire failed for chat=%s; using local guard: %s', chat_id, e)
        if chat_id in _COMPACTION_LOCAL_LOCKS:
            return None
        _COMPACTION_LOCAL_LOCKS.add(chat_id)
        return token
    return token if acquired else None


async def release_compaction_lock(request, chat_id: str | None, token: str | None) -> None:
    """Release the lease, only if this caller still owns it (token match)."""
    if not chat_id:
        return
    _COMPACTION_LOCAL_LOCKS.discard(chat_id)
    redis = _get_redis(request)
    if redis is None or token is None:
        return
    try:
        await redis.eval(_COMPACTION_LOCK_RELEASE_SCRIPT, 1, f'{CONTEXT_COMPACTION_LOCK_KEY}:{chat_id}', token)
    except Exception as e:
        # The key expires on its own; a failed release must not fail the turn.
        log.warning('Compaction lock release failed for chat=%s; it expires on its own: %s', chat_id, e)


async def _record_compaction_state(chat_id: str | None, status: str, **extra) -> bool:
    """Persist the lifecycle status in ``chat.meta`` (never on the message)."""
    return await save_meta_key(
        chat_id,
        CONTEXT_COMPACTION_STATE_KEY,
        {'status': status, 'updated_at': int(time.time()), **extra},
    )


async def settle_stale_compaction(chat_id: str | None) -> bool:
    """Settle a ``running`` record left by a crash/restart to ``failed``.

    Idempotent: a second call sees ``failed`` (or no record) and returns False.
    Called at the start of every compaction attempt; a host may also call it once
    per chat on startup to drain interrupted runs.

    A ``running`` record is only settled once it is older than the lock TTL. A
    concurrent attempt calls this *before* acquiring the lock, so without the age
    guard it could mark a genuinely in-flight sibling compaction (still inside
    ``_generate_summary``) as failed. Because the lease is exactly the window in
    which a run counts as live, an expired record can no longer be in flight.
    """
    record = (await load_chat_meta(chat_id)).get(CONTEXT_COMPACTION_STATE_KEY)
    if not isinstance(record, dict) or record.get('status') != STATUS_RUNNING:
        return False
    updated_at = record.get('updated_at')
    if isinstance(updated_at, (int, float)) and not isinstance(updated_at, bool):
        if time.time() - updated_at < CONTEXT_COMPACTION_LOCK_TTL:
            return False
    await _record_compaction_state(chat_id, STATUS_FAILED, error='Compaction was interrupted', settled=True)
    return True


def _error_text(error: Any) -> str:
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        try:
            return JSONCodec.dumps(error, ensure_ascii=False)
        except Exception:
            return str(error)
    detail = getattr(error, 'detail', None)
    if detail is not None:
        return detail if isinstance(detail, str) else str(detail)
    body = getattr(error, 'body', None)
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            return bytes(body).decode('utf-8', 'replace')
        return str(body)
    return str(error)


def is_context_overflow_error(error: Any) -> bool:
    """True for a provider context-overflow / too-many-tokens failure.

    Called only after a provider call has already failed, so a broad phrase can
    never reject a valid request. Matches a typed SDK exception name as well as
    the message text of a raw string, dict body, ``HTTPException`` detail, or
    response ``.body``.
    """
    if any(marker in type(error).__name__.lower() for marker in CONTEXT_OVERFLOW_ERROR_NAMES):
        return True
    text = _error_text(error).lower()
    return any(marker in text for marker in CONTEXT_OVERFLOW_MARKERS)


async def compact_for_overflow(
    request,
    user,
    messages: list[dict],
    metadata: dict,
    model_id: str,
    models: dict,
    system_prompt: str = '',
) -> tuple[list[dict], dict | None]:
    """One forced, lower-budget compaction for reactive overflow recovery.

    Reuses the normal projection with the auto/threshold gates bypassed and the
    retained tail cut to a quarter of the configured budget, so the retried
    request is materially smaller than the one that overflowed.
    """
    config = await _load_config()
    keep_tokens = max(1, (config.get('keep_tokens') or DEFAULT_CONTEXT_COMPACTION_KEEP_TOKENS) // 4)
    return await compact_messages_for_request(
        request,
        user,
        messages,
        metadata,
        model_id,
        models,
        system_prompt,
        force=True,
        keep_tokens_override=keep_tokens,
    )


async def compact_chat_branch(request, user, chat: Any, model_id: str, models: dict) -> dict:
    config = await _load_config()
    if not config['enable']:
        return {'ok': True, 'compacted': False, 'reason': 'disabled'}

    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return {'ok': True, 'compacted': False, 'reason': 'empty'}

    messages_map = await Chats.get_messages_map_by_chat_id(chat.id)
    if not messages_map:
        messages_map = history.get('messages') or {}

    messages, previous_checkpoint = _apply_latest_summary_checkpoint(get_message_list(messages_map, current_id))
    compacted_messages = messages[:-1]
    recent_messages = messages[-1:]
    if not compacted_messages or not recent_messages:
        return {'ok': True, 'compacted': False, 'reason': 'too_short'}

    previous_summary = previous_checkpoint.get('summary') if previous_checkpoint else None
    chat_id = chat.id
    await settle_stale_compaction(chat_id)
    lock_token = await acquire_compaction_lock(request, chat_id)
    if chat_id and lock_token is None:
        raise HTTPException(status_code=409, detail=COMPACTION_IN_PROGRESS_DETAIL)

    try:
        await _record_compaction_state(chat_id, STATUS_RUNNING, model=model_id, manual=True)
        summary = await _generate_summary(
            request,
            user,
            model_id,
            models,
            _truncate_tool_outputs_in_messages(compacted_messages),
            recent_messages,
            previous_summary,
            config['prompt_template'],
        )
        record = _build_checkpoint_record(
            summary=summary,
            recent=_serialize_recent_messages(recent_messages),
            dropped_messages=compacted_messages,
            model=model_id,
            tokens=None,
        )
        await Chats.upsert_message_to_chat_by_id_and_message_id(
            chat.id, current_id, {'contextSummary': _serialize_checkpoint(record)}, touch=False
        )
    except Exception as exc:
        await _record_compaction_state(chat_id, STATUS_FAILED, model=model_id, manual=True, error=str(exc)[:500])
        raise
    else:
        await _record_compaction_state(chat_id, STATUS_COMPLETED, model=model_id, manual=True)
    finally:
        await release_compaction_lock(request, chat_id, lock_token)

    return {
        'ok': True,
        'compacted': True,
        'dropped_messages': len(compacted_messages),
        'kept_messages': len(recent_messages),
        'summary_chars': len(summary),
    }


async def _load_config() -> dict:
    values = await Config.get_many(
        'chat.context_compaction.enable',
        'chat.context_compaction.token_threshold',
        'chat.context_compaction.token_cap',
        'chat.context_compaction.buffer',
        'chat.context_compaction.keep_tokens',
        'chat.context_compaction.auto',
        'chat.context_compaction.prompt_template',
    )
    token_threshold = _parse_positive_int(values.get('chat.context_compaction.token_threshold')) or 80000
    return {
        'enable': _as_bool(values.get('chat.context_compaction.enable'), False),
        'token_threshold': token_threshold,
        'token_cap': _parse_positive_int(values.get('chat.context_compaction.token_cap')) or token_threshold,
        'buffer': _parse_positive_int(values.get('chat.context_compaction.buffer'))
        or DEFAULT_CONTEXT_COMPACTION_BUFFER,
        'keep_tokens': _parse_positive_int(values.get('chat.context_compaction.keep_tokens'))
        or DEFAULT_CONTEXT_COMPACTION_KEEP_TOKENS,
        'auto': _as_bool(values.get('chat.context_compaction.auto'), True),
        'prompt_template': values.get('chat.context_compaction.prompt_template', '') or '',
    }


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _first_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return 0


def _cache_token_count(usage: dict) -> int:
    """Prompt-cache read/write tokens reported alongside the prompt.

    Providers disagree on whether ``prompt_tokens`` already contains the cached
    input; this is an estimate anchor only, mirroring the OpenCode v2
    ``input + cache.read + cache.write + output`` sum.
    """
    details = usage.get('prompt_tokens_details') if isinstance(usage.get('prompt_tokens_details'), dict) else {}
    input_details = usage.get('input_tokens_details') if isinstance(usage.get('input_tokens_details'), dict) else {}
    cache_read = _first_int(
        usage.get('cache_read_input_tokens'),
        usage.get('cached_tokens'),
        usage.get('total_cached_tokens'),
        usage.get('prompt_cache_hit_tokens'),
        details.get('cached_tokens'),
        input_details.get('cached_tokens'),
    )
    cache_write = _first_int(
        usage.get('cache_creation_input_tokens'),
        usage.get('cache_write_tokens'),
        details.get('cache_write_tokens'),
    )
    return cache_read + cache_write


def _resolve_token_threshold(global_threshold: int, global_cap: int, metadata: dict) -> int:
    configured_threshold = _parse_positive_int((metadata.get('params') or {}).get('compact_token_threshold'))
    return min(configured_threshold or global_threshold, global_cap)


def _model_context_limit(model_info: Any) -> dict | None:
    """Optional model-derived context limits, used only when metadata is present.

    This repository ships no model context-length metadata, so this returns
    ``None`` for every real model today and the configured threshold stays the
    primary trigger. It exists so a deployment that *does* annotate a model
    (``info.meta`` or ``info.params``) can lower the ceiling.
    """
    if not isinstance(model_info, dict):
        return None

    info = model_info.get('info') if isinstance(model_info.get('info'), dict) else model_info
    meta = info.get('meta') if isinstance(info.get('meta'), dict) else {}
    params = info.get('params') if isinstance(info.get('params'), dict) else {}

    def pick(*keys: str) -> int | None:
        for source in (meta, params, model_info):
            for key in keys:
                value = source.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
        return None

    context = pick('context_length', 'context_window', 'max_context_tokens', 'num_ctx')
    input_limit = pick('max_input_tokens', 'input_token_limit')
    output = pick('max_output_tokens', 'output_token_limit')
    if context is None and input_limit is None:
        return None
    return {'context': context, 'input': input_limit, 'output': output}


def _resolve_prompt_ceiling(config: dict, metadata: dict | None = None, model_info: Any = None) -> int:
    """Single prompt-budget ceiling shared by the trigger and the context ring.

    Primary source is the configured (or per-request) token threshold; model
    metadata can only lower it, and only when present.
    """
    ceiling = _resolve_token_threshold(config['token_threshold'], config['token_cap'], metadata or {})

    limit = _model_context_limit(model_info)
    if limit:
        buffer = config.get('buffer') or DEFAULT_CONTEXT_COMPACTION_BUFFER
        candidates = [ceiling]
        input_limit = limit.get('input')
        context = limit.get('context')
        if input_limit:
            candidates.append(max(1, input_limit - buffer))
        if context:
            output = min(limit.get('output') or 0, OUTPUT_TOKEN_MAX)
            candidates.append(max(1, context - max(output, buffer)))
        ceiling = min(candidates)

    return max(1, ceiling)


def _message_usage(message: Any) -> dict | None:
    if not isinstance(message, dict):
        return None
    usage = message.get('usage') or (message.get('info') or {}).get('usage')
    return usage if isinstance(usage, dict) else None


def _usage_token_count(usage: dict) -> int:
    prompt_tokens = int(usage.get('prompt_tokens') or usage.get('prompt_eval_count') or 0)
    if not prompt_tokens and (usage.get('prompt_n') is not None or usage.get('cache_n') is not None):
        prompt_tokens = int(usage.get('prompt_n') or 0) + int(usage.get('cache_n') or 0)
    if not prompt_tokens:
        prompt_tokens = int(usage.get('input_tokens') or 0)

    completion_tokens = int(
        usage.get('completion_tokens')
        or usage.get('output_tokens')
        or usage.get('eval_count')
        or usage.get('predicted_n')
        or 0
    )
    return prompt_tokens + completion_tokens + _cache_token_count(usage)


def _estimate_context_tokens(messages: list[dict], system_prompt: str = '', summary: str | None = None) -> int:
    """Estimate the live context size.

    Anchor on the newest message carrying provider ``usage`` (input + cache
    read/write + output) and add estimated deltas for the newer messages; fall
    back to a CJK-aware transcript estimate when no usage exists.
    """
    for idx in range(len(messages) - 1, -1, -1):
        usage = _message_usage(messages[idx])
        if usage and (tokens := _usage_token_count(usage)):
            return tokens + _estimate_messages_tokens(messages[idx + 1 :])

    return _estimate_tokens(system_prompt) + _estimate_tokens(summary or '') + _estimate_messages_tokens(messages)


def _exceeds_token_threshold(
    messages: list[dict], system_prompt: str, summary: str | None, threshold: int
) -> bool:
    if threshold <= 0:
        return False
    return _estimate_context_tokens(messages, system_prompt, summary) > threshold


async def get_chat_context_usage(chat: Any, model_id: str | None = None) -> dict | None:
    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return None

    messages_map = await Chats.get_messages_map_by_chat_id(chat.id)
    messages = get_message_list(messages_map or history.get('messages') or {}, current_id)
    if not messages:
        return None

    config = await _load_config()
    if not config['enable']:
        return None

    params = ((chat.chat or {}).get('params') or {}).copy()
    if model_id:
        params['model'] = model_id

    # Same ceiling and same single walk as the auto trigger, so the reported
    # percentage cannot diverge from what actually compacts.
    threshold = _resolve_prompt_ceiling(config, {'params': params})
    messages, checkpoint = _apply_latest_summary_checkpoint(messages)
    summary = checkpoint.get('summary') if checkpoint else None
    tokens = _estimate_context_tokens(messages, '', summary)
    return _build_context_usage(
        tokens,
        threshold,
        keep_tokens=config['keep_tokens'],
        buffer=config['buffer'],
    )


def _build_context_usage(
    tokens: int,
    threshold: int,
    *,
    keep_tokens: int | None = None,
    buffer: int | None = None,
) -> dict:
    return {
        'tokens': tokens,
        'estimated_tokens': tokens,
        'threshold': threshold,
        'percent': round((tokens / threshold) * 100) if threshold > 0 else 0,
        'source': 'estimated',
        'keep_tokens': keep_tokens,
        'buffer': buffer,
    }


def _newest_is_checkpoint(messages: list[dict]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        return _parse_checkpoint(message.get('contextSummary') or message.get('context_summary')) is not None
    return False


def _strip_transient_checkpoints(messages: list[dict]) -> list[dict]:
    """Drop transient ``<conversation-checkpoint>`` user messages.

    The checkpoint is never persisted; it is re-derived from the durable record
    at request assembly. A second projection pass in the same request (reactive
    overflow recovery) must not feed the previously rendered checkpoint back in
    as summarized history.
    """
    prefix = f'<{CHECKPOINT_TAG}>'
    return [
        message
        for message in messages
        if not (
            isinstance(message, dict)
            and message.get('role') == 'user'
            and isinstance(message.get('content'), str)
            and message['content'].startswith(prefix)
        )
    ]


def _apply_latest_summary_checkpoint(messages: list[dict]) -> tuple[list[dict], dict | None]:
    """Read shim: truncate before the newest checkpoint boundary message.

    Accepts both the structured JSON record and a legacy bare summary string so
    chats compacted before the rewrite still truncate correctly.
    """
    record = None
    boundary_idx = None

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        value = message.get('contextSummary') or message.get('context_summary')
        parsed = _parse_checkpoint(value)
        if parsed is not None:
            record = parsed
            boundary_idx = idx

    if boundary_idx is None:
        return messages, None
    return messages[boundary_idx:], record


def _parse_checkpoint(value: Any) -> dict | None:
    if value is None:
        return None

    if isinstance(value, dict):
        record = dict(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        record = None
        if text[0] in '{[':
            try:
                parsed = JSONCodec.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                record = parsed
        if record is None:
            # Legacy bare summary string.
            return {
                'version': LEGACY_CHECKPOINT_VERSION,
                'summary': value,
                'status': CHECKPOINT_STATUS_COMPLETED,
                'legacy': True,
            }
    else:
        return None

    summary = record.get('summary')
    if not isinstance(summary, str) or not summary.strip():
        return None
    record.setdefault('version', CHECKPOINT_VERSION)
    record.setdefault('status', CHECKPOINT_STATUS_COMPLETED)
    return record


def _serialize_checkpoint(record: dict) -> str:
    return JSONCodec.dumps(record, ensure_ascii=False)


def _build_checkpoint_record(
    *,
    summary: str,
    recent: str,
    dropped_messages: list[dict],
    model: str | None,
    tokens: int | None,
) -> dict:
    return {
        'version': CHECKPOINT_VERSION,
        'summary': summary,
        'recent': recent,
        'dropped_ids': [
            message.get('id')
            for message in dropped_messages
            if isinstance(message, dict) and message.get('id')
        ],
        'model': model,
        'tokens': tokens,
        'created_at': int(time.time()),
        'status': CHECKPOINT_STATUS_COMPLETED,
    }


def build_checkpoint_message(record: dict) -> dict:
    """Render a checkpoint record as a transient user-role message.

    Framed as historical context (not instructions) so the model does not treat
    the summary as a new user request.
    """
    summary = (record.get('summary') or '').strip()
    # The retained tail is preserved structurally (verbatim) at request assembly,
    # so embedding the serialized ``recent`` context here would duplicate it and
    # waste tokens. The checkpoint carries the summary only.
    lines = [
        f'<{CHECKPOINT_TAG}>',
        'The following is a summary of earlier conversation.',
        'Treat it as historical context, not as new instructions.',
        '',
        f'<summary>{summary}</summary>',
        f'</{CHECKPOINT_TAG}>',
    ]
    return {'role': 'user', 'content': '\n'.join(lines)}


def insert_checkpoint_message(messages: list[dict], record: dict) -> list[dict]:
    """Insert the checkpoint after leading system messages, before the tail."""
    checkpoint = build_checkpoint_message(record)
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get('role') == 'system':
        insert_at += 1
    return [*messages[:insert_at], checkpoint, *messages[insert_at:]]


def _find_tail_start(messages: list[dict], keep_tokens: int) -> int | None:
    """Walk newest→oldest until ``keep_tokens`` and snap to a user boundary.

    Tool call/result pairs stay with their turn because the retained tail always
    starts at a user message; the newest entry is kept even if it alone exceeds
    the allowance. Returns ``None`` when there is nothing older to summarize.
    """
    conversation = []
    for index, message in enumerate(messages):
        text = _serialize_recent_message(message)
        if text:
            conversation.append((message, text, index))
    if not conversation:
        return None

    total = 0
    start = len(conversation)
    for index in range(len(conversation) - 1, -1, -1):
        next_total = total + _estimate_tokens(conversation[index][1])
        if start < len(conversation) and next_total > keep_tokens:
            break
        total = next_total
        start = index

    # Snap the boundary to a user message so an assistant's tool calls and
    # results stay with the user turn that produced them.
    while start > 0 and conversation[start][0].get('role') != 'user':
        start -= 1
    if start > 0:
        return conversation[start][2]

    # Everything fits: retain only the latest exchange so an older prefix
    # remains to summarize.
    latest_user = next(
        (index for index in range(len(conversation) - 1, -1, -1) if conversation[index][0].get('role') == 'user'),
        None,
    )
    if latest_user is not None and latest_user > 0:
        return conversation[latest_user][2]
    return None


def _serialize_recent_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ''

    role = message.get('role') or 'unknown'
    content = get_content_from_message(message)

    if role == 'tool':
        return f'[Tool result]: {truncate_tool_output(content or "")}' if content else ''

    if role == 'assistant':
        return _serialize_assistant_message(message, content)

    if role == 'system':
        return ''

    if role == 'user':
        return f'[User]: {content}' if content else ''

    return f'[{role}]: {content}' if content else ''


def _serialize_assistant_message(message: dict, content: str | None) -> str:
    parts = [f'[Assistant]: {content}'] if content else []
    for item in message.get('output') or []:
        if isinstance(item, dict) and (line := _serialize_output_item(item)):
            parts.append(line)
    return '\n'.join(parts)


def _serialize_output_item(item: dict) -> str:
    item_type = item.get('type')
    if item_type == 'function_call':
        arguments = item.get('arguments')
        if not isinstance(arguments, str):
            try:
                arguments = JSONCodec.dumps(arguments, ensure_ascii=False)
            except Exception:
                arguments = str(arguments)
        return f'[Assistant tool call]: {item.get("name") or ""}({arguments})'
    if item_type == 'function_call_output':
        return f'[Tool result]: {truncate_tool_output(_output_parts_text(item.get("output")))}'
    return ''


def _serialize_recent_messages(messages: list[dict]) -> str:
    return '\n\n'.join(text for text in (_serialize_recent_message(message) for message in messages) if text)


def truncate_tool_output(value: Any) -> str:
    if not isinstance(value, str):
        if value is None:
            return ''
        try:
            value = JSONCodec.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    if len(value) <= TOOL_OUTPUT_MAX_CHARS:
        return value
    return f'{value[:TOOL_OUTPUT_MAX_CHARS]}\n[truncated]'


def _output_parts_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return ''
    texts = []
    for part in output:
        if isinstance(part, dict):
            text = part.get('text')
            if isinstance(text, str):
                texts.append(text)
    return '\n'.join(texts)


def _truncate_output_parts(parts: Any) -> Any:
    if not isinstance(parts, list):
        return parts
    truncated = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get('text'), str):
            truncated.append({**part, 'text': truncate_tool_output(part['text'])})
        else:
            truncated.append(part)
    return truncated


def _truncate_tool_outputs_in_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    result = dict(message)
    content = result.get('content')
    if result.get('role') == 'tool':
        if isinstance(content, str):
            result['content'] = truncate_tool_output(content)
        elif isinstance(content, list):
            result['content'] = _truncate_output_parts(content)

    output = result.get('output')
    if isinstance(output, list):
        cleaned = []
        for item in output:
            if isinstance(item, dict) and item.get('type') == 'function_call_output':
                if isinstance(item.get('output'), str):
                    item = {**item, 'output': truncate_tool_output(item['output'])}
                else:
                    item = {**item, 'output': _truncate_output_parts(item.get('output'))}
            cleaned.append(item)
        result['output'] = cleaned

    return result


def _truncate_tool_outputs_in_messages(messages: list[dict]) -> list[dict]:
    return [_truncate_tool_outputs_in_message(message) for message in messages]


def has_summary_section(summary: str) -> bool:
    """True when the summary uses at least one required ``##`` heading."""
    if not isinstance(summary, str):
        return False
    return any(line.strip() in SUMMARY_HEADINGS for line in summary.splitlines())


def _is_legacy_summary(summary: str | None) -> bool:
    """True for a non-empty summary written before the structured template.

    A legacy checkpoint (bare string or free-form text) has none of the new
    headings, so the update prompt asks the model to rewrite it at the new level
    of detail instead of carrying its old detail forward.
    """
    return bool(summary and summary.strip()) and not has_summary_section(summary)


def build_compaction_prompt(update: bool, legacy: bool = False, template: str = '') -> str:
    """Select the summary prompt template.

    An admin-configured ``template`` always wins so deployments keep full
    control. Otherwise a fresh prompt summarizes the first checkpoint and an
    update/consolidate prompt reconciles it with newer history. ``legacy`` adds
    an instruction to rewrite an existing free-form checkpoint at the new level
    of detail.
    """
    custom = (template or '').strip()
    if custom:
        return custom
    if not update:
        return DEFAULT_CONTEXT_COMPACTION_PROMPT
    if legacy:
        return f'{LEGACY_SUMMARY_INSTRUCTION}\n\n{UPDATE_CONTEXT_COMPACTION_PROMPT}'
    return UPDATE_CONTEXT_COMPACTION_PROMPT


def _summary_task_model_params(task_model_params: Any, model: Any) -> dict:
    """Resolve the summarizer request params with a bounded output size.

    An explicit ``max_tokens`` always wins; otherwise the task model's own
    parameter bound is used, then a sane default. The summarizer must never send
    an unbounded ``max_tokens``.
    """
    params = (
        {key: value for key, value in task_model_params.items() if value is not None and value != ''}
        if isinstance(task_model_params, dict)
        else {}
    )
    if params.get('max_tokens'):
        return params
    info = model.get('info') if isinstance(model, dict) and isinstance(model.get('info'), dict) else {}
    model_params = info.get('params') if isinstance(info.get('params'), dict) else {}
    return {**params, 'max_tokens': model_params.get('max_tokens') or DEFAULT_SUMMARY_MAX_TOKENS}


def _degraded_summary(previous_summary: str | None, compacted_messages: list[dict]) -> str:
    """Fallback when the summarizer fails to produce a structured summary."""
    parts = [previous_summary] if previous_summary else []
    for message in compacted_messages:
        content = get_content_from_message(message)
        if content:
            parts.append(f'- {message.get("role", "unknown")}: {content[:500]}')
    return '\n'.join(parts)[:4000]


async def _request_summary_completion(request, payload: dict, user) -> str:
    from open_webui.utils.chat import generate_chat_completion

    response = await generate_chat_completion(request, form_data=payload, user=user)
    return _response_text(response).strip()


async def _generate_summary_with_retry(
    request,
    user,
    payload: dict,
    previous_summary: str | None,
    compacted_messages: list[dict],
    require_headings: bool = True,
) -> str:
    """Generate the summary, allowing exactly one corrective retry.

    The first response carrying a required section heading wins. Otherwise the
    same request is retried once with an explicit instruction, and a second miss
    falls through to the degraded truncated-history fallback. Never loops more
    than once.

    ``require_headings`` is False when an admin-configured template drives the
    prompt: this module cannot know that template's section headings, so any
    non-empty response is accepted and a valid custom summary is never discarded
    for lacking the built-in headings.
    """

    def acceptable(text: str) -> bool:
        return has_summary_section(text) if require_headings else bool(text.strip())

    summary = await _request_summary_completion(request, payload, user)
    if acceptable(summary):
        return summary

    corrective_payload = {
        **payload,
        'messages': [
            *payload.get('messages', []),
            {'role': 'user', 'content': CORRECTIVE_SUMMARY_PROMPT},
        ],
    }
    summary = await _request_summary_completion(request, corrective_payload, user)
    if acceptable(summary):
        return summary

    log.warning('Context compaction summary did not match the required template; using degraded fallback')
    return _degraded_summary(previous_summary, compacted_messages)


async def _generate_summary(
    request,
    user,
    model_id: str,
    models: dict,
    compacted_messages: list[dict],
    recent_messages: list[dict],
    previous_summary: str | None,
    summary_prompt_template: str,
) -> str:
    task_config = await Config.get_many(
        'task.model.params',
        'chat.context_compaction.model',
    )
    context_compaction_model = task_config.get('chat.context_compaction.model')
    task_model_id = context_compaction_model if context_compaction_model in models else model_id
    if task_model_id not in models:
        raise ValueError('No available model for context compaction')

    previous_summary = previous_summary if previous_summary and previous_summary.strip() else None
    # A non-empty admin template controls its own output shape, so the built-in
    # heading check must not reject its (potentially heading-less) summaries.
    custom_template = bool((summary_prompt_template or '').strip())
    summary_prompt_template = build_compaction_prompt(
        update=previous_summary is not None,
        legacy=_is_legacy_summary(previous_summary),
        template=summary_prompt_template,
    )
    all_messages = [*compacted_messages, *recent_messages]
    prompt = replace_prompt_variable(summary_prompt_template, get_last_user_message(all_messages) or '')
    prompt = replace_messages_variable(prompt, all_messages)
    prompt = replace_messages_variable(prompt, compacted_messages, 'COMPACTED_MESSAGES')
    prompt = replace_messages_variable(prompt, recent_messages, 'RECENT_MESSAGES')
    prompt = prompt_variables_template(prompt, {'{{PREVIOUS_SUMMARY}}': previous_summary or ''})
    prompt = await prompt_template(prompt, user)

    task_model_params = _summary_task_model_params(task_config.get('task.model.params'), models[task_model_id])

    # Strip the chat/message anchors from the summary request metadata. The
    # metadata merge in ``generate_chat_completion`` lets ``request.state.metadata``
    # win over ``form_data['metadata']``, so removing them from a copy alone is not
    # enough: detach them from the shared request state for the duration of the
    # call (same pattern as ``selected_model_id``), or a task-model request can
    # re-enter the chat path and re-trigger compaction (recursion).
    anchors = ('chat_id', 'user_message_id', 'message_id', 'assistant_message_id')
    state_metadata = getattr(getattr(request, 'state', None), 'metadata', None)
    detached = {}
    if isinstance(state_metadata, dict):
        for key in anchors:
            if key in state_metadata:
                detached[key] = state_metadata.pop(key)

    summary_metadata = dict(state_metadata) if isinstance(state_metadata, dict) else {}
    summary_metadata['task'] = 'context_compaction'

    payload = {
        'model': task_model_id,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'metadata': summary_metadata,
    }

    payload = apply_params_to_form_data(payload, models[task_model_id], task_model_params)
    try:
        return await _generate_summary_with_retry(
            request,
            user,
            payload,
            previous_summary,
            compacted_messages,
            require_headings=not custom_template,
        )
    finally:
        # Restore the shared anchors the caller (and the response pipeline) relies on.
        if isinstance(state_metadata, dict):
            state_metadata.update(detached)


def _response_text(response: Any) -> str:
    if isinstance(response, list) and len(response) == 1:
        response = response[0]

    if isinstance(response, JSONResponse):
        try:
            response = JSONCodec.loads(response.body.decode('utf-8', 'replace'))
        except Exception:
            return ''

    if not isinstance(response, dict):
        return ''

    choices = response.get('choices') or []
    if choices:
        message = choices[0].get('message') or {}
        return message.get('content') or message.get('reasoning_content') or ''

    parts = []
    for item in response.get('output') or []:
        for content in item.get('content') or []:
            if isinstance(content, dict):
                parts.append(content.get('text') or content.get('content') or '')
    return '\n'.join(part for part in parts if part)


def _estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            total += _estimate_tokens(message)
            continue
        total += 4
        content = message.get('content')
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    total += _estimate_tokens(item)
                elif item.get('type') in {'image', 'image_url'}:
                    total += 1000
                else:
                    total += _estimate_tokens(item.get('text') or item.get('content') or item)
        else:
            total += _estimate_tokens(content)

        total += _estimate_tokens(message.get('output'))
        total += _estimate_tokens(message.get('tool_calls'))
        total += _estimate_tokens(message.get('files'))
    return total


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0

    if not isinstance(value, str):
        try:
            value = JSONCodec.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)

    if not value:
        return 0

    cjk = sum(1 for char in value if _is_cjk(char))
    non_cjk = len(value) - cjk
    return max(1, cjk + non_cjk // 4)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana / Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x20000 <= code <= 0x2FA1F  # CJK Extensions B–F / Compatibility Supplement
    )
