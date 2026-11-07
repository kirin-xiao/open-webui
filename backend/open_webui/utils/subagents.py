from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import re
import time
from datetime import timedelta
from uuid import uuid4

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from open_webui.internal.db import get_async_db
from open_webui.models.auths import Auths
from open_webui.models.chat_messages import ChatMessages
from open_webui.models.chats import Chat, ChatForm, Chats
from open_webui.models.config import Config
from open_webui.models.users import UserModel, Users
from open_webui.tasks import create_task, has_active_tasks
from open_webui.utils.auth import VERIFIED_USER_ROLES, create_token
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import get_message_list, is_pending_internal_message, resolve_history_tip
from sqlalchemy import select
from starlette.datastructures import Headers

log = logging.getLogger(__name__)

DEFAULT_SUBAGENT_SYSTEM_PROMPT = """You are a subagent working on a specific task assigned by the lead agent.

You have full access to the workspace — you can read, write, edit files, and run commands.
Focus exclusively on your assigned task. Do NOT work on anything outside your scope.

When done, end with a clear summary:
- What you did
- What files you changed (if any)
- Any issues or open questions
"""

MUTATING_MEMORY_TOOLS = {
    'add_memory',
    'delete_memory',
    'replace_memory_content',
    'update_memory',
}

# Returned by mutating memory tools when a compare-mode fan-out disables them.
# Compare mode runs one concurrent task per model; concurrent read-modify-write
# on the memory store duplicates rows and clobbers replaces (#30238).
COMPARE_MODE_MEMORY_WRITE_DISABLED_MESSAGE = (
    'Error: memory writes are disabled in model compare mode; nothing was saved. '
    'This state persists until the next user message.'
)

# How long `process_pending_internal_messages` waits for sibling tasks to drain
# before giving up. A stale Redis task entry must not hang finalization (W7).
PENDING_ACTIVE_TASK_WAIT_TIMEOUT = 30.0

# The only serializer for the parent-history write. Kept (not concurrency-limit
# machinery) and popped once drained so it does not grow without bound (W3).
_parent_locks: dict[str, asyncio.Lock] = {}


@contextlib.asynccontextmanager
async def _parent_lock_scope(chat_id: str):
    """Serialize parent-history writes for `chat_id`, then drop the lock.

    The lock is dropped only when no task holds it and no task is waiting, so a
    concurrent writer either shares the lock or acquires the next one. The
    check-and-pop is synchronous, so the event loop cannot interleave a new
    waiter between them.
    """
    lock = _parent_locks.setdefault(chat_id, asyncio.Lock())
    try:
        async with lock:
            yield
    finally:
        if not lock.locked() and not getattr(lock, '_waiters', None):
            _parent_locks.pop(chat_id, None)


async def _wait_for_active_tasks(redis, chat_id: str, timeout: float = PENDING_ACTIVE_TASK_WAIT_TIMEOUT) -> None:
    """Bounded drain wait. Returns (instead of hanging) once `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while await has_active_tasks(redis, chat_id):
        if time.monotonic() >= deadline:
            log.warning('Timed out waiting for active tasks on chat %s; proceeding with pending results.', chat_id)
            return
        await asyncio.sleep(0.25)


async def _subagent_chain_depth(chat_id: str) -> int:
    """Count subagent links between `chat_id` and the root chat.

    Walks the persisted internal linkage (`meta.type == 'subagent'` and
    `meta.parent_chat_id`) rather than an in-memory counter, so it is correct
    across processes and restarts. A root chat has depth 0; a chat spawned by
    the root has depth 1.
    """
    depth = 0
    seen: set[str] = set()
    current = chat_id
    while current and current not in seen:
        seen.add(current)
        chat = await Chats.get_chat_by_id(current)
        if not chat:
            break
        meta = chat.meta or {}
        if meta.get('internal') is not True or meta.get('type') != 'subagent':
            break
        parent = meta.get('parent_chat_id')
        if not parent:
            break
        depth += 1
        current = parent
    return depth


def _session_is_live(session_id: str | None, user_id: str) -> bool:
    """True when `session_id` is a live socket session owned by `user_id`.

    The pyodide code interpreter executes in the user's browser through this
    inherited socket session; when the session is gone the event call returns an
    immediate error. Missing Redis/socket wiring is treated as "not live".
    """
    try:
        from open_webui.socket.main import SESSION_POOL

        session = SESSION_POOL.get(session_id) if session_id else None
        return bool(session) and session.get('id') == user_id
    except Exception:
        return False


def _extract_summary(message: dict) -> str:
    """Best-effort final answer for a finished subagent turn.

    Prefers the last output item of type ``message`` (the final round). ``content``
    is only a fallback: it concatenates every tool-loop round, so it leaks interim
    narration ("I've made substantial progress...") ahead of the final answer.
    The whole-content join is the legacy fallback when no message item exists.
    """
    output = message.get('output') or []
    last_message = next((item for item in reversed(output) if item.get('type') == 'message'), None)
    if last_message is not None:
        summary = ''.join(
            str(part.get('text', ''))
            for part in last_message.get('content') or []
            if isinstance(part, dict) and part.get('text') is not None
        )
        if summary:
            return summary

    summary = message.get('content') or ''
    if isinstance(summary, list):
        summary = ''.join(
            str(item.get('text', ''))
            for item in summary
            if isinstance(item, dict) and item.get('type') == 'text'
        )
    if summary:
        return summary

    return ''.join(
        str(part.get('text', ''))
        for item in output
        if item.get('type') == 'message'
        for part in item.get('content') or []
        if isinstance(part, dict) and part.get('text') is not None
    )


async def _build_request(source: Request, user_id: str, *, internal: bool) -> Request:
    scope = {
        'type': 'http',
        'asgi': {'version': '3.0', 'spec_version': '2.0'},
        'method': 'POST',
        'path': '/api/v1/subagents/internal',
        'query_string': b'',
        'headers': Headers({}).raw,
        'client': ('127.0.0.1', 0),
        'server': ('127.0.0.1', 80),
        'scheme': 'http',
        'app': source.app,
    }
    request = Request(scope)
    auth = await Auths.get_auth_by_id(user_id)
    if auth is None or not auth.active:
        raise ValueError('Subagent owner is no longer active')
    token = create_token(
        data={'id': user_id, 'typ': 'subagent', 'session_stamp': auth.session_stamp},
        expires_delta=timedelta(hours=1),
    )
    request.state.token = HTTPAuthorizationCredentials(scheme='Bearer', credentials=token)
    request.state.enable_api_keys = False
    if internal:
        request.state.internal = True
    return request


async def process_pending_internal_messages(
    source_request: Request,
    parent_chat_id: str,
    user_id: str,
    run: dict,
) -> None:
    """Turn-end backstop for durable pending nodes in one chat.

    Drains, in its own turn, whichever shape the pending list starts with:
    timers, child completion notices (``source == 'subagent'``), or
    parent-targeted user steers that no model boundary consumed. Child-targeted
    instructions are never promoted here; they belong to the child chat and are
    drained by that chat's own boundary/turn. Pending nodes left by a different
    kind stay queued for the recursive drain after this synthesis.
    """
    await _wait_for_active_tasks(source_request.app.state.redis, parent_chat_id)

    # Imported here (not at module load) to avoid a circular import at startup.
    from open_webui.socket.main import sio

    async with _parent_lock_scope(parent_chat_id):
        if await has_active_tasks(source_request.app.state.redis, parent_chat_id):
            return

        user = await Users.get_user_by_id(user_id)
        if not user or user.role not in VERIFIED_USER_ROLES:
            return

        async with get_async_db() as db:
            stmt = select(Chat).where(Chat.id == parent_chat_id, Chat.user_id == user_id)
            if db.bind.dialect.name == 'postgresql':
                stmt = stmt.with_for_update()
            result = await db.execute(stmt)
            chat = result.scalar_one_or_none()
            if not chat:
                return

            history = copy.deepcopy((chat.chat or {}).get('history') or {})
            messages = history.get('messages') or {}
            pending = [
                message
                for message in messages.values()
                if _is_pending_completion_message(message) or _is_pending_parent_steer(message)
            ]
            if not pending:
                return

            first = pending[0]
            first_meta = first.get('meta') or {}
            if first_meta.get('internal') is True and first_meta.get('type') == 'timer':
                kind = 'timer'
            elif _is_pending_parent_steer(first):
                # A user steer that never reached a model request (for example,
                # the turn's provider call failed after it was queued) is
                # promoted into a normal turn here. This is the universal
                # turn-end backstop; child-targeted instructions are excluded
                # because they belong to the child chat, not this one.
                kind = 'steer'
            else:
                kind = 'subagent'
            # The enqueue-time anchor. Re-resolved below, before the synthesis turn
            # is built, because the tip may have advanced since this was written.
            stored_parent_id = first.get('parentId')
            if kind == 'timer' and first_meta.get('timer_id'):
                timer = await Chats.get_chat_by_id(first_meta['timer_id'])
                run = {**run, **(((timer.meta or {}).get('run') if timer else None) or {})}
            model_id = first.get('model') or run['model_id']
            if kind == 'timer':
                batch = [first]
            elif kind == 'steer':
                # Batch every parent-targeted steer for this chat/model into one
                # user turn. Completion notices stay pending for their own
                # branch/batch and are never double-delivered here.
                batch = [
                    message
                    for message in pending
                    if _is_pending_parent_steer(message) and (message.get('model') or model_id) == model_id
                ]
            else:
                batch = [
                    message
                    for message in pending
                    for meta in [message.get('meta') or {}]
                    if message.get('parentId') == stored_parent_id
                    and (message.get('model') or model_id) == model_id
                    and _is_child_completion_notice(meta)
                    and meta.get('status') in (None, 'pending')
                ]
            combined_content = '\n\n'.join(message.get('content', '') for message in batch if message.get('content'))
            if kind == 'timer':
                timer_ids = [
                    message['meta']['timer_id'] for message in batch if (message.get('meta') or {}).get('timer_id')
                ]
                combined_meta = {'internal': True, 'type': 'timer'}
                if len(timer_ids) == 1:
                    combined_meta['timer_id'] = timer_ids[0]
                elif timer_ids:
                    combined_meta['timer_ids'] = timer_ids
            elif kind == 'steer':
                # The promoted node becomes the visible user turn: keep the
                # steer identity for the UI, but mark it delivered so the
                # pending drain never sees it again.
                combined_meta = {
                    'internal': True,
                    'type': STEER_TYPE,
                    'source': first_meta.get('source') or 'user',
                    'status': STEER_DELIVERED,
                    # Explicit promotion marker. Readers must treat this node as
                    # a real visible turn, not a model-input-only steer. Older
                    # chats lack the marker and fall back to "has an assistant
                    # child" detection.
                    'promoted': True,
                }
            else:
                delegation_ids = [
                    message['meta']['delegation_id']
                    for message in batch
                    if (message.get('meta') or {}).get('delegation_id')
                ]
                subagent_chat_ids = [
                    message['meta']['subagent_chat_id']
                    for message in batch
                    if (message.get('meta') or {}).get('subagent_chat_id')
                ]
                child_ids = [
                    (message.get('meta') or {}).get('childID') or (message.get('meta') or {}).get('subagent_chat_id')
                    for message in batch
                    if (message.get('meta') or {}).get('childID') or (message.get('meta') or {}).get('subagent_chat_id')
                ]
                states = [message['meta']['state'] for message in batch if (message.get('meta') or {}).get('state')]
                descriptions = [
                    message['meta']['description']
                    for message in batch
                    if (message.get('meta') or {}).get('description')
                ]
                combined_meta = {'internal': True, 'type': 'subagent', 'source': 'subagent'}
                if len(delegation_ids) == 1:
                    combined_meta['delegation_id'] = delegation_ids[0]
                elif delegation_ids:
                    combined_meta['delegation_ids'] = delegation_ids
                if len(subagent_chat_ids) == 1:
                    combined_meta['subagent_chat_id'] = subagent_chat_ids[0]
                elif subagent_chat_ids:
                    combined_meta['subagent_chat_ids'] = subagent_chat_ids
                if len(child_ids) == 1:
                    combined_meta['childID'] = child_ids[0]
                elif child_ids:
                    combined_meta['childIDs'] = child_ids
                if len(states) == 1:
                    combined_meta['state'] = states[0]
                if len(descriptions) == 1:
                    combined_meta['description'] = descriptions[0]

            reuse_message = len(batch) == 1 and (first.get('meta') or {}).get('status') != 'pending'
            user_message_id = first['id'] if reuse_message else str(uuid4())
            removed_ids = set()
            if not reuse_message:
                removed_ids = {message['id'] for message in batch}
                for message_id in removed_ids:
                    messages.pop(message_id, None)
                if stored_parent_id and stored_parent_id in messages:
                    messages[stored_parent_id]['childrenIds'] = [
                        child_id
                        for child_id in messages[stored_parent_id].get('childrenIds', [])
                        if child_id not in removed_ids
                    ]

                # Re-resolve the anchor against the freshly-read history. A result
                # enqueued mid-turn carries the tip from enqueue time; the user's
                # turn may have advanced (or, in the fork bug, the enqueue-time
                # anchor was a completed sibling) since then. Appending at the
                # current tip at *delivery* time mirrors opencode's promote step
                # (its pending completion is promoted at the session tip, never at
                # the spawn point). The batch nodes are already removed above, so
                # the walk cannot land on a to-be-discarded result.
                #
                # TODO(future): carry a background response as a steer message —
                # an admission/queue promoted at the next step boundary — instead
                # of a parentId-anchored history write. This mirrors opencode's
                # `session.synthetic(delivery: "steer")` and would remove the
                # anchor/tree coupling entirely.
                tip_seed = history.get('currentId') if history.get('currentId') in messages else stored_parent_id
                parent_id = resolve_history_tip(
                    messages,
                    tip_seed,
                    stored_parent_id if stored_parent_id in messages else None,
                    blocked=is_pending_internal_message,
                )
            else:
                # A reused (already-delivered, non-pending) result keeps its
                # original anchor; there is nothing to re-resolve.
                parent_id = stored_parent_id

            assistant_message_id = str(uuid4())
            message_list = get_message_list(messages, parent_id)
            system_prompt = run.get('system_prompt')
            user_message = {
                'id': user_message_id,
                'parentId': parent_id,
                'childrenIds': [assistant_message_id],
                'role': 'user',
                'content': combined_content,
                'model': model_id,
                'meta': combined_meta,
                'timestamp': int(time.time()),
            }
            assistant_message = {
                'id': assistant_message_id,
                'parentId': user_message_id,
                'childrenIds': [],
                'role': 'assistant',
                'content': '',
                'done': False,
                'model': model_id,
                'timestamp': int(time.time()),
            }

            if parent_id and parent_id in messages:
                parent_children = [
                    child_id for child_id in messages[parent_id].get('childrenIds', []) if child_id != user_message_id
                ]
                parent_children.append(user_message_id)
                messages[parent_id]['childrenIds'] = parent_children
            messages[user_message_id] = {**messages.get(user_message_id, {}), **user_message}
            messages[assistant_message_id] = assistant_message
            history['messages'] = messages
            history['currentId'] = assistant_message_id
            chat.chat = {**(chat.chat or {}), 'history': history}
            # Keep the denormalized tip column in step with the in-JSON tip. The
            # read path trusts the column when present (and `loadChat` overrides
            # the loaded history with it), so a stale column would hide the
            # synthesis stream until the next save advanced it.
            chat.current_message_id = assistant_message_id
            chat.updated_at = int(time.time())
            await db.commit()

        if removed_ids:
            await ChatMessages.delete_message_ids_by_chat_id(parent_chat_id, removed_ids)
        await ChatMessages.upsert_message(user_message_id, parent_chat_id, user_id, user_message)
        await ChatMessages.upsert_message(assistant_message_id, parent_chat_id, user_id, assistant_message)

        # Build and register the synthesis while still holding the parent lock.
        # `run_background` checks `has_active_tasks` under this same lock before
        # deciding to defer, so registering here (rather than after release)
        # guarantees a child that finishes during this window sees the task and
        # defers (`status: pending`) instead of launching a parallel synthesis.
        form_data = {
            'model': model_id,
            'messages': [
                *([{'role': 'system', 'content': system_prompt}] if system_prompt else []),
                *message_list,
                {'role': 'user', 'content': combined_content},
            ],
            'stream': True,
            'chat_id': parent_chat_id,
            'id': assistant_message_id,
            'parent_id': parent_id,
            'user_message': user_message,
            'session_id': run.get('session_id') or f'{kind}-result:{parent_chat_id}',
            'background_tasks': {},
            'tool_ids': run.get('tool_ids') or [],
            'skill_ids': run.get('skill_ids') or [],
            'filter_ids': run.get('filter_ids') or [],
            'features': run.get('features') or {},
            'files': run.get('files') or [],
            'variables': run.get('variables') or {},
        }
        if run.get('terminal_id'):
            form_data['terminal_id'] = run['terminal_id']

        # The result-synthesis turn is internal: it must not advertise the
        # subagent tool, so a background result can never re-spawn a delegation
        # (W10). Internal runs also skip the pending-message reprocessing hook.
        request = await _build_request(source_request, user.id, internal=True)
        folder_id = await Chats.get_chat_folder_id(parent_chat_id, user.id)

        async def run_synthesis() -> None:
            # Mark the parent chat active before the turn streams so the client
            # keeps the just-reloaded assistant leaf alive instead of
            # force-marking it done. Register before the reload so the client
            # sees the task when it reconciles the reloaded history.
            await sio.emit(
                'events',
                {
                    'chat_id': parent_chat_id,
                    'message_id': assistant_message_id,
                    'data': {
                        'type': 'chat:active',
                        'data': {'active': True, 'folder_id': folder_id},
                    },
                },
                room=f'user:{user.id}',
            )
            await source_request.app.state.CHAT_COMPLETION_HANDLER(request, form_data, user=user)

        _, synthesis_task = await create_task(
            source_request.app.state.redis,
            run_synthesis(),
            id=parent_chat_id,
        )

    # Release the parent-history lock now that the synthesis is registered. The
    # synthesis request is internal, so `CHAT_COMPLETION_HANDLER` would otherwise
    # await the whole turn inline; holding the lock across it would delay a
    # concurrent timer or the next drain. This matches the timer path, which also
    # releases first.
    await sio.emit(
        'events',
        {
            'chat_id': parent_chat_id,
            'message_id': assistant_message_id,
            'data': {'type': 'chat:reload'},
        },
        room=f'user:{user.id}',
    )

    async def _after_synthesis() -> None:
        # Drain any result that deferred while this synthesis was running. The
        # drain waits for this just-finished task to clear via `_wait_for_active_tasks`.
        try:
            await process_pending_internal_messages(source_request, parent_chat_id, user.id, run)
        except Exception:
            log.exception('Failed to drain pending internal messages for chat %s', parent_chat_id)
        if not await has_active_tasks(source_request.app.state.redis, parent_chat_id):
            await sio.emit(
                'events',
                {
                    'chat_id': parent_chat_id,
                    'message_id': assistant_message_id,
                    'data': {'type': 'chat:active', 'data': {'active': False}},
                },
                room=f'user:{user.id}',
            )

    # Do not emit active:false from `run_synthesis`: the task is still registered
    # then. Flip it off only after the drain and once nothing else is active.
    synthesis_task.add_done_callback(lambda _task: asyncio.create_task(_after_synthesis()))


# ---------------------------------------------------------------------------
# Steer records
#
# A steer is a durable pending history node, not a new table or an inbox. It is
# never part of the visible branch while pending: the drain only reads it, and
# `is_pending_internal_message` keeps it from becoming an anchor. The middleware
# appends the text to the *model request* at a turn boundary, then marks the
# record delivered. Long-lived state stays in the existing chat history blob.
# ---------------------------------------------------------------------------

STEER_TYPE = 'steer'
STEER_PENDING = 'pending'
STEER_DELIVERED = 'delivered'
STEER_SUPERSEDED = 'superseded'


def _steer_meta(message: dict | None) -> dict | None:
    """Return a steer node's meta, or None when `message` is not a steer."""
    if not isinstance(message, dict) or message.get('role') != 'user':
        return None
    meta = message.get('meta')
    if not isinstance(meta, dict) or meta.get('internal') is not True or meta.get('type') != STEER_TYPE:
        return None
    return meta


def _is_pending_steer(message: dict | None) -> bool:
    meta = _steer_meta(message)
    if meta is None or message.get('childrenIds'):
        return False
    return meta.get('status') in (None, STEER_PENDING)


_SUBAGENT_ENVELOPE_SESSION = re.compile(r'<subagent\b[^>]*\bsessionID="([^"]+)"')


def _linked_child_id(message: dict, meta: dict) -> str | None:
    """The child a completion notice belongs to, including legacy shapes.

    Current notices carry `childID` and `subagent_chat_id`; very old ones may
    only have the `<subagent sessionID="...">` envelope in their content.
    """
    for key in ('childID', 'subagent_chat_id'):
        value = meta.get(key)
        if value:
            return str(value)
    # A batched notice covers several children. Superseding it for one child
    # would drop the others, so it never gets an envelope-derived single link.
    if meta.get('childIDs') or meta.get('subagent_chat_ids'):
        return None
    content = message.get('content')
    if isinstance(content, str):
        matches = _SUBAGENT_ENVELOPE_SESSION.findall(content)
        if len(matches) == 1:
            return matches[0]
    return None


def _is_child_completion_notice(meta: dict | None) -> bool:
    """True for a node that carries a child's completion into the parent.

    Completion notices are the legacy ``type: subagent`` pending nodes and the
    newer ``type: steer`` synthetic steers (``source: subagent``). Both shape a
    parent synthesis turn; only the steer form is boundary-deliverable.
    """
    if not isinstance(meta, dict) or meta.get('internal') is not True:
        return False
    if meta.get('type') == 'subagent':
        return True
    return meta.get('type') == STEER_TYPE and meta.get('source') == 'subagent'


def _is_pending_completion_message(message: dict | None) -> bool:
    if not isinstance(message, dict) or message.get('role') != 'user' or message.get('childrenIds'):
        return False
    meta = message.get('meta') or {}
    if meta.get('internal') is not True:
        return False
    if meta.get('type') == 'timer':
        return True
    return _is_child_completion_notice(meta) and meta.get('status') in (None, STEER_PENDING)


def _is_pending_parent_steer(message: dict | None) -> bool:
    """A pending steer addressed to *this* chat, for the turn-end backstop.

    This is the user/system steer that normally Slice A consumes at a model
    boundary. It is promoted into a normal turn when no boundary consumed it
    (for example, the provider call failed after it was queued).

    Child-targeted instructions (``childID`` set) are excluded: they belong in
    the child's chat and must never start a turn in the wrong chat. Completion
    notices (``source == 'subagent'``) are excluded too; they have their own
    drain path and must not be double-delivered.
    """
    if not isinstance(message, dict) or message.get('role') != 'user' or message.get('childrenIds'):
        return False
    meta = _steer_meta(message)
    if meta is None or meta.get('status') not in (None, STEER_PENDING):
        return False
    if meta.get('source') == 'subagent' or meta.get('childID'):
        return False
    return True


def _pending_child_completion_notice(messages: dict, child_chat_id: str) -> dict | None:
    """The newest undrained completion notice for ``child_chat_id``, if any."""
    if not messages or not child_chat_id:
        return None
    notices = [
        message
        for message in messages.values()
        if isinstance(message, dict)
        and not message.get('childrenIds')
        and _is_child_completion_notice(message.get('meta') or {})
        and (message.get('meta') or {}).get('status') in (None, STEER_PENDING)
        and _linked_child_id(message, message.get('meta') or {}) == child_chat_id
    ]
    if not notices:
        return None
    return max(notices, key=lambda message: message.get('timestamp', 0))


def _completion_notice_body(notice: dict) -> str:
    content = notice.get('content') or ''
    match = re.match(r'\s*<subagent\b[^>]*>\n?(.*?)\n?</subagent>\s*$', content, re.S)
    return (match.group(1) if match else content).strip()


def _terminal_state_for_tip(tip: dict | None) -> str:
    """Best-effort terminal state for a finished child tip (late-steer race)."""
    error = (tip or {}).get('error') if isinstance(tip, dict) else None
    if not error:
        return 'completed'
    content = error.get('content') if isinstance(error, dict) else str(error)
    if content and 'cancel' in content.lower():
        return 'interrupted'
    return 'error'


def _pre_steer_response(chat_id: str, state: str | None, body: str) -> str:
    """Return a stale final answer with an explicit late-steer note.

    The new turn is already queued in the background; this only tells the parent
    that the text predates the steering instruction.
    """
    note = 'Note: this response was generated before your steering message; a new response is coming.'
    return (
        f'<subagent sessionID="{chat_id}" state="{state or "completed"}" pre_steer="true">\n'
        f'{body}\n\n{note}\n</subagent>'
    )


async def _latest_child_tip_message(chat_id: str, fallback_id: str) -> dict | None:
    """Read the child's current branch tip instead of a spawn-time snapshot.

    A steer or continuation may have attached a newer assistant leaf while the
    same run was finishing. Prefer that leaf so the completion notice describes
    what the child actually ended with.
    """
    chat = await Chats.get_chat_by_id(chat_id)
    messages = ((chat.chat or {}).get('history') or {}).get('messages') or {} if chat else {}
    tip_id = resolve_history_tip(
        messages,
        ((chat.chat or {}).get('history') or {}).get('currentId') if chat else fallback_id,
        fallback_id if fallback_id in messages else None,
        blocked=is_pending_internal_message,
    )
    message = messages.get(tip_id)
    if isinstance(message, dict) and message.get('role') == 'assistant':
        return message
    fallback = messages.get(fallback_id)
    if isinstance(fallback, dict) and fallback.get('role') == 'assistant':
        return fallback
    return await Chats.get_message_by_id_and_message_id(chat_id, fallback_id)


def _stale_child_result_updates(
    messages: dict,
    child_chat_id: str,
    *,
    delegation_id: str | None = None,
    now: int | None = None,
) -> dict[str, dict]:
    """Build status-only updates for superseded results of one child.

    A result node is stale when it is an undelivered (or explicitly pending)
    completion notice for `child_chat_id`. It is marked `superseded` rather than
    removed so a reader can still audit it; the drain ignores anything that is
    not pending, and `is_pending_internal_message` keeps it from anchoring.

    Multi-child (already batched) notices are never touched: they have children
    and/or only plural ids, and superseding one would drop another child's
    result. A legacy notice with no structured child link may still be matched
    by `delegation_id`.
    """
    if not child_chat_id:
        return {}
    timestamp = int(time.time()) if now is None else now
    updates: dict[str, dict] = {}
    for message_id, message in (messages or {}).items():
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        if message.get('childrenIds'):
            continue
        meta = message.get('meta') or {}
        if not _is_child_completion_notice(meta):
            continue
        if meta.get('status') not in (None, STEER_PENDING):
            continue
        linked_child = _linked_child_id(message, meta)
        if linked_child is not None:
            if linked_child != child_chat_id:
                continue
        elif not delegation_id or meta.get('delegation_id') != delegation_id:
            continue
        updates[message_id] = {
            **message,
            'meta': {**meta, 'status': STEER_SUPERSEDED, 'superseded_at': timestamp},
        }
    return updates


async def supersede_pending_child_results(
    parent_chat_id: str,
    user_id: str,
    child_chat_id: str,
    *,
    generation_id: str | None = None,
) -> list[str]:
    """Supersede undrained completion notices for `(parent, child)`.

    Called when a child is continued or when a new completion notice is
    enqueued. Otherwise a notice from the previous generation can be drained
    after the continuation and inject an answer that predates it.

    A continuation also claims the child's new generation here, while holding
    the parent lock: a previous run that finishes after this point sees the
    generation mismatch and skips both its late notice and its terminal state
    write, so the old answer cannot be resurrected after supersession.
    """
    if not parent_chat_id or not user_id or not child_chat_id:
        return []

    async with _parent_lock_scope(parent_chat_id):
        parent = await Chats.get_chat_by_id(parent_chat_id)
        if not parent or parent.user_id != user_id:
            return []
        child = await Chats.get_chat_by_id(child_chat_id)
        child_meta = (child.meta or {}) if child else {}
        owns_child = bool(
            child
            and child.user_id == user_id
            and (not child_meta.get('parent_chat_id') or child_meta.get('parent_chat_id') == parent_chat_id)
        )
        if generation_id and owns_child:
            # Claim first, so an old run's parent-lock section either writes
            # before this supersession (and is marked) or after it (and skips).
            await _begin_child_run(child_chat_id, generation_id)
        messages = ((parent.chat or {}).get('history') or {}).get('messages') or {}
        updates = _stale_child_result_updates(
            messages,
            child_chat_id,
            delegation_id=child_meta.get('delegation_id') if owns_child else None,
        )
        if not updates:
            return []
        await Chats.update_chat_by_id(
            parent_chat_id,
            {'history': {'messages': updates}},
        )
    return list(updates)


def _build_steer_meta(
    *,
    source: str,
    child_id: str | None,
    generation_id: str | None,
    pre_steer: bool,
    state: str | None,
    description: str | None,
    delegation_id: str | None,
) -> dict:
    meta = {
        'internal': True,
        'type': STEER_TYPE,
        'source': source,
        'status': STEER_PENDING,
    }
    if child_id:
        meta['childID'] = child_id
    if generation_id:
        meta['generation_id'] = generation_id
    if pre_steer:
        meta['pre_steer'] = True
    if state:
        meta['state'] = state
    if description:
        meta['description'] = description
    if source == 'subagent' and child_id:
        # Legacy completion nodes carried these two fields; keep them so older
        # readers and supersession paths still match a steer-shaped notice.
        meta['subagent_chat_id'] = child_id
        if delegation_id:
            meta['delegation_id'] = delegation_id
    return meta


async def _steer_generation_allowed(child_id: str | None, expected_generation_id: str | None) -> bool:
    """False when a continuation has replaced the child's run since spawning.

    Called under the parent lock (the same lock a continuation uses to claim the
    next generation), so a stale run cannot enqueue its completion afterwards.
    """
    if expected_generation_id is None:
        return True
    child = await Chats.get_chat_by_id(child_id) if child_id else None
    if child is None:
        return False
    child_generation = (child.meta or {}).get('generation_id')
    # A missing generation on a legacy child is allowed (nothing to compare).
    return child_generation is None or child_generation == expected_generation_id


async def _mark_child_steer_queued(child_id: str) -> None:
    child = await Chats.get_chat_by_id(child_id)
    child_meta = (child.meta or {}) if child else {}
    if child_meta.get('state') in ('running', 'steer_queued', None):
        await _set_child_state(
            child_id,
            'steer_queued',
            generation_id=child_meta.get('generation_id'),
            only_if_state=('running', 'steer_queued', None),
        )


async def enqueue_steer(
    parent_chat_id: str,
    user_id: str,
    content: str,
    *,
    source: str = 'user',
    child_id: str | None = None,
    generation_id: str | None = None,
    pre_steer: bool = False,
    parent_id: str | None = None,
    model_id: str | None = None,
    message_id: str | None = None,
    state: str | None = None,
    description: str | None = None,
    delegation_id: str | None = None,
    expected_generation_id: str | None = None,
    request: Request | None = None,
) -> str | None:
    """Persist a pending steer node and return its id (None on invalid input).

    Write into the chat that will *consume* the steer: the child chat for a
    parent -> child instruction (``source='parent'``, ``childID=<child>``), and
    the parent/target chat for a user steer or a child completion notice
    (``source='subagent'``). The node is anchored at the current branch tip but
    never advances `currentId`, so it stays invisible until a boundary consumes
    it. The optional `request` is accepted for symmetry with the timer/subagent
    enqueue paths; it is not used by the durable write.
    """
    del request  # Reserved for a future socket notification; the write is durable.
    text = (content or '').strip()
    if not text or not parent_chat_id or not user_id:
        return None

    steer_id = message_id or str(uuid4())
    timestamp = int(time.time())
    meta = _build_steer_meta(
        source=source,
        child_id=child_id,
        generation_id=generation_id,
        pre_steer=pre_steer,
        state=state,
        description=description,
        delegation_id=delegation_id,
    )

    message = {
        'id': steer_id,
        'parentId': None,
        'childrenIds': [],
        'role': 'user',
        'content': text,
        'timestamp': timestamp,
        'meta': meta,
    }
    if model_id:
        message['model'] = model_id

    async with _parent_lock_scope(parent_chat_id):
        parent = await Chats.get_chat_by_id(parent_chat_id)
        if not parent or parent.user_id != user_id:
            return None
        if not await _steer_generation_allowed(child_id, expected_generation_id):
            return None
        history = (parent.chat or {}).get('history') or {}
        messages = history.get('messages') or {}
        if steer_id in messages:
            # Idempotent re-enqueue: never reset a delivered/superseded steer
            # (or overwrite an unrelated message that happens to reuse the id).
            return steer_id if _steer_meta(messages[steer_id]) is not None else None
        anchor = resolve_history_tip(
            messages,
            getattr(parent, 'current_message_id', None) or history.get('currentId'),
            parent_id if parent_id in messages else None,
            blocked=is_pending_internal_message,
        )
        message['parentId'] = anchor
        await Chats.update_chat_by_id(
            parent_chat_id,
            {'history': {'messages': {steer_id: message}}},
        )

    await ChatMessages.upsert_message(
        message_id=steer_id,
        chat_id=parent_chat_id,
        user_id=user_id,
        data=message,
    )

    if child_id and source != 'subagent':
        # Only advertise a queued steer while the child is actually active.
        # Re-check under the child lock so a run finishing in this window is
        # not clobbered back to `steer_queued`. Completion notices name their
        # source child but are not instructions for it, so they never touch its
        # state.
        await _mark_child_steer_queued(child_id)

    return steer_id


async def drain_pending_steers(
    parent_chat_id: str,
    user_id: str,
    *,
    child_id: str | None = None,
) -> list[dict]:
    """Read pending steers without consuming them from ``parent_chat_id``.

    Callers append the returned content to their model request, then call
    `mark_steer_delivered` once the request has started. On a failed request
    they leave the records pending for the existing turn-end drain backstop.

    The chat passed here must be the one that holds the records:

    * A parent -> child instruction is stored in the **child** chat and drained
      with ``parent_chat_id=<child id>, child_id=<child id>``.
    * A user steer or child completion notice (``source == 'subagent'``) is
      stored in the **parent/target** chat and drained with its id and
      ``child_id=None``.

    ``child_id=None`` selects parent-targeted records: nodes without a child, a
    steer stored in this chat itself (``childID == parent_chat_id``), and
    completion notices that name their source child but are delivered to the
    parent. Passing a child id selects only parent->child instructions for that
    child, never a completion notice.
    """
    parent = await Chats.get_chat_by_id(parent_chat_id)
    if not parent or parent.user_id != user_id:
        return []
    messages = ((parent.chat or {}).get('history') or {}).get('messages') or {}
    drained: list[dict] = []
    for message in messages.values():
        if not _is_pending_steer(message):
            continue
        meta = message.get('meta') or {}
        target = meta.get('childID')
        if child_id is None:
            if target and target != parent_chat_id and meta.get('source') != 'subagent':
                continue
        else:
            if meta.get('source') == 'subagent' or target != child_id:
                continue
        drained.append(copy.deepcopy(message))
    return drained


def _next_steer_meta(meta: dict, status: str, timestamp: int) -> dict | None:
    """Return the updated meta for a status transition, or None if terminal."""
    current = meta.get('status')
    # Delivered is terminal: a consumed steer must never be delivered again and
    # must not be relabelled as superseded after the model already saw it.
    if status == STEER_DELIVERED and current not in (None, STEER_PENDING):
        return None
    if status == STEER_SUPERSEDED and current in (STEER_DELIVERED, STEER_SUPERSEDED):
        return None
    new_meta = {**meta, 'status': status}
    if status == STEER_DELIVERED:
        new_meta['delivered_at'] = timestamp
    elif status == STEER_SUPERSEDED:
        new_meta['superseded_at'] = timestamp
    return new_meta


async def _mirror_steer_statuses(parent_chat_id: str, user_id: str, messages: list[dict]) -> None:
    """Mirror steer statuses into the normalized message table."""
    for message in messages:
        try:
            await ChatMessages.upsert_message(
                message_id=message['id'],
                chat_id=parent_chat_id,
                user_id=user_id,
                data=message,
            )
        except Exception:
            log.exception('Failed to mirror steer status for %s', message.get('id'))


async def _resume_children_after_steer_delivery(messages: list[dict]) -> None:
    """Resume children paused in `steer_queued` once their steer was consumed.

    `only_if_state` keeps a concurrent completion (or interruption) authoritative.
    Completion notices (``source == 'subagent'``) name the child that produced
    them and must not touch its terminal state.
    """
    for message in messages:
        meta = message.get('meta') or {}
        if meta.get('source') == 'subagent':
            continue
        child_id = meta.get('childID')
        if child_id:
            await _set_child_state(child_id, 'running', only_if_state=('steer_queued',))


async def _update_steer_status(
    parent_chat_id: str,
    user_id: str,
    message_ids: str | list[str] | None,
    status: str,
) -> list[str]:
    if isinstance(message_ids, str):
        message_ids = [message_ids]
    ids = [str(message_id) for message_id in (message_ids or []) if message_id]
    if not ids:
        return []

    updated: list[str] = []
    upserts: list[dict] = []
    async with _parent_lock_scope(parent_chat_id):
        parent = await Chats.get_chat_by_id(parent_chat_id)
        if not parent or parent.user_id != user_id:
            return []
        messages = ((parent.chat or {}).get('history') or {}).get('messages') or {}
        timestamp = int(time.time())
        updates: dict[str, dict] = {}
        for message_id in ids:
            message = messages.get(message_id)
            meta = _steer_meta(message)
            if meta is None:
                continue
            new_meta = _next_steer_meta(meta, status, timestamp)
            if new_meta is None:
                continue
            updates[message_id] = {**message, 'meta': new_meta}
            updated.append(message_id)
        if updates:
            await Chats.update_chat_by_id(
                parent_chat_id,
                {'history': {'messages': updates}},
            )
            upserts = list(updates.values())

    # The history blob is the drain's source of truth, but other readers use
    # chat_message; keep both in step.
    await _mirror_steer_statuses(parent_chat_id, user_id, upserts)
    if status == STEER_DELIVERED:
        await _resume_children_after_steer_delivery(upserts)
    return updated


async def mark_steer_delivered(
    parent_chat_id: str,
    user_id: str,
    message_ids: str | list[str] | None,
) -> list[str]:
    """Mark consumed steers delivered once the model request has started."""
    return await _update_steer_status(parent_chat_id, user_id, message_ids, STEER_DELIVERED)


async def mark_steer_superseded(
    parent_chat_id: str,
    user_id: str,
    message_ids: str | list[str] | None,
) -> list[str]:
    """Mark steers superseded so they can never be delivered."""
    return await _update_steer_status(parent_chat_id, user_id, message_ids, STEER_SUPERSEDED)


async def _begin_child_run(chat_id: str, generation_id: str) -> None:
    """Claim the child's active generation and mark it running (best effort).

    Later writes from an older generation check this value and no-op, so a
    continuation cannot be clobbered by the run it superseded.
    """
    if not chat_id or not generation_id:
        return
    try:
        async with _parent_lock_scope(chat_id):
            await Chats.update_chat_meta_by_id(
                chat_id,
                {'generation_id': generation_id, 'state': 'running'},
            )
    except Exception:
        log.exception('Failed to begin subagent generation %s for chat %s', generation_id, chat_id)


async def _set_child_state(
    chat_id: str,
    state: str,
    *,
    generation_id: str | None = None,
    only_if_state: tuple[str, ...] | None = None,
) -> None:
    """Persist a child chat's lifecycle state in `meta.state` (best effort).

    With `generation_id`, the write is ignored when another run has since
    claimed the child. `only_if_state` additionally guards transitions (for
    example, never overwrite a terminal state with `steer_queued`).
    """
    if not chat_id:
        return
    try:
        if generation_id is not None or only_if_state is not None:
            async with _parent_lock_scope(chat_id):
                child = await Chats.get_chat_by_id(chat_id)
                if not child:
                    return
                meta = child.meta or {}
                if generation_id is not None and meta.get('generation_id') != generation_id:
                    return
                if only_if_state is not None and meta.get('state') not in only_if_state:
                    return
                await Chats.update_chat_meta_by_id(chat_id, {'state': state})
        else:
            await Chats.update_chat_meta_by_id(chat_id, {'state': state})
    except Exception:
        log.exception('Failed to persist subagent state %s for chat %s', state, chat_id)


async def delegate(
    description: str,
    prompt: str,
    context: str,
    background: bool,
    *,
    file_ids: list[str] | None = None,
    session_id: str | None = None,
    model: str | None = None,
    request: Request,
    user_data: dict,
    metadata: dict,
    parent_chat_id: str,
    parent_message_id: str | None,
    tool_call_id: str | None = None,
) -> str:
    prompt = (prompt or '').strip()
    description = (description or '').strip()
    if not prompt:
        return 'Error: prompt must not be empty.'
    if not parent_chat_id or not user_data.get('id'):
        return 'Error: chat and user context are required.'

    config = await Config.get_many(
        'subagents.depth',
        'subagents.model',
        'subagents.max_iterations',
        'subagents.max_output',
        'subagents.system_prompt',
    )
    raw_depth = config.get('subagents.depth')
    depth_limit = 1 if raw_depth is None else int(raw_depth)
    max_iterations = int(config.get('subagents.max_iterations') or 30)
    max_output = int(config.get('subagents.max_output') or 30_000)
    admin_model = str(config.get('subagents.model') or '').strip()

    features = copy.deepcopy(metadata.get('features') or {})

    # Model precedence: explicit per-call > admin default > parent chat model.
    parent_model = metadata.get('model_id') or (metadata.get('model') or {}).get('id')
    requested_model = (model or '').strip()
    available_models = getattr(request.app.state, 'MODELS', None)
    if requested_model and available_models and requested_model not in available_models:
        return f'Error: model "{requested_model}" is not available.'
    effective_model = requested_model or admin_model or parent_model

    run = {
        'model_id': effective_model,
        'session_id': metadata.get('session_id'),
        'tool_ids': copy.deepcopy(metadata.get('tool_ids') or []),
        'skill_ids': copy.deepcopy(metadata.get('skill_ids') or []),
        'system_prompt': metadata.get('system_prompt'),
        # TODO: tool_servers are dropped for background runs because their MCP
        # transport is tied to the caller's live socket session; revisit with the
        # same liveness gating as code_interpreter below.
        'tool_servers': [] if background else copy.deepcopy(metadata.get('tool_servers') or []),
        'filter_ids': copy.deepcopy(metadata.get('filter_ids') or []),
        'terminal_id': metadata.get('terminal_id'),
        'features': features,
        'files': copy.deepcopy(metadata.get('files') or []),
        'variables': copy.deepcopy(metadata.get('variables') or {}),
        'direct': bool(metadata.get('direct')),
    }
    if not run.get('model_id'):
        return 'Error: model context is required.'
    if run.get('direct'):
        return 'Error: subagents are unavailable for direct connections.'

    # Depth cap (opencode `experimental.subagent_depth`, default 1). A negative
    # limit disables the cap. Checked before any child chat is created or reused.
    if depth_limit >= 0:
        depth = await _subagent_chain_depth(parent_chat_id)
        if depth >= depth_limit:
            return (
                f'Error: subagent depth limit reached ({depth_limit}). '
                'Subagents cannot spawn nested subagents; increase subagents.depth to allow it.'
            )

    if file_ids:
        requested_file_ids = {str(file_id) for file_id in file_ids if file_id}
        run['files'] = [
            copy.deepcopy(file)
            for file in metadata.get('files') or []
            if str(file.get('id') or '') in requested_file_ids
            or str(file.get('url') or '') in requested_file_ids
            or (isinstance(file.get('file'), dict) and str(file.get('file', {}).get('id') or '') in requested_file_ids)
        ]
        found_file_ids = {
            str(value)
            for file in run['files']
            for value in (
                file.get('id'),
                file.get('url'),
                file.get('file', {}).get('id') if isinstance(file.get('file'), dict) else None,
            )
            if value
        }
        missing_file_ids = sorted(requested_file_ids - found_file_ids)
        if missing_file_ids:
            return f'Error: file_ids not attached or unavailable: {", ".join(missing_file_ids)}'
    else:
        run['files'] = []

    user = UserModel(**user_data)
    # One token per child run. A continuation claims a fresh one before it
    # supersedes the previous result, so a late writer from the old run can
    # recognise that it no longer owns the child.
    generation_id = str(uuid4())

    # pyodide executes in the user's browser via the inherited socket session;
    # when the session is gone the event call returns an immediate error and the
    # subagent continues without code, so dropping it is safe. A live, owned
    # session keeps it. (Jupyter runs server-side, independent of the session.)
    # TODO: bound concurrent code-exec-capable background children / consider
    # keying the pyodide runtime by parent chat id.
    if (
        background
        and features.get('code_interpreter')
        and await Config.get('code_interpreter.engine', 'pyodide') != 'jupyter'
        and not _session_is_live(run.get('session_id'), user.id)
    ):
        features.pop('code_interpreter')

    delegation_id = f'deleg_{uuid4().hex[:8]}'
    prompt_text = f'{prompt}\n\n## Context\n{context}' if context else prompt

    # Continuation: reuse a child chat owned by this user whose internal meta
    # links it to this caller. Anything else is rejected rather than silently
    # creating a new chat.
    existing_chat = None
    existing_messages: dict = {}
    continuation_tip_id = None
    late_body: str | None = None
    late_state: str | None = None
    if session_id:
        existing_chat = await Chats.get_chat_by_id(session_id)
        if not existing_chat or existing_chat.user_id != user.id:
            return f'Error: sessionID "{session_id}" was not found.'
        existing_meta = existing_chat.meta or {}
        if existing_meta.get('internal') is not True or existing_meta.get('type') != 'subagent':
            return f'Error: sessionID "{session_id}" is not a subagent session.'
        if existing_meta.get('parent_chat_id') != parent_chat_id:
            return f'Error: sessionID "{session_id}" does not belong to this chat.'

        existing_messages = copy.deepcopy((existing_chat.chat or {}).get('history', {}).get('messages') or {})
        continuation_tip_id = (existing_chat.chat or {}).get('history', {}).get('currentId')
        if continuation_tip_id not in existing_messages:
            continuation_tip_id = None
        if continuation_tip_id is None and existing_messages:
            continuation_tip_id = max(existing_messages.values(), key=lambda message: message.get('timestamp', 0)).get(
                'id'
            )

        # A child that is still producing a turn is steered, not continued: the
        # instruction is queued durably and the child's middleware consumes it at
        # its next request boundary. Continuing here would race the live turn's
        # history writes.
        tip = existing_messages.get(continuation_tip_id) if continuation_tip_id else None
        child_running = bool(
            (tip and tip.get('role') == 'assistant' and tip.get('done') is False)
            or await has_active_tasks(request.app.state.redis, existing_chat.id)
        )
        if child_running:
            # TODO(late-steer boundary race): if the child has already generated
            # its final response but its turn has not finalized, Slice A may have
            # no further model request to consume this steer. It then stays
            # pending until the next child turn. The acceptance-time race below
            # is handled, but the clean fix for this narrower window is a child
            # turn-end sweep that starts a new turn (with the pre_steer note).
            # The record lives in the child chat: that is the chat whose
            # boundary drain (child_id == existing_chat.id) consumes it.
            steer_id = await enqueue_steer(
                existing_chat.id,
                user.id,
                prompt_text,
                source='parent',
                child_id=existing_chat.id,
                generation_id=existing_meta.get('generation_id'),
                model_id=run['model_id'],
            )
            if steer_id is None:
                return f'Error: could not queue a steering message for sessionID "{session_id}".'

            # The run may have finished between the activity check and the write.
            # If so the queued steer has no live consumer: tombstone it and fall
            # through to the normal continuation/late-steer path instead of
            # leaving the child stranded in `steer_queued`.
            refreshed = await Chats.get_chat_by_id(existing_chat.id)
            refreshed_history = ((refreshed.chat or {}).get('history') or {}) if refreshed else {}
            refreshed_messages = refreshed_history.get('messages') or {}
            refreshed_tip_id = resolve_history_tip(
                refreshed_messages,
                refreshed_history.get('currentId') or continuation_tip_id,
                continuation_tip_id,
                blocked=is_pending_internal_message,
            )
            refreshed_tip = refreshed_messages.get(refreshed_tip_id)
            still_running = bool(
                (refreshed_tip and refreshed_tip.get('role') == 'assistant' and refreshed_tip.get('done') is False)
                or await has_active_tasks(request.app.state.redis, existing_chat.id)
            )
            if still_running:
                return JSONCodec.dumps(
                    {
                        'sessionID': existing_chat.id,
                        'status': 'steer_queued',
                        'output': (
                            f'The steering message was queued for the running subagent '
                            f'(sessionID: {existing_chat.id}). It will be applied at the next safe '
                            'boundary. Do NOT poll, sleep, or duplicate its work.'
                        ),
                    },
                    ensure_ascii=False,
                )

            await mark_steer_superseded(existing_chat.id, user.id, steer_id)
            if refreshed and (refreshed.meta or {}).get('state') == 'steer_queued':
                await _set_child_state(
                    existing_chat.id,
                    _terminal_state_for_tip(refreshed_tip),
                    only_if_state=('steer_queued',),
                )
            # Continue from the refreshed branch, not the pre-write snapshot.
            existing_messages = copy.deepcopy(refreshed_messages)
            continuation_tip_id = refreshed_tip_id

        # Detect a late steer: the child finished after the parent decided to send
        # an instruction, and its completion notice is still undrained. The old
        # answer is returned now (with a pre_steer note) while the new turn runs.
        #
        # TODO(late-steer race): an intentional foreground continuation of an
        # idle child is indistinguishable from a late foreground steer, so we
        # only take the note path for `background=True` (the shape a parent uses
        # when it expects an immediate handle). Foreground calls keep the existing
        # continuation semantics and never return a stale answer. A dedicated
        # tool-level mode is the clean fix if this ambiguity matters.
        late_notice = None
        if background:
            parent_chat = await Chats.get_chat_by_id(parent_chat_id)
            if parent_chat and parent_chat.user_id == user.id:
                parent_messages = ((parent_chat.chat or {}).get('history') or {}).get('messages') or {}
                late_notice = _pending_child_completion_notice(parent_messages, existing_chat.id)
        late_body = _completion_notice_body(late_notice) if late_notice else None
        late_state = (late_notice.get('meta') or {}).get('state') if late_notice else None

        # A completion notice from the previous generation may still be waiting
        # in the parent's history. Continuing the child invalidates it: draining
        # it later would inject an answer that predates this turn. Claim the new
        # generation and supersede the old notice atomically, before the new
        # generation writes anything.
        await supersede_pending_child_results(
            parent_chat_id,
            user.id,
            existing_chat.id,
            generation_id=generation_id,
        )

        # A continued turn keeps the child's workspace unless the caller selects
        # files explicitly: it inherits the files on the child's most recent user
        # message, otherwise the per-call `file_ids` selection above applies.
        if not file_ids:
            latest_user_with_files = max(
                (
                    message
                    for message in existing_messages.values()
                    if message.get('role') == 'user' and message.get('files')
                ),
                key=lambda message: message.get('timestamp', 0),
                default=None,
            )
            if latest_user_with_files:
                run['files'] = copy.deepcopy(latest_user_with_files.get('files') or [])

    prompt_files = copy.deepcopy(run.get('files') or [])
    child_message_list: list[dict] = []

    try:
        if existing_chat is not None:
            chat_id = existing_chat.id
            user_message_id = str(uuid4())
            assistant_message_id = str(uuid4())
            user_message = {
                'id': user_message_id,
                'parentId': continuation_tip_id,
                'childrenIds': [assistant_message_id],
                'role': 'user',
                'content': prompt_text,
                'timestamp': int(time.time()),
                'models': [run['model_id']],
                **({'files': prompt_files} if prompt_files else {}),
            }
            assistant_message = {
                'id': assistant_message_id,
                'parentId': user_message_id,
                'childrenIds': [],
                'role': 'assistant',
                'content': '',
                'done': False,
                'model': run['model_id'],
                'timestamp': int(time.time()),
            }
            # Maintain parentId/childrenIds/currentId through the model helper.
            await Chats.upsert_message_to_chat_by_id_and_message_id(chat_id, user_message_id, user_message)
            await Chats.upsert_message_to_chat_by_id_and_message_id(chat_id, assistant_message_id, assistant_message)
            child_message_list = get_message_list(existing_messages, continuation_tip_id)
        else:
            chat_id = str(uuid4())
            user_message_id = str(uuid4())
            assistant_message_id = str(uuid4())
            user_message = {
                'id': user_message_id,
                'parentId': None,
                'childrenIds': [assistant_message_id],
                'role': 'user',
                'content': prompt_text,
                'timestamp': int(time.time()),
                'models': [run['model_id']],
                **({'files': prompt_files} if prompt_files else {}),
            }
            chat = await Chats.insert_new_chat(
                chat_id,
                user.id,
                ChatForm(
                    chat={
                        'id': chat_id,
                        'title': f'Subagent: {(description or prompt)[:60]}',
                        'models': [run['model_id']],
                        'history': {
                            'currentId': assistant_message_id,
                            'messages': {
                                user_message_id: user_message,
                                assistant_message_id: {
                                    'id': assistant_message_id,
                                    'parentId': user_message_id,
                                    'childrenIds': [],
                                    'role': 'assistant',
                                    'content': '',
                                    'done': False,
                                    'model': run['model_id'],
                                    'timestamp': int(time.time()),
                                },
                            },
                        },
                        'messages': [
                            {
                                'role': 'user',
                                'content': prompt_text,
                                **({'files': prompt_files} if prompt_files else {}),
                            }
                        ],
                        'files': prompt_files,
                    }
                ),
                internal_meta={
                    'internal': True,
                    'type': 'subagent',
                    'parent_chat_id': parent_chat_id,
                    'parent_message_id': parent_message_id,
                    'delegation_id': delegation_id,
                    'description': description,
                    'model': run['model_id'],
                    'mode': 'background' if background else 'foreground',
                    **({'tool_call_id': tool_call_id} if tool_call_id else {}),
                },
            )
            if not chat:
                raise RuntimeError('Failed to create subagent chat')
    except Exception as exc:
        # A continuation already claimed the child generation above; make sure a
        # write failure does not leave it stuck in `running`.
        await _set_child_state(chat_id, 'error', generation_id=generation_id)
        prefix = 'background ' if background else ''
        return f'Error: failed to create {prefix}subagent: {exc}'

    # Continuations reuse the existing child chat, so seed the call linkage on its
    # meta as well; the creation path already carries it via `internal_meta`. This
    # is the durable seed a later reader can use to rebuild the call -> child map.
    if existing_chat is not None and tool_call_id:
        await Chats.update_chat_meta_by_id(chat_id, {'tool_call_id': tool_call_id})

    # Tell the parent chat which child this exact tool call spawned, so its row can
    # link into the child while the run is still in progress (the `<subagent ...>`
    # result envelope only exists once the child finishes). Best-effort: a failed
    # emit must never break the delegation.
    if tool_call_id:
        try:
            from open_webui.socket.main import get_event_emitter

            event_emitter = await get_event_emitter(metadata)
            if event_emitter:
                await event_emitter(
                    {
                        'type': 'subagent:created',
                        'data': {
                            'call_id': tool_call_id,
                            'sessionID': chat_id,
                            'message_id': metadata.get('message_id'),
                            'description': description,
                            'background': background,
                        },
                    }
                )
        except Exception:
            log.exception('Failed to emit subagent:created for child %s', chat_id)

    async def run_reserved() -> dict:
        try:
            child_request = await _build_request(request, user.id, internal=True)
            # W5: `-1` means unlimited, matching CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS.
            child_request.state.max_tool_call_iterations = None if max_iterations == -1 else max_iterations
            parent_system_prompt = run.get('system_prompt') or ''
            subagent_system_prompt = (
                str(config.get('subagents.system_prompt') or '').strip() or DEFAULT_SUBAGENT_SYSTEM_PROMPT
            )
            form_data = {
                'model': run['model_id'],
                'messages': [
                    {
                        'role': 'system',
                        'content': (
                            f'{parent_system_prompt}\n\n{subagent_system_prompt}'
                            if parent_system_prompt
                            else subagent_system_prompt
                        ),
                    },
                    *child_message_list,
                    {
                        'role': 'user',
                        'content': prompt_text,
                        **({'files': prompt_files} if prompt_files else {}),
                    },
                ],
                'stream': True,
                'chat_id': chat_id,
                'id': assistant_message_id,
                'parent_id': user_message.get('parentId'),
                'user_message': user_message,
                'session_id': run.get('session_id') or f'subagent:{chat_id}',
                'background_tasks': {},
                'tool_ids': run.get('tool_ids') or [],
                'skill_ids': run.get('skill_ids') or [],
                'filter_ids': run.get('filter_ids') or [],
                'features': run.get('features') or {},
                'files': run.get('files') or [],
                'variables': run.get('variables') or {},
            }
            if run.get('terminal_id'):
                form_data['terminal_id'] = run['terminal_id']
            if run.get('tool_servers'):
                form_data['tool_servers'] = run['tool_servers']
            await request.app.state.CHAT_COMPLETION_HANDLER(child_request, form_data, user=user)
            # Prefer the child's current tip over the spawn-time assistant id: a
            # steer/continuation completed during this run may have attached a
            # newer final answer.
            message = await _latest_child_tip_message(chat_id, assistant_message_id)
            if not message:
                await _set_child_state(chat_id, 'error', generation_id=generation_id)
                return {
                    'status': 'error',
                    'summary': '',
                    'error': 'Subagent chat or completion message no longer exists.',
                }

            summary = _extract_summary(message)
            if len(summary) > max_output:
                summary = f'{summary[:max_output]}\n\n[output truncated]'
            error = message.get('error')
            await _set_child_state(chat_id, 'error' if error else 'completed', generation_id=generation_id)
            return {
                'status': 'error' if error else 'completed',
                'summary': summary or ('Subagent produced no output.' if not error else ''),
                'error': error,
            }
        except asyncio.CancelledError:
            await _set_child_state(chat_id, 'interrupted', generation_id=generation_id)
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                assistant_message_id,
                {'done': True, 'error': {'content': 'Subagent cancelled.'}},
            )
            raise
        except Exception as exc:
            await _set_child_state(chat_id, 'error', generation_id=generation_id)
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                assistant_message_id,
                {'done': True, 'error': {'content': str(exc)}},
            )
            raise

    async def run_background() -> dict:
        cancelled = False
        try:
            result = await run_reserved()
        except asyncio.CancelledError:
            result = {'status': 'interrupted', 'summary': '', 'error': 'cancelled'}
            cancelled = True
        except Exception as exc:
            result = {'status': 'error', 'summary': '', 'error': str(exc)}

        state = result.get('status') or 'completed'
        # `run_reserved` already writes the terminal state, but a task cancelled
        # before its first await never reaches that code. Make the background
        # wrapper authoritative so the child cannot stay `running` forever.
        # A generation check keeps a superseded run from clobbering the
        # continuation that has already taken over the child.
        await _set_child_state(chat_id, state, generation_id=generation_id)
        summary = result.get('summary') or ''
        if state == 'interrupted':
            body = 'The subagent was interrupted before completing.'
            if summary:
                body = f'{body}\n\nPartial output:\n{summary}'
        elif state == 'error':
            detail = f' {result.get("error")}' if result.get('error') else ''
            body = f'The subagent did not complete successfully.{detail}'
            if summary:
                body = f'{body}\n\nPartial output:\n{summary}'
        else:
            body = summary or 'Subagent completed without a final summary.'
        envelope = f'<subagent sessionID="{chat_id}" state="{state}">\n{body}\n</subagent>'

        # Completion is delivered as a synthetic steer: it rides the same durable
        # pending-node path as user/parent steers, so the parent boundary drain
        # (or the turn-end drain when the parent is idle) can deliver it. The
        # deterministic id keeps one notice per child generation.
        notice_id = f'subagent-completion-{chat_id}-{generation_id}'

        async with _parent_lock_scope(parent_chat_id):
            # Read the parent (ownership-checked) to confirm it still exists and
            # that no continuation has claimed the child while this run finished.
            async with get_async_db() as db:
                stmt = select(Chat).where(Chat.id == parent_chat_id, Chat.user_id == user.id)
                result_row = await db.execute(stmt)
                parent = result_row.scalar_one_or_none()
                if not parent:
                    if cancelled:
                        raise asyncio.CancelledError
                    return result

                # A continuation may have claimed a newer generation while this
                # run was finishing. Its old answer is stale: do not enqueue a
                # notice that could be drained before the new answer lands.
                child = await Chats.get_chat_by_id(chat_id)
                child_meta = (child.meta or {}) if child else {}
                child_generation = child_meta.get('generation_id')
                if child is not None and child_generation is not None and child_generation != generation_id:
                    if cancelled:
                        raise asyncio.CancelledError
                    return result

        # Replace any undrained notice from an earlier generation before this one
        # lands. `enqueue_steer` re-checks the generation under the parent lock,
        # closing the window against a continuation that starts here.
        await supersede_pending_child_results(parent_chat_id, user.id, chat_id)
        steer_id = await enqueue_steer(
            parent_chat_id,
            user.id,
            envelope,
            source='subagent',
            child_id=chat_id,
            generation_id=generation_id,
            model_id=run['model_id'],
            message_id=notice_id,
            state=state,
            description=description,
            delegation_id=delegation_id,
            parent_id=parent_message_id,
            expected_generation_id=generation_id,
        )
        if steer_id is None:
            # Parent vanished or a continuation took over the child; the answer
            # this run produced must not be delivered.
            if cancelled:
                raise asyncio.CancelledError
            return result

        parent_active = await has_active_tasks(request.app.state.redis, parent_chat_id)
        if parent_active:
            from open_webui.socket.main import sio

            await sio.emit(
                'events',
                {
                    'chat_id': parent_chat_id,
                    'message_id': steer_id,
                    'data': {'type': 'chat:reload'},
                },
                room=f'user:{user.id}',
            )
        if not parent_active:
            await process_pending_internal_messages(request, parent_chat_id, user.id, run)
        if cancelled:
            raise asyncio.CancelledError
        return result

    # Claim a fresh generation and mark `running` before registering the task so
    # the caller's handle (and the list endpoint) reflects the transition
    # immediately. The same token guards the task's terminal writes.
    await _begin_child_run(chat_id, generation_id)
    # A late steer returns the old final answer immediately, so the new turn must
    # run in the background even when the caller did not ask for background.
    run_in_background = background or late_body is not None
    try:
        _, child_task = await create_task(
            request.app.state.redis,
            run_background() if run_in_background else run_reserved(),
            id=chat_id,
        )
    except Exception as exc:
        await _set_child_state(chat_id, 'error', generation_id=generation_id)
        return f'Error: {exc}'

    if child_task is not None and hasattr(child_task, 'add_done_callback'):

        def _finalize_cancelled(task) -> None:
            # Covers a task cancelled before its first await: the coroutine never
            # reached its own CancelledError handler, so finalize it here.
            if task.cancelled():
                asyncio.create_task(_set_child_state(chat_id, 'interrupted', generation_id=generation_id))

        child_task.add_done_callback(_finalize_cancelled)

    if late_body is not None:
        return _pre_steer_response(chat_id, late_state, late_body)

    if background:
        return JSONCodec.dumps(
            {
                'sessionID': chat_id,
                'status': 'running',
                'output': '\n\n'.join(
                    [
                        f'The subagent is working in the background (sessionID: {chat_id}). '
                        'You will be notified automatically when it finishes.',
                        'DO NOT sleep, poll for progress, ask the subagent for status, or duplicate this '
                        "subagent's work; avoid working with the same files or topics it is using.",
                        'Work on non-overlapping tasks, or briefly tell the user what you launched and end your '
                        'response.',
                    ]
                ),
            },
            ensure_ascii=False,
        )

    try:
        result = await child_task
    except asyncio.CancelledError:
        if asyncio.current_task() and asyncio.current_task().cancelling():
            raise
        return 'Error: subagent was cancelled.'
    except Exception as exc:
        return f'Error: {exc}'

    if result.get('status') != 'completed':
        return f'Error: {result.get("error") or "subagent failed."}'
    summary = result.get('summary') or 'Subagent produced no output.'
    return f'<subagent sessionID="{chat_id}" state="completed">\n{summary}\n</subagent>'
