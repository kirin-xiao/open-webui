from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import time
from datetime import timedelta
from uuid import uuid4

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from open_webui.internal.db import get_async_db
from open_webui.models.chat_messages import ChatMessages
from open_webui.models.chats import Chat, ChatForm, Chats
from open_webui.models.config import Config
from open_webui.models.users import UserModel, Users
from open_webui.tasks import create_task, has_active_tasks
from open_webui.utils.auth import create_token
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


def _build_request(source: Request, user_id: str, *, internal: bool) -> Request:
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
    token = create_token(
        data={'id': user_id, 'typ': 'subagent'},
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
    await _wait_for_active_tasks(source_request.app.state.redis, parent_chat_id)

    # Imported here (not at module load) to avoid a circular import at startup.
    from open_webui.socket.main import sio

    async with _parent_lock_scope(parent_chat_id):
        if await has_active_tasks(source_request.app.state.redis, parent_chat_id):
            return

        user = await Users.get_user_by_id(user_id)
        if not user:
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
                for meta in [message.get('meta') or {}]
                if message.get('role') == 'user'
                and not message.get('childrenIds')
                and (
                    (
                        meta.get('internal') is True
                        and meta.get('type') == 'subagent'
                        and meta.get('status') in (None, 'pending')
                    )
                    or (meta.get('internal') is True and meta.get('type') == 'timer')
                )
            ]
            if not pending:
                return

            first = pending[0]
            first_meta = first.get('meta') or {}
            kind = 'timer' if first_meta.get('internal') is True and first_meta.get('type') == 'timer' else 'subagent'
            # The enqueue-time anchor. Re-resolved below, before the synthesis turn
            # is built, because the tip may have advanced since this was written.
            stored_parent_id = first.get('parentId')
            if kind == 'timer' and first_meta.get('timer_id'):
                timer = await Chats.get_chat_by_id(first_meta['timer_id'])
                run = {**run, **(((timer.meta or {}).get('run') if timer else None) or {})}
            model_id = first.get('model') or run['model_id']
            if kind == 'timer':
                batch = [first]
            else:
                batch = [
                    message
                    for message in pending
                    for meta in [message.get('meta') or {}]
                    if message.get('parentId') == stored_parent_id
                    and (message.get('model') or model_id) == model_id
                    and (
                        meta.get('internal') is True
                        and meta.get('type') == 'subagent'
                        and meta.get('status') in (None, 'pending')
                    )
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
        request = _build_request(source_request, user.id, internal=True)
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

        # A child that is still producing a turn has an unfinished assistant tip.
        # Continuing it now would race that turn's history writes, so reject until
        # it finishes (the background handle is returned while the child runs).
        tip = existing_messages.get(continuation_tip_id) if continuation_tip_id else None
        if tip and tip.get('role') == 'assistant' and tip.get('done') is False:
            return f'Error: sessionID "{session_id}" is still running; wait for it to finish before continuing.'

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
            child_request = _build_request(request, user.id, internal=True)
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
            message = await Chats.get_message_by_id_and_message_id(chat_id, assistant_message_id)
            if not message:
                return {
                    'status': 'error',
                    'summary': '',
                    'error': 'Subagent chat or completion message no longer exists.',
                }

            summary = _extract_summary(message)
            if len(summary) > max_output:
                summary = f'{summary[:max_output]}\n\n[output truncated]'
            error = message.get('error')
            return {
                'status': 'error' if error else 'completed',
                'summary': summary or ('Subagent produced no output.' if not error else ''),
                'error': error,
            }
        except asyncio.CancelledError:
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                assistant_message_id,
                {'done': True, 'error': {'content': 'Subagent cancelled.'}},
            )
            raise
        except Exception as exc:
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

        pending_message_id = str(uuid4())
        pending_meta = {
            'internal': True,
            'type': 'subagent',
            'source': 'subagent',
            'childID': chat_id,
            'state': state,
            # opencode carries the label through to the parent result so the UI
            # can show it instead of the first line of the output.
            'description': description,
            # Persisted for legacy chats that still key off the old handle.
            'delegation_id': delegation_id,
            'subagent_chat_id': chat_id,
        }
        pending_message = {
            'id': pending_message_id,
            'parentId': None,
            'childrenIds': [],
            'role': 'user',
            'content': envelope,
            'model': run['model_id'],
            'meta': pending_meta,
            'timestamp': int(time.time()),
        }

        async with _parent_lock_scope(parent_chat_id):
            # Read the parent (ownership-checked) just to resolve the anchor. The
            # session is closed before the merge write: the write opens its own
            # session, and holding a `FOR UPDATE` row lock across it would
            # self-deadlock on Postgres.
            async with get_async_db() as db:
                stmt = select(Chat).where(Chat.id == parent_chat_id, Chat.user_id == user.id)
                result_row = await db.execute(stmt)
                parent = result_row.scalar_one_or_none()
                if not parent:
                    if cancelled:
                        raise asyncio.CancelledError
                    return result

                parent_history = (parent.chat or {}).get('history') or {}
                updated_messages = parent_history.get('messages') or {}
                # Anchor the result at the chat's current branch tip, never at the
                # "newest completed assistant". A live turn is persisted with
                # `done: False`, so a timestamp scan skips it and forks the result
                # off an earlier turn — orphaning the user's in-flight message.
                # opencode's promote step appends a background completion at the
                # session tip, so mirror that here. Prefer the denormalized tip
                # column, falling back to the in-JSON tip for older rows.
                result_parent_id = resolve_history_tip(
                    updated_messages,
                    parent.current_message_id or parent_history.get('currentId'),
                    parent_message_id,
                    blocked=is_pending_internal_message,
                )
                pending_message['parentId'] = result_parent_id
                if await has_active_tasks(request.app.state.redis, parent_chat_id):
                    pending_message['meta']['status'] = 'pending'

            # Merge only this node instead of overwriting the whole history blob.
            # `update_chat_by_id` re-reads the row and runs `merge_history`, so a
            # normal turn that wrote since our read is preserved. `currentId` is
            # intentionally omitted: merge_history keeps the stored tip, and this
            # node must not steal it (it may still be `pending`, and the drain
            # advances the tip when the synthesis lands).
            await Chats.update_chat_by_id(
                parent_chat_id,
                {'history': {'messages': {pending_message_id: pending_message}}},
            )

            await ChatMessages.upsert_message(
                message_id=pending_message_id,
                chat_id=parent_chat_id,
                user_id=user.id,
                data=pending_message,
            )

        if pending_message['meta'].get('status') == 'pending':
            from open_webui.socket.main import sio

            await sio.emit(
                'events',
                {
                    'chat_id': parent_chat_id,
                    'message_id': pending_message_id,
                    'data': {'type': 'chat:reload'},
                },
                room=f'user:{user.id}',
            )
        if not await has_active_tasks(request.app.state.redis, parent_chat_id):
            await process_pending_internal_messages(request, parent_chat_id, user.id, run)
        if cancelled:
            raise asyncio.CancelledError
        return result

    try:
        _, child_task = await create_task(
            request.app.state.redis,
            run_background() if background else run_reserved(),
            id=chat_id,
        )
    except Exception as exc:
        return f'Error: {exc}'

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
