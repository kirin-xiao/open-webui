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
from open_webui.utils.misc import get_message_list
from sqlalchemy import select
from starlette.datastructures import Headers

log = logging.getLogger(__name__)

DEFAULT_SUBAGENT_SYSTEM_PROMPT = """You are a sub-agent working on a specific task assigned by the lead agent.

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
    """Count sub-agent links between `chat_id` and the root chat.

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
            parent_id = first.get('parentId')
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
                    if message.get('parentId') == parent_id
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
                if parent_id and parent_id in messages:
                    messages[parent_id]['childrenIds'] = [
                        child_id
                        for child_id in messages[parent_id].get('childrenIds', [])
                        if child_id not in removed_ids
                    ]

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
            chat.updated_at = int(time.time())
            await db.commit()

        if removed_ids:
            await ChatMessages.delete_message_ids_by_chat_id(parent_chat_id, removed_ids)
        await ChatMessages.upsert_message(user_message_id, parent_chat_id, user_id, user_message)
        await ChatMessages.upsert_message(assistant_message_id, parent_chat_id, user_id, assistant_message)

    # Release the parent-history lock before the synthesis turn. The synthesis
    # request is internal, so `CHAT_COMPLETION_HANDLER` awaits the whole turn
    # inline; holding the lock across it would delay a concurrent timer or the
    # next drain. This matches the timer path, which also releases first.
    from open_webui.socket.main import sio

    await sio.emit(
        'events',
        {
            'chat_id': parent_chat_id,
            'message_id': assistant_message_id,
            'data': {'type': 'chat:reload'},
        },
        room=f'user:{user.id}',
    )

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
    # sub-agent tool, so a background result can never re-spawn a delegation
    # (W10). Internal runs also skip the pending-message reprocessing hook.
    request = _build_request(source_request, user.id, internal=True)
    await source_request.app.state.CHAT_COMPLETION_HANDLER(request, form_data, user=user)


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
    if (
        background
        and features.get('code_interpreter')
        and await Config.get('code_interpreter.engine', 'pyodide') != 'jupyter'
    ):
        features.pop('code_interpreter')

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
        return 'Error: sub-agents are unavailable for direct connections.'

    # Depth cap (opencode `experimental.subagent_depth`, default 1). A negative
    # limit disables the cap. Checked before any child chat is created or reused.
    if depth_limit >= 0:
        depth = await _subagent_chain_depth(parent_chat_id)
        if depth >= depth_limit:
            return (
                f'Error: sub-agent depth limit reached ({depth_limit}). '
                'Sub-agents cannot spawn nested sub-agents; increase subagents.depth to allow it.'
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
            return f'Error: sessionID "{session_id}" is not a sub-agent session.'
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
                        'title': f'Sub-agent: {(description or prompt)[:60]}',
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
                },
            )
            if not chat:
                raise RuntimeError('Failed to create sub-agent chat')
    except Exception as exc:
        prefix = 'background ' if background else ''
        return f'Error: failed to create {prefix}sub-agent: {exc}'

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
                    'error': 'Sub-agent chat or completion message no longer exists.',
                }

            summary = message.get('content') or ''
            if isinstance(summary, list):
                summary = ''.join(
                    str(item.get('text', ''))
                    for item in summary
                    if isinstance(item, dict) and item.get('type') == 'text'
                )
            if not summary:
                summary = ''.join(
                    str(part.get('text', ''))
                    for item in message.get('output') or []
                    if item.get('type') == 'message'
                    for part in item.get('content') or []
                    if part.get('type') == 'output_text'
                )
            if len(summary) > max_output:
                summary = f'{summary[:max_output]}\n\n[output truncated]'
            error = message.get('error')
            return {
                'status': 'error' if error else 'completed',
                'summary': summary or ('Sub-agent produced no output.' if not error else ''),
                'error': error,
            }
        except asyncio.CancelledError:
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                assistant_message_id,
                {'done': True, 'error': {'content': 'Sub-agent cancelled.'}},
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
            body = 'The sub-agent was interrupted before completing.'
            if summary:
                body = f'{body}\n\nPartial output:\n{summary}'
        elif state == 'error':
            detail = f' {result.get("error")}' if result.get('error') else ''
            body = f'The sub-agent did not complete successfully.{detail}'
            if summary:
                body = f'{body}\n\nPartial output:\n{summary}'
        else:
            body = summary or 'Sub-agent completed without a final summary.'
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
            async with get_async_db() as db:
                stmt = select(Chat).where(Chat.id == parent_chat_id, Chat.user_id == user.id)
                if db.bind.dialect.name == 'postgresql':
                    stmt = stmt.with_for_update()
                result_row = await db.execute(stmt)
                parent = result_row.scalar_one_or_none()
                if not parent:
                    if cancelled:
                        raise asyncio.CancelledError
                    return result

                updated_chat = copy.deepcopy(parent.chat or {})
                updated_history = updated_chat.setdefault('history', {})
                updated_messages = updated_history.setdefault('messages', {})
                done_assistants = [
                    message
                    for message in updated_messages.values()
                    if message.get('role') == 'assistant' and message.get('done') is not False
                ]
                result_parent_id = (
                    max(done_assistants, key=lambda message: message.get('timestamp', 0)).get('id')
                    if done_assistants
                    else parent_message_id
                )
                pending_message['parentId'] = result_parent_id
                if await has_active_tasks(request.app.state.redis, parent_chat_id):
                    pending_message['meta']['status'] = 'pending'
                updated_messages[pending_message_id] = pending_message
                if result_parent_id and result_parent_id in updated_messages:
                    children = updated_messages[result_parent_id].setdefault('childrenIds', [])
                    if pending_message_id not in children:
                        children.append(pending_message_id)
                updated_history['messages'] = updated_messages
                parent.chat = {**(parent.chat or {}), **updated_chat, 'history': updated_history}
                parent.updated_at = int(time.time())
                await db.commit()

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
                'output': (
                    f'The sub-agent is working in the background (sessionID: {chat_id}). '
                    'You will be notified when it finishes. DO NOT sleep, poll, or duplicate its work.'
                ),
            },
            ensure_ascii=False,
        )

    try:
        result = await child_task
    except asyncio.CancelledError:
        if asyncio.current_task() and asyncio.current_task().cancelling():
            raise
        return 'Error: sub-agent was cancelled.'
    except Exception as exc:
        return f'Error: {exc}'

    if result.get('status') != 'completed':
        return f'Error: {result.get("error") or "sub-agent failed."}'
    summary = result.get('summary') or 'Sub-agent produced no output.'
    return f'<subagent sessionID="{chat_id}" state="completed">\n{summary}\n</subagent>'
