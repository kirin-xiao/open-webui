"""Tests for the user-steer chat endpoint (Slice C).

The endpoint must be ownership-checked, mark the steer as a user steer in the
durable pending-node store, and never start a second completion turn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.routers.chats import SteerChatForm, steer_chat_by_id


class FakeChats:
    def __init__(self):
        self.chat = SimpleNamespace(id='chat-1', user_id='user-1')

    async def get_chat_by_id_and_user_id(self, id, user_id, db=None):
        if id == 'chat-1' and user_id == 'user-1':
            return self.chat
        return None


def make_request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))


def make_user(user_id='user-1'):
    return SimpleNamespace(id=user_id, role='user')


@pytest.mark.asyncio
async def test_steer_endpoint_queues_user_steer_without_starting_a_turn(monkeypatch):
    import open_webui.routers.chats as chats_router

    calls: list[tuple] = []
    active_checks: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        active_checks.append((redis, chat_id))
        return True

    async def fake_enqueue_steer(chat_id, user_id, content, **kwargs):
        calls.append((chat_id, user_id, content, kwargs))
        return 'steer-1'

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='  inspect the failing test  ', model_id='model-1'),
        user=make_user(),
        db=None,
    )

    assert response.queued is True
    assert response.id == 'steer-1'
    assert active_checks == [(None, 'chat-1')]
    # Exactly one durable enqueue; no task id, completion, or second turn.
    assert calls == [
        (
            'chat-1',
            'user-1',
            'inspect the failing test',
            {'source': 'user', 'model_id': 'model-1'},
        )
    ]


@pytest.mark.asyncio
async def test_steer_endpoint_falls_back_when_no_active_task(monkeypatch):
    import open_webui.routers.chats as chats_router

    enqueued: list[tuple] = []

    async def fake_has_active_tasks(redis, chat_id):
        return False

    async def fake_enqueue_steer(*args, **kwargs):
        enqueued.append((args, kwargs))
        return 'steer-1'

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())
    monkeypatch.setattr(chats_router, 'has_active_tasks', fake_has_active_tasks)
    monkeypatch.setattr(chats_router, 'enqueue_steer', fake_enqueue_steer)

    response = await steer_chat_by_id(
        make_request(),
        'chat-1',
        SteerChatForm(content='too late'),
        user=make_user(),
        db=None,
    )

    assert response.queued is False
    assert response.reason == 'no_active_task'
    assert enqueued == []


@pytest.mark.asyncio
async def test_steer_endpoint_rejects_non_owner(monkeypatch):
    import open_webui.routers.chats as chats_router

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())

    with pytest.raises(HTTPException) as exc:
        await steer_chat_by_id(
            make_request(),
            'chat-1',
            SteerChatForm(content='steer'),
            user=make_user('other-user'),
            db=None,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_steer_endpoint_requires_content(monkeypatch):
    import open_webui.routers.chats as chats_router

    monkeypatch.setattr(chats_router, 'Chats', FakeChats())

    with pytest.raises(HTTPException) as exc:
        await steer_chat_by_id(
            make_request(),
            'chat-1',
            SteerChatForm(content='   '),
            user=make_user(),
            db=None,
        )

    assert exc.value.status_code == 400
