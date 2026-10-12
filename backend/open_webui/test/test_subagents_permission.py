"""Permission-gate tests for `features.subagents` (W4 / T2).

Exercises the real registration path in ``utils/tools.get_builtin_tools`` with the
real ``access_control.has_permission`` semantics (group permissions combined with
``DEFAULT_USER_PERMISSIONS``), so the "default enabled" invariant is checked
against the shipped defaults rather than a stub.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import open_webui.utils.tools as tools_module
from open_webui.config import DEFAULT_USER_PERMISSIONS
from open_webui.utils import access_control

SUBAGENT_TOOLS = {'subagent', 'timer'}


class FakeConfig:
    def __init__(self, values: dict):
        self.values = values

    async def get_many(self, *keys):
        return {key: self.values.get(key) for key in keys}

    async def get(self, key, default=None):
        return self.values.get(key, default)


def _request(*, internal=False, direct=False):
    return SimpleNamespace(state=SimpleNamespace(internal=internal, direct=direct))


async def _get_tools(
    monkeypatch,
    *,
    role='user',
    subagents_permission=True,
    group_permissions=None,
    global_enable=True,
    builtin_enabled=True,
    internal=False,
    direct=False,
):
    async def _fake_groups(user_id, db=None):
        return [] if group_permissions is None else [SimpleNamespace(permissions=group_permissions)]

    monkeypatch.setattr(access_control.Groups, 'get_groups_by_member_id', _fake_groups)
    # Use the production permission matcher so defaults + group combination are real.
    monkeypatch.setattr(tools_module, 'has_permission', access_control.has_permission)

    defaults = copy.deepcopy(DEFAULT_USER_PERMISSIONS)
    defaults['features']['subagents'] = subagents_permission
    monkeypatch.setattr(
        tools_module,
        'Config',
        FakeConfig({'subagents.enable': global_enable, 'user.permissions': defaults}),
    )

    model = {'info': {'meta': {'builtinTools': {'subagents': builtin_enabled}, 'capabilities': {}}}}
    extra_params = {'__user__': {'id': 'u1', 'role': role}, '__metadata__': {'chat_id': 'c1'}}
    return await tools_module.get_builtin_tools(
        _request(internal=internal, direct=direct), extra_params, features={}, model=model
    )


def test_features_subagents_defaults_to_enabled():
    """Upgrading deployments must not silently lose the tool (reviewer invariant)."""
    assert DEFAULT_USER_PERMISSIONS['features']['subagents'] is True


@pytest.mark.asyncio
async def test_admin_always_gets_subagent_tools(monkeypatch):
    builtins = await _get_tools(monkeypatch, role='admin', subagents_permission=False)
    assert SUBAGENT_TOOLS <= set(builtins)


@pytest.mark.asyncio
async def test_non_admin_without_permission_denied(monkeypatch):
    builtins = await _get_tools(monkeypatch, role='user', subagents_permission=False)
    assert not (SUBAGENT_TOOLS & set(builtins))


@pytest.mark.asyncio
async def test_non_admin_default_enabled_grants(monkeypatch):
    builtins = await _get_tools(monkeypatch, role='user', subagents_permission=True)
    assert SUBAGENT_TOOLS <= set(builtins)


@pytest.mark.asyncio
async def test_group_permission_grants_when_default_denies(monkeypatch):
    builtins = await _get_tools(
        monkeypatch,
        role='user',
        subagents_permission=False,
        group_permissions={'features': {'subagents': True}},
    )
    assert SUBAGENT_TOOLS <= set(builtins)


@pytest.mark.asyncio
async def test_group_denial_does_not_override_enabled_default(monkeypatch):
    """A group that sets False must not remove a default-enabled feature."""
    builtins = await _get_tools(
        monkeypatch,
        role='user',
        subagents_permission=True,
        group_permissions={'features': {'subagents': False}},
    )
    assert SUBAGENT_TOOLS <= set(builtins)


@pytest.mark.asyncio
async def test_global_switch_suppresses_even_for_admin(monkeypatch):
    builtins = await _get_tools(monkeypatch, role='admin', global_enable=False)
    assert not (SUBAGENT_TOOLS & set(builtins))


@pytest.mark.asyncio
async def test_internal_and_direct_requests_get_no_subagent_tools(monkeypatch):
    internal = await _get_tools(monkeypatch, role='admin', internal=True)
    direct = await _get_tools(monkeypatch, role='admin', direct=True)
    assert not (SUBAGENT_TOOLS & set(internal))
    assert not (SUBAGENT_TOOLS & set(direct))


@pytest.mark.asyncio
async def test_model_builtin_tool_toggle_suppresses(monkeypatch):
    builtins = await _get_tools(monkeypatch, role='admin', builtin_enabled=False)
    assert not (SUBAGENT_TOOLS & set(builtins))


@pytest.mark.asyncio
async def test_denied_call_is_tool_not_found(monkeypatch):
    """A call for a tool the gate removed fails as a normal unknown tool."""
    builtins = await _get_tools(monkeypatch, role='user', subagents_permission=False)
    assert 'subagent' not in builtins

    from open_webui.test.test_subagents import _legacy_call, _load_middleware_namespace

    ns = _load_middleware_namespace()
    ns['tools'] = builtins
    ns['metadata'] = {}
    ns['form_data'] = {}
    ns['seen_subagent_sessions'] = set()

    async def _never(*a, **k):  # pragma: no cover
        raise AssertionError('no callable should run')

    ns['get_updated_tool_function'] = _never
    _, result, tool, _, _ = await ns['execute_tool_call'](_legacy_call('delegate_task', {'task': 'x'}))
    assert result == 'Error: Tool "delegate_task" not found.'
    assert tool is None
