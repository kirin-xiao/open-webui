"""Config-form tests for `POST /configs/subagents` (part of T1).

The endpoint gained optional ``SUBAGENTS_DEPTH`` / ``SUBAGENTS_MODEL`` fields.
These tests pin the compatibility contract: a payload that omits them must not
422 and must not clobber stored values, while a payload that supplies ``model``
must persist it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_webui.routers import configs


class RecordingConfig:
    def __init__(self, store: dict | None = None):
        self.store = dict(store or {})
        self.upserts: list[dict] = []

    async def upsert(self, updates: dict):
        self.upserts.append(dict(updates))
        self.store.update(updates)

    async def get_many(self, *keys):
        return {key: self.store[key] for key in keys if key in self.store}


def _base_payload(**overrides) -> dict:
    payload = {
        'ENABLE_SUBAGENTS': True,
        'SUBAGENTS_MAX_ITERATIONS': 30,
        'SUBAGENTS_MAX_OUTPUT': 30000,
        'SUBAGENTS_SYSTEM_PROMPT': '',
    }
    payload.update(overrides)
    return payload


async def _post(monkeypatch, payload, store=None):
    recorder = RecordingConfig(store)
    published: list[dict] = []

    async def _publish(request, event, **kwargs):
        published.append({'event': event, **kwargs})

    monkeypatch.setattr(configs, 'Config', recorder)
    monkeypatch.setattr(configs, 'publish_event', _publish)

    form = configs.SubagentsConfigForm(**payload)
    result = await configs.set_subagents_config(
        SimpleNamespace(app=None), form, user=SimpleNamespace(id='admin', role='admin')
    )
    return recorder, published, result


def test_form_defaults_depth_and_model_without_422():
    form = configs.SubagentsConfigForm(**_base_payload())
    assert form.SUBAGENTS_DEPTH == 1
    assert form.SUBAGENTS_MODEL == ''
    # The defaults were not explicitly supplied, so they must be excluded on save.
    dumped = form.model_dump(exclude_unset=True)
    assert 'SUBAGENTS_DEPTH' not in dumped
    assert 'SUBAGENTS_MODEL' not in dumped


@pytest.mark.asyncio
async def test_omitting_depth_and_model_does_not_clobber(monkeypatch):
    recorder, published, result = await _post(
        monkeypatch,
        _base_payload(),
        store={'subagents.depth': 5, 'subagents.model': 'stored-model'},
    )

    assert recorder.upserts, 'expected an upsert'
    saved = recorder.upserts[-1]
    assert 'subagents.depth' not in saved
    assert 'subagents.model' not in saved
    # Pre-existing values survive.
    assert recorder.store['subagents.depth'] == 5
    assert recorder.store['subagents.model'] == 'stored-model'
    assert result['ENABLE_SUBAGENTS'] is True
    assert published[-1]['subject_id'] == 'subagents'


@pytest.mark.asyncio
async def test_supplying_model_persists_it(monkeypatch):
    recorder, _, _ = await _post(
        monkeypatch,
        _base_payload(SUBAGENTS_MODEL='chosen-model', SUBAGENTS_DEPTH=3),
    )

    saved = recorder.upserts[-1]
    assert saved['subagents.model'] == 'chosen-model'
    assert saved['subagents.depth'] == 3
    assert recorder.store['subagents.model'] == 'chosen-model'


@pytest.mark.asyncio
async def test_config_updates_maps_only_known_fields(monkeypatch):
    updates = configs.config_updates(
        {'ENABLE_SUBAGENTS': True, 'UNKNOWN_FIELD': 'ignored', 'SUBAGENTS_MODEL': 'm'},
        configs.SUBAGENTS_CONFIG_KEYS,
    )
    assert updates == {'subagents.enable': True, 'subagents.model': 'm'}
