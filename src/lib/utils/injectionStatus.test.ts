import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
	INJECTION_STATUS_KEYS,
	injectionStatusCount,
	injectionStatusKeys,
	injectionStatusTranslation,
	isInjectionStatusAction
} from './injectionStatus';

/** Minimal `t()` that interpolates `{{count}}`, mirroring i18next with en-US keys. */
const stubI18n = {
	t: (key: string, values: Record<string, unknown> = {}) =>
		key.replace(/\{\{(\w+)\}\}/g, (_, token: string) => String(values[token] ?? ''))
};

function localeKeys(locale: string): Set<string> {
	const path = fileURLToPath(
		new URL(`../i18n/locales/${locale}/translation.json`, import.meta.url)
	);
	return new Set(Object.keys(JSON.parse(readFileSync(path, 'utf-8'))));
}

describe('isInjectionStatusAction', () => {
	it('accepts only known injection actions', () => {
		expect(isInjectionStatusAction('memory_context_updated')).toBe(true);
		expect(isInjectionStatusAction('memory_context_update')).toBe(true);
		expect(isInjectionStatusAction('knowledge_retrieved')).toBe(true);

		expect(isInjectionStatusAction('web_search')).toBe(false);
		expect(isInjectionStatusAction('sources_retrieved')).toBe(false);
		expect(isInjectionStatusAction(undefined)).toBe(false);
		expect(isInjectionStatusAction(null)).toBe(false);
	});
});

describe('injectionStatusCount', () => {
	it('clamps junk counts to a non-negative integer', () => {
		expect(injectionStatusCount({ count: 3 })).toBe(3);
		expect(injectionStatusCount({ count: 2.9 })).toBe(2);
		expect(injectionStatusCount({ count: 0 })).toBe(0);
		expect(injectionStatusCount({ count: -5 })).toBe(0);
		expect(injectionStatusCount({ count: Number.NaN })).toBe(0);
		expect(injectionStatusCount({})).toBe(0);
		expect(injectionStatusCount(null)).toBe(0);
	});
});

describe('injectionStatusTranslation', () => {
	it('renders a one-liner for each injection action', () => {
		expect(
			injectionStatusTranslation({ action: 'memory_context_updated', count: 2 }, stubI18n)
		).toBe('Memory context updated (2 entries)');
		expect(
			injectionStatusTranslation({ action: 'memory_context_update', count: 1 }, stubI18n)
		).toBe('Memory context updated (1 changes)');
		expect(injectionStatusTranslation({ action: 'knowledge_retrieved', count: 3 }, stubI18n)).toBe(
			'Retrieved from knowledge base (3 chunks)'
		);
	});

	it('returns null for statuses that are not injections', () => {
		expect(injectionStatusTranslation({ action: 'web_search', count: 2 }, stubI18n)).toBeNull();
		expect(injectionStatusTranslation(null, stubI18n)).toBeNull();
	});

	it('never carries the injection text, only the kind and count', () => {
		// The descriptor is intentionally tiny: adding source text would defeat the
		// privacy point of surfacing only *that* an injection happened.
		const text = injectionStatusTranslation(
			{ action: 'knowledge_retrieved', count: 3, text: 'SECRET MEMORY' } as never,
			stubI18n
		);
		expect(text).not.toContain('SECRET');
	});
});

describe('i18n catalogue', () => {
	it('has every emitted key present in en-US and zh-CN', () => {
		const emitted = injectionStatusKeys();
		expect(emitted).toEqual(Object.values(INJECTION_STATUS_KEYS));
		expect(emitted.length).toBeGreaterThan(0);

		for (const locale of ['en-US', 'zh-CN']) {
			const available = localeKeys(locale);
			const missing = emitted.filter((key) => !available.has(key));
			expect(missing, `${locale} is missing ${missing.join(', ')}`).toEqual([]);
		}
	});
});
