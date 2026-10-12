/**
 * Presentation helper for harness-injection visibility (P1-3).
 *
 * When a turn emits a tail injection, the backend publishes a lightweight
 * `status` event whose data is `{ action, count, done }`. Memory deltas, memory
 * edits, and retrieval blocks each get their own `action`. The event never
 * carries the frozen injection text, so this module only renders *that* it
 * happened and roughly how much.
 *
 * The `t()` calls in `injectionStatusTranslation` use literal keys on purpose:
 * `i18next-parser` only understands literal translation calls, so a purely
 * dynamic map would be deleted by `npm run i18n:parse`.
 */

export type InjectionStatusAction =
	| 'memory_context_updated'
	| 'memory_context_update'
	| 'knowledge_retrieved';

export type InjectionStatusLike = {
	action?: string | null;
	count?: number | null;
};

/** Translation keys this module can emit, keyed by backend status action. */
export const INJECTION_STATUS_KEYS: Record<InjectionStatusAction, string> = {
	memory_context_updated: 'Memory context updated ({{count}} entries)',
	memory_context_update: 'Memory context updated ({{count}} changes)',
	knowledge_retrieved: 'Retrieved from knowledge base ({{count}} chunks)'
};

export function isInjectionStatusAction(action: unknown): action is InjectionStatusAction {
	return typeof action === 'string' && action in INJECTION_STATUS_KEYS;
}

/** Every translation key this module can emit. Used by tests and the catalogue. */
export function injectionStatusKeys(): string[] {
	return Object.values(INJECTION_STATUS_KEYS);
}

/** Coerce a status event's count to a non-negative display value. */
export function injectionStatusCount(status: InjectionStatusLike | null | undefined): number {
	const count = status?.count;
	return typeof count === 'number' && Number.isFinite(count) && count > 0 ? Math.floor(count) : 0;
}

/**
 * Resolve the collapsed one-liner for an injection status, or `null` when the
 * status is not one of ours (so callers can fall through to existing status
 * rendering). `i18n` is passed in to keep the module free of i18next imports.
 */
export function injectionStatusTranslation(
	status: InjectionStatusLike | null | undefined,
	i18n: { t: (key: string, values?: Record<string, unknown>) => string }
): string | null {
	const action = status?.action;
	if (!isInjectionStatusAction(action)) return null;

	const values = { count: injectionStatusCount(status) };
	const t = i18n.t.bind(i18n);

	switch (action) {
		case 'memory_context_updated':
			return t('Memory context updated ({{count}} entries)', values);
		case 'memory_context_update':
			return t('Memory context updated ({{count}} changes)', values);
		case 'knowledge_retrieved':
			return t('Retrieved from knowledge base ({{count}} chunks)', values);
	}
}
