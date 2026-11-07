/**
 * Helpers for rendering durable steering records and reconciling subagent
 * status. A steer is a pending internal history node (never part of the live
 * branch while pending); a consumed steer is persisted as `meta.steers` on the
 * assistant message and/or the same node with `status: "delivered"`.
 *
 * Everything here is tolerant of the legacy/current shapes so old chats keep
 * rendering exactly as before:
 * - child state: `meta.state` (Slice 0) or legacy `meta.status`
 * - steer annotation: strings or `{ content, source, pre_steer }` objects
 */

export type SteerStatus = 'pending' | 'delivered' | 'superseded';

export type SteerEntry = {
	id: string;
	content: string;
	source: string;
	status: SteerStatus;
	preSteer: boolean;
	childID?: string;
	timestamp?: number;
};

type HistoryMessage = {
	id?: unknown;
	role?: unknown;
	content?: unknown;
	childrenIds?: unknown;
	timestamp?: unknown;
	meta?: Record<string, unknown> | null;
};

type HistoryMessages = Record<string, HistoryMessage> | null | undefined;

// Child states written by the subagent layer. Anything not explicitly active
// is treated as terminal so a missed socket event cannot pin a spinner forever.
// `completed` / `error` / `interrupted` are the terminal states Slice 0 writes.
// Anything else non-active is also terminal so future states do not pin a row.
const ACTIVE_SUBAGENT_STATES = new Set(['running', 'pending', 'steer_queued']);

function normalizeState(state: unknown): string {
	return typeof state === 'string' ? state.trim().toLowerCase() : '';
}

/** True for a terminal child state (including unknown non-empty states). */
export function isTerminalSubagentState(state: unknown): boolean {
	const value = normalizeState(state);
	return !!value && !ACTIVE_SUBAGENT_STATES.has(value);
}

/** True while a child still runs (or has an undelivered steer waiting). */
export function isActiveSubagentState(state: unknown): boolean {
	return ACTIVE_SUBAGENT_STATES.has(normalizeState(state));
}

/**
 * Child ids reported terminal by `GET /chats/{id}/subagents`.
 * Accepts a Map or a plain object so callers/tests can use either.
 */
export function subagentStatesToCompleted(
	states: Map<string, string> | Record<string, string> | null | undefined
): Set<string> {
	const completed = new Set<string>();
	if (!states) return completed;

	const entries: Array<[string, string]> =
		states instanceof Map
			? [...states.entries()].map(([id, state]) => [id, String(state ?? '')])
			: Object.entries(states).map(([id, state]) => [id, String(state ?? '')]);

	for (const [id, state] of entries) {
		if (!id || !normalizeState(state)) continue;
		if (isTerminalSubagentState(state)) completed.add(id);
	}
	return completed;
}

/**
 * Merge the history-derived completion set with persisted child states.
 *
 * Persisted state wins for ids it knows about: a new generation that is
 * `running` must keep spinning even if an older completion result still exists
 * in history, and a terminal state must stop a row whose completion event was
 * missed.
 */
export function mergeSubagentCompletionStates(
	completedIds: Set<string> | null | undefined,
	states: Map<string, string> | Record<string, string> | null | undefined
): Set<string> {
	const completed = new Set(completedIds ?? []);
	if (!states) return completed;

	const entries: Array<[string, string]> =
		states instanceof Map
			? [...states.entries()]
			: Object.entries(states).map(([id, state]) => [id, String(state ?? '')]);

	for (const [id, state] of entries) {
		if (!id || !normalizeState(state)) continue;
		if (isTerminalSubagentState(state)) {
			completed.add(id);
		} else if (isActiveSubagentState(state)) {
			completed.delete(id);
		}
	}
	return completed;
}

function isSteerMeta(meta: unknown): meta is Record<string, unknown> {
	return (
		!!meta &&
		typeof meta === 'object' &&
		(meta as Record<string, unknown>).internal === true &&
		(meta as Record<string, unknown>).type === 'steer'
	);
}

/** True for a persisted steer history node. */
export function isSteerMessage(message: HistoryMessage | null | undefined): boolean {
	if (!message || message.role !== 'user' || !isSteerMeta(message.meta)) return false;
	return typeof message.content === 'string' && message.content.trim().length > 0;
}

/**
 * Remove steer nodes from every message's `childrenIds`.
 *
 * The durable enqueue anchors a steer at the current tip, and the backend
 * rebuilds `childrenIds` from `parentId`. That would make a steer a sibling
 * branch of the next user turn, exposing an undelivered steer as a normal user
 * bubble through the branch controls. Steers are model-input only, so the
 * client prunes them from the visible tree after every load/merge.
 *
 * Returns true when anything changed (so callers can reassign the history).
 */
export function pruneSteerBranches(messages: HistoryMessages): boolean {
	if (!messages) return false;
	let changed = false;
	for (const message of Object.values(messages)) {
		if (!message || !Array.isArray(message.childrenIds)) continue;
		const children = message.childrenIds as unknown[];
		const kept = children.filter((childId) => {
			if (typeof childId !== 'string') return true;
			const child = messages[childId];
			return !(child && child.role === 'user' && isSteerMeta(child.meta));
		});
		if (kept.length !== children.length) {
			message.childrenIds = kept;
			changed = true;
		}
	}
	return changed;
}

function steerStatus(meta: Record<string, unknown>): SteerStatus {
	const status = typeof meta.status === 'string' ? meta.status : '';
	if (status === 'delivered') return 'delivered';
	if (status === 'superseded') return 'superseded';
	return 'pending';
}

function toSteerEntry(message: HistoryMessage, fallbackId: string): SteerEntry | null {
	if (!isSteerMessage(message)) return null;
	const meta = message.meta ?? {};
	const content = String(message.content ?? '').trim();
	if (!content) return null;
	return {
		id: typeof message.id === 'string' && message.id ? message.id : fallbackId,
		content,
		source: typeof meta.source === 'string' ? meta.source : 'user',
		status: steerStatus(meta),
		preSteer: meta.pre_steer === true,
		...(typeof meta.childID === 'string' && meta.childID ? { childID: meta.childID } : {}),
		...(typeof message.timestamp === 'number' ? { timestamp: message.timestamp } : {})
	};
}

/**
 * Persisted steer nodes, oldest first. Superseded nodes are excluded by
 * default because they were never delivered and are audit-only.
 */
export function collectSteerNodes(
	messages: HistoryMessages,
	{ includeSuperseded = false }: { includeSuperseded?: boolean } = {}
): SteerEntry[] {
	const entries: SteerEntry[] = [];
	for (const [id, message] of Object.entries(messages ?? {})) {
		const entry = toSteerEntry(message, id);
		if (!entry) continue;
		if (!includeSuperseded && entry.status === 'superseded') continue;
		entries.push(entry);
	}
	return entries.sort((a, b) => {
		const at = a.timestamp ?? 0;
		const bt = b.timestamp ?? 0;
		return at === bt ? a.id.localeCompare(b.id) : at - bt;
	});
}

/**
 * User steers waiting for the next model boundary, rendered as queued chips.
 * Child-targeted steers belong to the child's run, not the parent transcript.
 */
export function collectPendingSteers(messages: HistoryMessages): SteerEntry[] {
	return collectSteerNodes(messages).filter(
		(entry) => entry.status === 'pending' && !entry.childID
	);
}

function annotationContent(value: unknown): string {
	if (typeof value === 'string') return value.trim();
	if (!value || typeof value !== 'object' || Array.isArray(value)) return '';
	const record = value as Record<string, unknown>;
	const content = record.content ?? record.text ?? record.message;
	return typeof content === 'string' ? content.trim() : '';
}

function annotationId(value: unknown): string {
	if (!value || typeof value !== 'object' || Array.isArray(value)) return '';
	const record = value as Record<string, unknown>;
	const id = record.id ?? record.message_id;
	return typeof id === 'string' ? id : '';
}

/** Normalize an assistant message's `meta.steers` annotation. */
export function steerEntriesFromAnnotation(value: unknown): SteerEntry[] {
	if (!value) return [];
	const items =
		typeof value === 'string'
			? [value]
			: Array.isArray(value)
				? value
				: typeof value === 'object'
					? Object.values(value as Record<string, unknown>)
					: [];

	const entries: SteerEntry[] = [];
	items.forEach((item, index) => {
		const content = annotationContent(item);
		if (!content) return;
		const record =
			item && typeof item === 'object' && !Array.isArray(item)
				? (item as Record<string, unknown>)
				: {};
		entries.push({
			id: annotationId(item) || `annotated-steer-${index}`,
			content,
			source: typeof record.source === 'string' ? record.source : 'user',
			status: 'delivered',
			preSteer: record.pre_steer === true
		});
	});
	return entries;
}

/** All steer texts annotated on assistant messages in this history. */
function annotatedSteers(messages: HistoryMessages): { texts: Set<string>; ids: Set<string> } {
	const texts = new Set<string>();
	const ids = new Set<string>();
	for (const message of Object.values(messages ?? {})) {
		if (message?.role !== 'assistant') continue;
		for (const entry of steerEntriesFromAnnotation(message.meta?.steers)) {
			texts.add(entry.content);
			if (!entry.id.startsWith('annotated-steer-')) ids.add(entry.id);
		}
	}
	return { texts, ids };
}

/**
 * Delivered steer nodes that are not yet reflected by an assistant
 * `meta.steers` annotation. Used as the backward-compatible fallback when the
 * middleware has not annotated the response (or the chat predates Slice A).
 */
export function collectUnannotatedDeliveredSteers(messages: HistoryMessages): SteerEntry[] {
	const annotated = annotatedSteers(messages);
	return collectSteerNodes(messages).filter(
		(entry) =>
			entry.status === 'delivered' &&
			!entry.childID &&
			!annotated.ids.has(entry.id) &&
			!annotated.texts.has(entry.content)
	);
}

/** True when an assistant message should warn that it predates a steer. */
export function hasPreSteerNote(meta: Record<string, unknown> | null | undefined): boolean {
	if (!meta) return false;
	if (meta.pre_steer === true) return true;
	return steerEntriesFromAnnotation(meta.steers).some((entry) => entry.preSteer);
}

export type BusySubmitPlan = 'steer' | 'queue' | 'interrupt' | 'submit';

/**
 * Decide what to do with a user submission while a turn is already active.
 *
 * `steerOutcome` is undefined until the steer endpoint has been tried; after it
 * runs, `queued` means the message is handled, `inactive` means the server no
 * longer has an active task (so the caller may submit normally after
 * reconciling stale state), and `failed` falls back to queue/interrupt.
 *
 * Attachments have no steer representation, so they never use the steer path.
 */
export function planBusySubmit({
	delivery,
	hasFiles,
	enableMessageQueue,
	steerOutcome
}: {
	delivery: 'steer' | 'queue';
	hasFiles: boolean;
	enableMessageQueue: boolean;
	steerOutcome?: 'queued' | 'inactive' | 'failed';
}): BusySubmitPlan {
	if (steerOutcome === 'queued') return 'steer';
	if (steerOutcome === 'inactive') return 'submit';
	if (steerOutcome === undefined && delivery === 'steer' && !hasFiles) return 'steer';
	if (delivery === 'queue' || enableMessageQueue) return 'queue';
	return 'interrupt';
}
