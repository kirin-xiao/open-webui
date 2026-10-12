/**
 * Helpers for the structured context-compaction checkpoint record.
 *
 * The authoring backend stores a checkpoint as JSON on the boundary message's
 * `contextSummary`/`context_summary` field:
 *
 *   { version, summary, recent, dropped_ids, model, tokens, created_at, status }
 *
 * Chats compacted before the rewrite carry a legacy bare summary string, so the
 * parser tolerates both shapes. This mirrors the backend `_parse_checkpoint`
 * read shim so the ring/timeline never disagree about whether a checkpoint
 * exists.
 */

export type ContextCheckpointRecord = {
	version?: number;
	summary: string;
	recent?: string;
	dropped_ids?: string[];
	model?: string | null;
	tokens?: number | null;
	created_at?: number | null;
	status?: string;
	legacy?: boolean;
	error?: string;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

export const parseContextCheckpoint = (value: unknown): ContextCheckpointRecord | null => {
	if (value === null || value === undefined) {
		return null;
	}

	if (typeof value === 'string') {
		const text = value.trim();
		if (!text) {
			return null;
		}

		if (text[0] === '{' || text[0] === '[') {
			let parsed: unknown;
			try {
				parsed = JSON.parse(text);
			} catch {
				parsed = undefined;
			}

			if (isRecord(parsed)) {
				// A JSON object is a structured record. A missing or empty summary is
				// treated as *absent* (matching the backend), so it must never fall
				// through to the legacy branch and display the raw JSON as a summary.
				return typeof parsed.summary === 'string' && parsed.summary.trim()
					? { version: 1, status: 'completed', ...(parsed as ContextCheckpointRecord) }
					: null;
			}
			// Arrays, primitives and invalid JSON are not a structured record; fall
			// through to the legacy bare-string shape (same as the backend).
		}

		return { version: 0, summary: text, status: 'completed', legacy: true };
	}

	if (isRecord(value)) {
		if (typeof value.summary === 'string' && value.summary.trim()) {
			return { version: 1, status: 'completed', ...(value as ContextCheckpointRecord) };
		}
	}

	return null;
};

/** Read the checkpoint record off a chat message, if any. */
export const getMessageCheckpoint = (message: unknown): ContextCheckpointRecord | null => {
	if (!isRecord(message)) {
		return null;
	}
	// Mirror the backend field selection (`contextSummary or context_summary`): an
	// empty camelCase value must fall through to the snake_case field so both sides
	// agree about whether a checkpoint exists.
	return parseContextCheckpoint(message.contextSummary || message.context_summary);
};
