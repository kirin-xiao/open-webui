import { describe, expect, it } from 'vitest';

import { getMessageCheckpoint, parseContextCheckpoint } from './contextCompaction';

/**
 * Parity tests for the frontend read shim against the backend `_parse_checkpoint`
 * (``backend/open_webui/utils/context_compaction.py``). The two must agree about
 * whether a checkpoint exists, otherwise the context ring and timeline disagree
 * with the request-assembly truncation.
 */
describe('parseContextCheckpoint', () => {
	it('reads the structured JSON record', () => {
		const record = parseContextCheckpoint(
			JSON.stringify({ version: 1, summary: '## Objective\n- ship it', status: 'completed' })
		);

		expect(record?.summary).toBe('## Objective\n- ship it');
		expect(record?.legacy).not.toBe(true);
	});

	it('reads a legacy bare summary string', () => {
		const record = parseContextCheckpoint('a legacy summary');

		expect(record?.summary).toBe('a legacy summary');
		expect(record?.legacy).toBe(true);
	});

	it('reads an already-parsed structured record', () => {
		expect(parseContextCheckpoint({ summary: 'structured' })?.summary).toBe('structured');
	});

	it('treats an empty summary as absent for both shapes', () => {
		expect(parseContextCheckpoint('')).toBeNull();
		expect(parseContextCheckpoint('   ')).toBeNull();
		expect(parseContextCheckpoint({ summary: '' })).toBeNull();
		expect(parseContextCheckpoint({ summary: '   ' })).toBeNull();
	});

	it('treats a JSON object without a valid summary as absent (never legacy)', () => {
		// Backend `_parse_checkpoint` returns None here; the frontend must not fall
		// back to the legacy branch and render the raw JSON as a summary.
		expect(parseContextCheckpoint('{}')).toBeNull();
		expect(parseContextCheckpoint('{"version":1}')).toBeNull();
		expect(parseContextCheckpoint('{"summary":42}')).toBeNull();
		expect(parseContextCheckpoint('{"summary":""}')).toBeNull();
	});

	it('treats arrays and non-JSON braces as legacy bare strings', () => {
		expect(parseContextCheckpoint('[]')?.legacy).toBe(true);
		expect(parseContextCheckpoint('{not json')?.legacy).toBe(true);
	});
});

describe('getMessageCheckpoint', () => {
	it('reads both camelCase and snake_case fields', () => {
		expect(getMessageCheckpoint({ contextSummary: 'camel' })?.summary).toBe('camel');
		expect(getMessageCheckpoint({ context_summary: 'snake' })?.summary).toBe('snake');
	});

	it('returns null for messages without a checkpoint', () => {
		expect(getMessageCheckpoint({ content: 'hi' })).toBeNull();
		expect(getMessageCheckpoint(null)).toBeNull();
	});

	it('falls through an empty camelCase field to the snake_case one (backend `or` parity)', () => {
		// The backend selects the field with `contextSummary or context_summary`, so
		// an empty camelCase value must not mask a legacy snake_case record.
		expect(getMessageCheckpoint({ contextSummary: '', context_summary: 'snake' })?.summary).toBe(
			'snake'
		);
		expect(
			getMessageCheckpoint({ contextSummary: '', context_summary: '' })
		).toBeNull();
	});
});
