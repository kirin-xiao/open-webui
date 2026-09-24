import { describe, expect, it } from 'vitest';

import {
	buildOutputDisplayItems,
	collectCompletedSubagentIds,
	parseSubagentResult,
	type OutputDetailToken,
	type OutputDisplayItem,
	type OutputItem
} from './structuredOutput';

function allTokens(items: OutputDisplayItem[]): OutputDetailToken[] {
	return items.flatMap((item) =>
		item.type === 'detail_single'
			? [item.token]
			: item.type === 'detail_group'
				? item.tokens
				: []
	);
}

function subagentOutput(resultText: string, background: boolean): OutputItem[] {
	return [
		{
			type: 'function_call',
			name: 'subagent',
			call_id: 'call-1',
			status: 'completed',
			arguments: JSON.stringify({ description: 'do the thing', background })
		},
		{
			type: 'function_call_output',
			call_id: 'call-1',
			output: [{ type: 'output_text', text: resultText }]
		}
	];
}

describe('parseSubagentResult', () => {
	it('reads the background dispatch handle', () => {
		expect(
			parseSubagentResult(
				JSON.stringify({ sessionID: 'c1', status: 'running', output: 'working' }),
				'Background subagent: "x"'
			)
		).toEqual({ sessionID: 'c1', state: 'running', background: true });
	});

	it('still detects the legacy hyphenated background display name', () => {
		expect(
			parseSubagentResult(
				JSON.stringify({ sessionID: 'c1', status: 'running', output: 'working' }),
				'Background sub-agent: "x"'
			)
		).toEqual({ sessionID: 'c1', state: 'running', background: true });
	});

	it('reads the result envelope', () => {
		expect(
			parseSubagentResult('<subagent sessionID="c2" state="completed">done</subagent>')
		).toEqual({ sessionID: 'c2', state: 'completed', background: false });
	});

	it('returns empty values when neither shape is present', () => {
		expect(parseSubagentResult('')).toEqual({ sessionID: '', state: '', background: false });
	});
});

describe('buildOutputDisplayItems subagent rows', () => {
	it('keeps a background dispatch spinning while the child is not complete', () => {
		const tokens = allTokens(
			buildOutputDisplayItems(
				subagentOutput(JSON.stringify({ sessionID: 'c1', status: 'running' }), true)
			)
		);
		const token = tokens.find((t) => t.attributes.type === 'tool_calls');

		expect(token?.attributes.done).not.toBe('true');
		expect(token?.attributes.sessionID).toBe('c1');
		expect(token?.attributes.status).toBe('in_progress');
	});

	it('marks a background dispatch done once the child is completed', () => {
		const tokens = allTokens(
			buildOutputDisplayItems(
				subagentOutput(JSON.stringify({ sessionID: 'c1', status: 'running' }), true),
				new Set(['c1'])
			)
		);
		const token = tokens.find((t) => t.attributes.type === 'tool_calls');

		expect(token?.attributes.done).toBe('true');
		expect(token?.attributes.sessionID).toBe('c1');
	});

	it('marks a foreground result done and links to the child', () => {
		const tokens = allTokens(
			buildOutputDisplayItems(
				subagentOutput('<subagent sessionID="c2" state="completed">done</subagent>', false)
			)
		);
		const token = tokens.find((t) => t.attributes.type === 'tool_calls');

		expect(token?.attributes.sessionID).toBe('c2');
		expect(token?.attributes.done).toBe('true');
	});

	it('emits the new (unhyphenated) subagent display names', () => {
		const foreground = allTokens(
			buildOutputDisplayItems(
				subagentOutput('<subagent sessionID="c2" state="completed">done</subagent>', false)
			)
		).find((t) => t.attributes.type === 'tool_calls');
		expect(foreground?.attributes.name).toBe('Subagent: "do the thing"');

		const background = allTokens(
			buildOutputDisplayItems(
				subagentOutput(JSON.stringify({ sessionID: 'c1', status: 'running' }), true)
			)
		).find((t) => t.attributes.type === 'tool_calls');
		expect(background?.attributes.name).toBe('Background subagent: "do the thing"');
	});

	it('links a continuation from the call arguments before a result exists', () => {
		const output: OutputItem[] = [
			{
				type: 'function_call',
				name: 'subagent',
				call_id: 'call-9',
				status: 'completed',
				arguments: JSON.stringify({ description: 'follow up', sessionID: 'child-9' })
			}
		];
		const token = allTokens(buildOutputDisplayItems(output)).find(
			(t) => t.attributes.type === 'tool_calls'
		);

		expect(token?.attributes.sessionID).toBe('child-9');
	});

	it('links a running row from the live subagent:created event store', () => {
		const output: OutputItem[] = [
			{
				type: 'function_call',
				name: 'subagent',
				call_id: 'call-1',
				status: 'completed',
				arguments: JSON.stringify({ description: 'do the thing', background: false })
			}
		];
		const token = allTokens(
			buildOutputDisplayItems(
				output,
				undefined,
				new Map([['msg-1:call-1', 'live-child']]),
				'msg-1'
			)
		).find((t) => t.attributes.type === 'tool_calls');

		expect(token?.attributes.sessionID).toBe('live-child');
	});
});

describe('collectCompletedSubagentIds', () => {
	it('excludes unfinished rows and includes completed/legacy child ids', () => {
		const ids = collectCompletedSubagentIds({
			pending: {
				content: '',
				meta: { internal: true, type: 'subagent', status: 'pending', childID: 'pending-child' }
			},
			running: {
				content: '',
				meta: { internal: true, type: 'subagent', state: 'running', childID: 'running-child' }
			},
			completed: {
				content: '<subagent sessionID="c2" state="completed">ok</subagent>',
				meta: {
					internal: true,
					type: 'subagent',
					state: 'completed',
					childID: 'c2',
					childIDs: ['c3']
				}
			},
			// Deferred completion: the child finished, only the parent synthesis
			// is still pending, so the dispatch row must flip to done.
			deferred: {
				content: '<subagent sessionID="deferred-child" state="completed">ok</subagent>',
				meta: {
					internal: true,
					type: 'subagent',
					state: 'completed',
					status: 'pending',
					childID: 'deferred-child'
				}
			},
			legacy: {
				content: '',
				meta: {
					internal: true,
					type: 'subagent',
					state: 'error',
					subagent_chat_id: 'c4',
					subagent_chat_ids: ['c5']
				}
			},
			unrelated: {
				content: '',
				meta: { internal: true, type: 'note', childID: 'note-child' }
			}
		});

		expect(ids.has('pending-child')).toBe(false);
		expect(ids.has('running-child')).toBe(false);
		expect(ids.has('c2')).toBe(true);
		expect(ids.has('c3')).toBe(true);
		expect(ids.has('c4')).toBe(true);
		expect(ids.has('c5')).toBe(true);
		expect(ids.has('deferred-child')).toBe(true);
		expect(ids.has('note-child')).toBe(false);
	});
});