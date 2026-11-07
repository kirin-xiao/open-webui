import { describe, expect, it } from 'vitest';

import {
	collectPendingSteers,
	collectUnannotatedDeliveredSteers,
	hasPreSteerNote,
	isSteerMessage,
	mergeSubagentCompletionStates,
	steerEntriesFromAnnotation,
	subagentStatesToCompleted
} from './steers';

function steerNode(
	id: string,
	content: string,
	meta: Record<string, unknown> = {},
	extra: Record<string, unknown> = {}
) {
	return {
		id,
		role: 'user',
		content,
		childrenIds: [],
		timestamp: 10,
		meta: { internal: true, type: 'steer', source: 'user', status: 'pending', ...meta },
		...extra
	};
}

describe('steer history nodes', () => {
	it('recognizes the steer subtype and ignores normal user messages', () => {
		expect(isSteerMessage(steerNode('s1', 'look closer'))).toBe(true);
		expect(isSteerMessage({ role: 'user', content: 'hi', meta: {} })).toBe(false);
		expect(
			isSteerMessage({ role: 'assistant', content: 'hi', meta: { internal: true, type: 'steer' } })
		).toBe(false);
	});

	it('collects pending parent steers as queued chips', () => {
		const pending = collectPendingSteers({
			s1: steerNode('s1', 'first'),
			s2: steerNode('s2', 'second', { status: 'delivered' }),
			s3: steerNode('s3', 'child steer', { childID: 'child-1' }),
			s4: steerNode('s4', 'superseded', { status: 'superseded' })
		});

		expect(pending.map((entry) => entry.id)).toEqual(['s1']);
		expect(pending[0].content).toBe('first');
		expect(pending[0].status).toBe('pending');
	});

	it('renders delivered nodes only when the assistant has no annotation yet', () => {
		const delivered = steerNode('s1', 'focus on tests', { status: 'delivered' });
		const before = collectUnannotatedDeliveredSteers({ s1: delivered });
		expect(before.map((entry) => entry.id)).toEqual(['s1']);

		const after = collectUnannotatedDeliveredSteers({
			s1: delivered,
			a1: {
				id: 'a1',
				role: 'assistant',
				content: 'done',
				meta: { steers: [{ content: 'focus on tests' }] }
			}
		});
		expect(after).toEqual([]);
	});
});

describe('assistant meta.steers annotations', () => {
	it('accepts strings and objects and exposes the pre-steer flag', () => {
		expect(steerEntriesFromAnnotation(['one', { content: 'two', pre_steer: true }])).toEqual([
			expect.objectContaining({ content: 'one', status: 'delivered', preSteer: false }),
			expect.objectContaining({ content: 'two', status: 'delivered', preSteer: true })
		]);
		expect(hasPreSteerNote({ steers: [{ content: 'late', pre_steer: true }] })).toBe(true);
		expect(hasPreSteerNote({ pre_steer: true })).toBe(true);
		expect(hasPreSteerNote({ steers: ['early'] })).toBe(false);
	});

	it('tolerates missing and legacy annotation shapes', () => {
		expect(steerEntriesFromAnnotation(undefined)).toEqual([]);
		expect(steerEntriesFromAnnotation({ a: 'wrapped' }).map((entry) => entry.content)).toEqual([
			'wrapped'
		]);
		expect(steerEntriesFromAnnotation([{ text: 'legacy text' }])[0].content).toBe('legacy text');
	});
});

describe('subagent status reconciliation', () => {
	it('maps terminal child states to completed ids', () => {
		const completed = subagentStatesToCompleted(
			new Map([
				['done', 'completed'],
				['failed', 'error'],
				['stopped', 'interrupted'],
				['running', 'running'],
				['queued', 'steer_queued']
			])
		);

		expect([...completed].sort()).toEqual(['done', 'failed', 'stopped']);
	});

	it('lets persisted state stop a missed-completion spinner', () => {
		// No completion message in history; the status endpoint is the only truth.
		const completed = mergeSubagentCompletionStates(new Set(), new Map([['child-1', 'completed']]));

		expect(completed.has('child-1')).toBe(true);
	});

	it('keeps a newer running generation spinning over an old result', () => {
		const completed = mergeSubagentCompletionStates(
			new Set(['child-1']),
			new Map([['child-1', 'running']])
		);

		expect(completed.has('child-1')).toBe(false);
	});

	it('keeps an undelivered steer active', () => {
		const completed = mergeSubagentCompletionStates(
			new Set(),
			new Map([['child-1', 'steer_queued']])
		);

		expect(completed.has('child-1')).toBe(false);
	});
});
