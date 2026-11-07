import { describe, expect, it } from 'vitest';

import {
	collectPendingSteers,
	collectUnannotatedDeliveredSteers,
	hasPreSteerNote,
	isSteerMessage,
	isSubagentSteer,
	mergeSubagentCompletionStates,
	planBusySubmit,
	pruneSteerBranches,
	repairSteerCurrentId,
	steerEntriesFromAnnotation,
	steerEntryFromMessage,
	steerIdsInOutput,
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

	it('does not show a fallback bubble for a delivered steer already rendered inline', () => {
		const delivered = steerNode('s1', 'focus on tests', { status: 'delivered' });
		const messages = {
			s1: delivered,
			a1: {
				id: 'a1',
				role: 'assistant',
				content: 'done',
				output: [
					{
						type: 'open_webui:steering',
						id: 'steering-s1',
						steer_id: 's1',
						source: 'user',
						text: 'focus on tests'
					}
				]
			}
		};

		expect(steerIdsInOutput(messages)).toEqual(new Set(['s1']));
		expect(collectUnannotatedDeliveredSteers(messages)).toEqual([]);
	});

	it('does not duplicate a promoted steer that is rendered inline', () => {
		const byChild = steerNode(
			's1',
			'promoted by child',
			{ status: 'delivered' },
			{ childrenIds: ['a1'] }
		);
		const byMarker = steerNode(
			's2',
			'promoted by marker',
			{ status: 'delivered', promoted: true },
			{ childrenIds: [] }
		);
		const messages = {
			s1: byChild,
			s2: byMarker,
			a1: { id: 'a1', role: 'assistant', content: 'answer', parentId: 's1', childrenIds: [] }
		};

		expect(collectUnannotatedDeliveredSteers(messages)).toEqual([]);
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

	it('routes consumed subagent completions to the result row, not the steer chip', () => {
		const [entry] = steerEntriesFromAnnotation([
			{
				id: 'completion-1',
				source: 'subagent',
				content: '<subagent sessionID="child-1" state="completed">\nCHILD_TIP\n</subagent>'
			}
		]);

		expect(isSubagentSteer(entry)).toBe(true);
		expect(isSubagentSteer(steerEntriesFromAnnotation(['user steer'])[0])).toBe(false);
		expect(isSubagentSteer(undefined)).toBe(false);
	});
});

describe('subagent status reconciliation', () => {
	it('maps terminal child states to completed ids', () => {
		const completed = subagentStatesToCompleted(
			new Map([
				['done', 'completed'],
				['failed', 'error'],
				['stopped', 'interrupted'],
				['old', 'superseded'],
				['unknown', 'future-state'],
				['running', 'running'],
				['queued', 'steer_queued'],
				['blank', '']
			])
		);

		// Anything non-active is terminal, so a future state cannot pin the row.
		expect([...completed].sort()).toEqual(['done', 'failed', 'old', 'stopped', 'unknown']);
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

describe('steer branch pruning', () => {
	it('removes every steer node from childrenIds so it never becomes a branch', () => {
		const messages = {
			tip: { id: 'tip', role: 'assistant', content: 'done', childrenIds: ['s1', 'u2'] },
			s1: steerNode('s1', 'steer me'),
			u2: { id: 'u2', role: 'user', content: 'next', parentId: 'tip', childrenIds: ['s2'] },
			s2: steerNode('s2', 'delivered steer', { status: 'delivered' }),
			a2: { id: 'a2', role: 'assistant', content: 'ok', parentId: 'u2', childrenIds: [] }
		};

		expect(pruneSteerBranches(messages)).toBe(true);
		expect(messages.tip.childrenIds).toEqual(['u2']);
		expect(messages.u2.childrenIds).toEqual([]);
		// The nodes stay in history so the chip/annotation fallback can render.
		expect(messages.s1.content).toBe('steer me');
		expect(messages.s2.meta.status).toBe('delivered');
	});

	it('is a no-op for normal histories', () => {
		const messages = {
			u1: { id: 'u1', role: 'user', content: 'hi', childrenIds: ['a1'] },
			a1: { id: 'a1', role: 'assistant', content: 'hello', parentId: 'u1', childrenIds: [] }
		};

		expect(pruneSteerBranches(messages)).toBe(false);
		expect(messages.u1.childrenIds).toEqual(['a1']);
	});
});

describe('steer currentId repair', () => {
	it('moves a boundary steer currentId back to the nearest ancestor', () => {
		const messages = {
			tip: { id: 'tip', role: 'assistant', content: 'done', childrenIds: ['s1'] },
			s1: steerNode('s1', 'delivered steer', { status: 'delivered' }, { parentId: 'tip' })
		};
		const history = { currentId: 's1', messages };

		expect(repairSteerCurrentId(history)).toBe(true);
		expect(history.currentId).toBe('tip');
		expect(messages.tip.childrenIds).toEqual([]);
		// The node is still available for chip/annotation fallback rendering.
		expect(messages.s1.content).toBe('delivered steer');
	});

	it('moves a promoted steer currentId to its assistant descendant', () => {
		const messages = {
			tip: { id: 'tip', role: 'assistant', content: 'done', childrenIds: ['s1'] },
			s1: steerNode(
				's1',
				'promoted steer',
				{ status: 'delivered' },
				{ parentId: 'tip', childrenIds: ['a1'] }
			),
			a1: { id: 'a1', role: 'assistant', content: 'answer', parentId: 's1', childrenIds: [] }
		};
		const history = { currentId: 's1', messages };

		expect(repairSteerCurrentId(history)).toBe(true);
		expect(history.currentId).toBe('a1');
		expect(messages.s1.childrenIds).toEqual(['a1']);
	});

	it('re-anchors a polluted next turn and never keeps the steer as currentId', () => {
		const messages = {
			tip: { id: 'tip', role: 'assistant', content: 'done', childrenIds: ['s1'] },
			s1: steerNode(
				's1',
				'delivered steer',
				{ status: 'delivered' },
				{ parentId: 'tip', childrenIds: ['u2'] }
			),
			u2: { id: 'u2', role: 'user', content: 'next', parentId: 's1', childrenIds: [] }
		};
		const history = { currentId: 's1', messages };

		expect(repairSteerCurrentId(history)).toBe(true);
		expect(messages.u2.parentId).toBe('tip');
		expect(messages.tip.childrenIds).toEqual(['u2']);
		expect(messages.s1.childrenIds).toEqual([]);
		expect(history.currentId).toBe('u2');
	});

	it('re-anchors past a chain of non-promoted steers', () => {
		const messages = {
			tip: { id: 'tip', role: 'assistant', content: 'done', childrenIds: ['s1'] },
			s1: steerNode(
				's1',
				'outer',
				{ status: 'delivered' },
				{ parentId: 'tip', childrenIds: ['s2'] }
			),
			s2: steerNode(
				's2',
				'inner',
				{ status: 'delivered' },
				{ parentId: 's1', childrenIds: ['u2'] }
			),
			u2: { id: 'u2', role: 'user', content: 'next', parentId: 's2', childrenIds: [] }
		};
		const history = { currentId: 's1', messages };

		expect(repairSteerCurrentId(history)).toBe(true);
		expect(messages.u2.parentId).toBe('tip');
		expect(messages.tip.childrenIds).toContain('u2');
		expect(messages.s2.childrenIds).toEqual([]);
		expect(history.currentId).not.toBe('s1');
	});

	it('moves a marker-promoted steer currentId to its assistant descendant', () => {
		const messages = {
			tip: { id: 'tip', role: 'assistant', content: 'done', childrenIds: ['s1'] },
			s1: steerNode(
				's1',
				'promoted steer',
				{ status: 'delivered', promoted: true },
				{ parentId: 'tip', childrenIds: ['a1'] }
			),
			a1: { id: 'a1', role: 'assistant', content: 'answer', parentId: 's1', childrenIds: [] }
		};
		const history = { currentId: 's1', messages };

		expect(repairSteerCurrentId(history)).toBe(true);
		expect(history.currentId).toBe('a1');
	});

	it('leaves healthy histories untouched', () => {
		const messages = {
			u1: { id: 'u1', role: 'user', content: 'hi', childrenIds: ['a1'] },
			a1: { id: 'a1', role: 'assistant', content: 'hello', parentId: 'u1', childrenIds: [] }
		};
		const history = { currentId: 'a1', messages };

		expect(repairSteerCurrentId(history)).toBe(false);
		expect(history.currentId).toBe('a1');
	});
});

describe('steer transcript rendering guard', () => {
	it('normalizes a steer node into the compact block entry', () => {
		const entry = steerEntryFromMessage(
			steerNode('s1', 'look closer', { status: 'delivered' }),
			's1'
		);

		expect(entry).toEqual(
			expect.objectContaining({ id: 's1', content: 'look closer', status: 'delivered' })
		);
	});

	it('does not treat a normal user message or an empty steer as a steer block', () => {
		expect(steerEntryFromMessage({ id: 'u1', role: 'user', content: 'hi' }, 'u1')).toBeNull();
		expect(steerEntryFromMessage(steerNode('s2', '   '), 's2')).toBeNull();
	});
});

describe('busy-turn submission routing', () => {
	const base = { delivery: 'steer' as const, hasFiles: false, enableMessageQueue: true };

	it('steers by default while a turn runs', () => {
		expect(planBusySubmit(base)).toBe('steer');
	});

	it('queues when the user chose queue-by-default', () => {
		expect(planBusySubmit({ ...base, delivery: 'queue' })).toBe('queue');
	});

	it('never steers attachments', () => {
		expect(planBusySubmit({ ...base, hasFiles: true })).toBe('queue');
		expect(planBusySubmit({ ...base, hasFiles: true, enableMessageQueue: false })).toBe(
			'interrupt'
		);
	});

	it('falls back to queue or interrupt when the steer request fails', () => {
		expect(planBusySubmit({ ...base, steerOutcome: 'failed' })).toBe('queue');
		expect(planBusySubmit({ ...base, steerOutcome: 'failed', enableMessageQueue: false })).toBe(
			'interrupt'
		);
	});

	it('submits normally only after the server reports no active task', () => {
		expect(planBusySubmit({ ...base, steerOutcome: 'inactive' })).toBe('submit');
		expect(planBusySubmit({ ...base, steerOutcome: 'queued' })).toBe('steer');
	});
});
