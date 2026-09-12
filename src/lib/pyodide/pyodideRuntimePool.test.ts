import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./createPyodideWorker', () => ({
	createPyodideWorker: vi.fn()
}));

import { config } from '$lib/stores';
import {
	__setPyodideWorkerFactoryForTests,
	disposePyodideRuntimes,
	getPyodideRuntime,
	resetPyodideRuntime
} from './pyodideRuntimePool';

function makeMockWorker() {
	return {
		terminate: vi.fn(),
		postMessage: vi.fn(),
		addEventListener: vi.fn(),
		removeEventListener: vi.fn()
	} as unknown as Worker;
}

const terminated = (worker: Worker) =>
	(worker.terminate as ReturnType<typeof vi.fn>).mock.calls.length;

describe('pyodide runtime pool', () => {
	let created: Worker[];
	let factory: () => Worker;

	beforeEach(() => {
		config.set({ features: { enable_pyodide_file_persistence: false } } as never);
		created = [];
		factory = vi.fn(() => {
			const worker = makeMockWorker();
			created.push(worker);
			return worker;
		});
		__setPyodideWorkerFactoryForTests(factory, 2);
	});

	afterEach(() => {
		disposePyodideRuntimes();
		__setPyodideWorkerFactoryForTests(null, 2);
		config.set(undefined as never);
	});

	it('creates one runtime per chat and reuses it within the chat', () => {
		const a1 = getPyodideRuntime('chat-a');
		const a2 = getPyodideRuntime('chat-a');

		expect(factory).toHaveBeenCalledTimes(1);
		expect(a1.worker).toBe(a2.worker);
	});

	it('isolates different chats in different interpreters', () => {
		const a = getPyodideRuntime('chat-a');
		const b = getPyodideRuntime('chat-b');

		expect(factory).toHaveBeenCalledTimes(2);
		expect(a.worker).not.toBe(b.worker);
	});

	it('falls back to a shared default runtime when chat id is missing', () => {
		const a = getPyodideRuntime(undefined);
		const b = getPyodideRuntime('');
		const c = getPyodideRuntime(null);

		expect(factory).toHaveBeenCalledTimes(1);
		expect(a.worker).toBe(b.worker);
		expect(b.worker).toBe(c.worker);
	});

	it('evicts the least-recently-used idle runtime past the cap', () => {
		const a = getPyodideRuntime('chat-a');
		getPyodideRuntime('chat-b');
		// Touch a so b becomes least-recently-used, then create c.
		getPyodideRuntime('chat-a');
		const c = getPyodideRuntime('chat-c');

		expect(factory).toHaveBeenCalledTimes(3);
		// chat-b was idle and least recently used -> evicted.
		expect(terminated(a.worker)).toBe(0);
		expect(terminated(created[1])).toBe(1);
		expect(terminated(c.worker)).toBe(0);
	});

	it('never evicts a busy runtime', () => {
		const a = getPyodideRuntime('chat-a');
		a.acquire();

		getPyodideRuntime('chat-b');
		getPyodideRuntime('chat-c');

		// chat-a is busy so chat-b (older but idle) is evicted instead.
		expect(terminated(created[1])).toBe(1);
		expect(terminated(created[0])).toBe(0);
	});

	it('allows eviction after a busy runtime is released', () => {
		const a = getPyodideRuntime('chat-a');
		a.acquire();
		const b = getPyodideRuntime('chat-b');
		const c = getPyodideRuntime('chat-c');

		// b was evicted when c was created because a was busy (and therefore not
		// an eviction candidate).
		expect(terminated(b.worker)).toBe(1);

		// Releasing a lets eviction proceed again: creating d evicts c.
		a.release();
		getPyodideRuntime('chat-d');

		expect(terminated(a.worker)).toBe(0);
		expect(terminated(c.worker)).toBe(1);
	});

	it('reset terminates the runtime and a later request recreates it', () => {
		const first = getPyodideRuntime('chat-a').worker;
		resetPyodideRuntime('chat-a');

		expect(terminated(first)).toBe(1);

		const second = getPyodideRuntime('chat-a').worker;
		expect(factory).toHaveBeenCalledTimes(2);
		expect(second).not.toBe(first);
	});

	it('defers reset termination until a busy runtime goes idle', () => {
		const runtime = getPyodideRuntime('chat-a');
		runtime.acquire();

		resetPyodideRuntime('chat-a');

		// Still running, so the worker must not be killed yet.
		expect(terminated(runtime.worker)).toBe(0);

		runtime.release();

		expect(terminated(runtime.worker)).toBe(1);

		// A fresh runtime is handed out afterwards.
		const next = getPyodideRuntime('chat-a');
		expect(factory).toHaveBeenCalledTimes(2);
		expect(next.worker).not.toBe(runtime.worker);
	});

	it('handle reset targets the exact runtime and defers while busy', () => {
		const runtime = getPyodideRuntime('chat-a');
		runtime.acquire();

		runtime.reset();

		expect(terminated(runtime.worker)).toBe(0);
		// The chat is immediately free to get a fresh interpreter.
		const next = getPyodideRuntime('chat-a');
		expect(factory).toHaveBeenCalledTimes(2);
		expect(next.worker).not.toBe(runtime.worker);

		// The old handle terminates itself on release and never touches the new.
		runtime.release();
		expect(terminated(runtime.worker)).toBe(1);
		expect(terminated(next.worker)).toBe(0);
	});

	it('release after reset terminates and does not corrupt live pool usage', () => {
		const runtime = getPyodideRuntime('chat-a');
		runtime.acquire();
		resetPyodideRuntime('chat-a');

		expect(() => runtime.release()).not.toThrow();
		expect(terminated(runtime.worker)).toBe(1);

		// The pool is still usable and creates a fresh runtime for the chat.
		const next = getPyodideRuntime('chat-a');
		expect(factory).toHaveBeenCalledTimes(2);
		expect(next.worker).not.toBe(runtime.worker);
	});

	it('keeps a single shared runtime while IDBFS persistence is enabled', () => {
		config.set({ features: { enable_pyodide_file_persistence: true } } as never);

		const a = getPyodideRuntime('chat-a');
		const b = getPyodideRuntime('chat-b');

		expect(factory).toHaveBeenCalledTimes(1);
		expect(a.worker).toBe(b.worker);
	});

	it('drops pooled runtimes when the persistence mode changes', () => {
		const a = getPyodideRuntime('chat-a').worker;

		config.set({ features: { enable_pyodide_file_persistence: true } } as never);

		// Mode flip disposes the old pool, so chat-a's old worker is terminated
		// and a subsequent request builds a fresh one under the new key scheme.
		getPyodideRuntime('chat-b');
		expect(terminated(a)).toBe(1);
		expect(factory).toHaveBeenCalledTimes(2);
	});

	it('defers mode-change disposal of a busy runtime', () => {
		const runtime = getPyodideRuntime('chat-a');
		runtime.acquire();

		config.set({ features: { enable_pyodide_file_persistence: true } } as never);

		// Triggers syncPersistenceMode -> dispose, but the runtime is busy.
		const b = getPyodideRuntime('chat-b');
		expect(terminated(runtime.worker)).toBe(0);
		expect(b.worker).not.toBe(runtime.worker);

		runtime.release();
		expect(terminated(runtime.worker)).toBe(1);
		expect(terminated(b.worker)).toBe(0);
	});

	it('dispose terminates every pooled runtime', () => {
		const a = getPyodideRuntime('chat-a').worker;
		const b = getPyodideRuntime('chat-b').worker;

		disposePyodideRuntimes();

		expect(terminated(a)).toBe(1);
		expect(terminated(b)).toBe(1);
	});

	it('dispose defers termination of a busy runtime', () => {
		const runtime = getPyodideRuntime('chat-a');
		runtime.acquire();

		disposePyodideRuntimes();

		expect(terminated(runtime.worker)).toBe(0);
		runtime.release();
		expect(terminated(runtime.worker)).toBe(1);
	});
});
