import 'fake-indexeddb/auto';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { config, user } from '$lib/stores';
import { clearAllFiles, readAllFiles, resetPyodideFileStoreForTests } from './pyodideFileStore';
import {
	__setPyodideFileSyncTimeoutForTests,
	flushPyodideFiles,
	hydratePyodideFiles
} from './pyodideFileSync';

type Message = { type?: string; id?: string; [key: string]: unknown };
type Listener = (event: { data: unknown }) => void;

/**
 * A minimal in-memory stand-in for the Pyodide sandbox worker's FS. It answers
 * the `fs:*` messages used by the sync layer and records uploaded/read files.
 * Individual handlers can be overridden to simulate failures.
 */
class FakeWorker {
	onmessage: Listener | null = null;
	files = new Map<string, { data: Uint8Array; mtime: number }>();
	dirs = new Set<string>(['/mnt', '/mnt/uploads']);
	mtime = 1;
	/** When set, requests of this type respond with `{ stderr }` instead. */
	failTypes = new Set<string>();

	private listeners = new Set<Listener>();

	addEventListener(type: string, listener: Listener) {
		if (type === 'message') this.listeners.add(listener);
	}

	removeEventListener(type: string, listener: Listener) {
		if (type === 'message') this.listeners.delete(listener);
	}

	terminate() {}

	postMessage(message: Message) {
		queueMicrotask(() => this.respond(message));
	}

	private emit(data: Record<string, unknown>) {
		for (const listener of this.listeners) listener({ data });
	}

	private respond(message: Message) {
		const { id, type } = message;
		if (type && this.failTypes.has(type)) {
			// Mirrors the sandbox's outer catch: no `error`, no `type`.
			this.emit({ id, stderr: `${type} failed` });
			return;
		}
		switch (type) {
			case 'fs:mkdir':
				this.dirs.add(message.path);
				this.emit({ id, type, success: true });
				break;
			case 'fs:upload':
				for (const file of message.files) {
					const path = `${message.dir}/${file.name}`.replace(/\/+/g, '/');
					this.files.set(path, { data: new Uint8Array(file.data), mtime: ++this.mtime });
				}
				this.emit({ id, type, success: true });
				break;
			case 'fs:read': {
				const entry = this.files.get(message.path);
				if (entry) {
					const buffer = entry.data.buffer.slice(0);
					this.emit({ id, type, data: buffer });
				} else {
					this.emit({ id, type, error: 'ENOENT' });
				}
				break;
			}
			case 'fs:tree': {
				const entries: Array<Record<string, unknown>> = [];
				for (const dir of this.dirs) entries.push({ path: dir, type: 'directory', mtime: 0 });
				for (const [path, entry] of this.files)
					entries.push({ path, type: 'file', size: entry.data.length, mtime: entry.mtime });
				this.emit({ id, type, entries });
				break;
			}
			default:
				this.emit({ id, type, success: true });
		}
	}
}

const runtime = (worker: FakeWorker) => ({ worker: worker as unknown as Worker });

/** Read the store under the namespace used by the test user. */
const readStore = () => readAllFiles('user-1');

const upload = (worker: FakeWorker, path: string, text: string) => {
	worker.files.set(path, { data: new TextEncoder().encode(text), mtime: ++worker.mtime });
};

const uploadMtime = (worker: FakeWorker, path: string, text: string) => {
	upload(worker, path, text);
	return worker.files.get(path)!.mtime;
};

describe('pyodide file sync', () => {
	beforeEach(async () => {
		resetPyodideFileStoreForTests();
		__setPyodideFileSyncTimeoutForTests(50);
		config.set({ features: { enable_pyodide_file_persistence: true } } as never);
		user.set({ id: 'user-1' } as never);
		await clearAllFiles('user-1');
	});

	afterEach(() => {
		__setPyodideFileSyncTimeoutForTests(null);
		config.set(undefined as never);
		user.set(undefined as never);
	});

	it('is a no-op when persistence is disabled', async () => {
		config.set({ features: { enable_pyodide_file_persistence: false } } as never);
		const worker = new FakeWorker();
		upload(worker, '/mnt/uploads/a.txt', 'a');

		await hydratePyodideFiles(runtime(worker));
		await flushPyodideFiles(runtime(worker));

		expect(await readStore()).toEqual([]);
	});

	it('holds the runtime busy for the duration of a sync pass', async () => {
		const worker = new FakeWorker();
		let busy = 0;
		let sawBusy = false;
		worker.addEventListener('message', (event) => {
			if ((event.data as { id?: string })?.id?.startsWith('pfs-')) sawBusy = busy > 0;
		});

		await flushPyodideFiles({
			worker: worker as unknown as Worker,
			acquire: () => (busy += 1),
			release: () => (busy -= 1)
		});

		expect(sawBusy).toBe(true);
		expect(busy).toBe(0);
	});

	it('hydrates persisted files into a fresh runtime', async () => {
		await uploadToStore('/mnt/uploads/a.txt', 'hello');
		await uploadToStore('/mnt/uploads/sub/b.txt', 'world');

		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));

		expect(new TextDecoder().decode(worker.files.get('/mnt/uploads/a.txt')!.data)).toBe('hello');
		expect(new TextDecoder().decode(worker.files.get('/mnt/uploads/sub/b.txt')!.data)).toBe(
			'world'
		);
	});

	it('flushes added and changed files to the store', async () => {
		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));

		upload(worker, '/mnt/uploads/out.txt', 'result');
		await flushPyodideFiles(runtime(worker));

		const stored = await readStore();
		const entry = stored.find((file) => file.path === '/mnt/uploads/out.txt');
		expect(entry).toBeTruthy();
		expect(new TextDecoder().decode(entry!.data)).toBe('result');
	});

	it('deletes store paths removed from the FS within the hydration snapshot', async () => {
		await uploadToStore('/mnt/uploads/keep.txt', 'keep');
		await uploadToStore('/mnt/uploads/drop.txt', 'drop');

		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));
		worker.files.delete('/mnt/uploads/drop.txt');

		await flushPyodideFiles(runtime(worker));

		const paths = (await readStore()).map((file) => file.path);
		expect(paths).toContain('/mnt/uploads/keep.txt');
		expect(paths).not.toContain('/mnt/uploads/drop.txt');
	});

	it('does not delete store files that were never hydrated (partial hydration)', async () => {
		await uploadToStore('/mnt/uploads/other.txt', 'other');

		// Runtime created empty and never hydrated: no snapshot, so flush must
		// not treat the stored file as deleted.
		const worker = new FakeWorker();
		upload(worker, '/mnt/uploads/mine.txt', 'mine');
		await flushPyodideFiles(runtime(worker));

		const paths = (await readStore()).map((file) => file.path);
		expect(paths).toContain('/mnt/uploads/other.txt');
		expect(paths).toContain('/mnt/uploads/mine.txt');
	});

	it('does not rewrite unchanged files on repeated flushes', async () => {
		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));
		const mtime = uploadMtime(worker, '/mnt/uploads/once.txt', 'once');

		await flushPyodideFiles(runtime(worker));
		const stored = await readStore();
		const entry = stored.find((file) => file.path === '/mnt/uploads/once.txt');
		expect(entry!.mtime).toBe(mtime);

		// A second flush with no FS change keeps the same stored copy.
		await flushPyodideFiles(runtime(worker));
		expect((await readStore()).find((file) => file.path === '/mnt/uploads/once.txt')!.mtime).toBe(
			mtime
		);
	});

	it('never rejects when the runtime does not answer', async () => {
		const worker = new FakeWorker();
		worker.postMessage = () => {};

		await expect(hydratePyodideFiles(runtime(worker))).resolves.toBeUndefined();
		await expect(flushPyodideFiles(runtime(worker))).resolves.toBeUndefined();
	});

	it('releases the runtime busy hold at the hydration deadline', async () => {
		await uploadToStore('/mnt/uploads/a.txt', 'hello');

		const worker = new FakeWorker();
		worker.postMessage = () => {};
		let busy = 0;

		await hydratePyodideFiles({
			worker: worker as unknown as Worker,
			acquire: () => (busy += 1),
			release: () => (busy -= 1)
		});

		// ready resolved, and the runtime is no longer pinned busy despite the
		// hydration task still draining its timed-out requests.
		expect(busy).toBe(0);
	});

	// ── Failure-mode regressions ────────────────────────────────────────────

	it('does not delete the store when fs:tree fails', async () => {
		await uploadToStore('/mnt/uploads/survivor.txt', 'survivor');

		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));
		worker.failTypes.add('fs:tree');

		await flushPyodideFiles(runtime(worker));

		expect((await readStore()).map((file) => file.path)).toContain('/mnt/uploads/survivor.txt');
	});

	it('does not mark a failed upload as hydrated, so it is not deleted later', async () => {
		await uploadToStore('/mnt/uploads/a.txt', 'hello');

		const worker = new FakeWorker();
		// Hydration's uploads fail (e.g. Pyodide failed to load).
		worker.failTypes.add('fs:upload');
		await hydratePyodideFiles(runtime(worker));

		// A later flush sees an empty FS; because a.txt was never hydrated it
		// must be left in the store, not treated as deleted.
		worker.failTypes.delete('fs:upload');
		await flushPyodideFiles(runtime(worker));

		expect((await readStore()).map((file) => file.path)).toContain('/mnt/uploads/a.txt');
	});

	it('retries an unreadable persisted file instead of deleting it', async () => {
		await uploadToStore('/mnt/uploads/unreadable.txt', 'data');

		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));

		// The file is modified, but reading it back fails this pass.
		worker.failTypes.add('fs:read');
		await flushPyodideFiles(runtime(worker));

		expect((await readStore()).map((file) => file.path)).toContain('/mnt/uploads/unreadable.txt');

		// Next pass, with reads working, the change is captured.
		worker.failTypes.delete('fs:read');
		upload(worker, '/mnt/uploads/unreadable.txt', 'updated');
		await flushPyodideFiles(runtime(worker));

		const entry = (await readStore()).find((file) => file.path === '/mnt/uploads/unreadable.txt');
		expect(new TextDecoder().decode(entry!.data)).toBe('updated');
	});

	it('keeps the previous stored copy of a file that grows past the cap', async () => {
		await uploadToStore('/mnt/uploads/big.bin', 'small');

		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));

		// Oversized entry in the tree (over 32MB) with a changed mtime.
		worker.files.set('/mnt/uploads/big.bin', {
			data: new Uint8Array(33 * 1024 * 1024),
			mtime: ++worker.mtime
		});
		await flushPyodideFiles(runtime(worker));

		const entry = (await readStore()).find((file) => file.path === '/mnt/uploads/big.bin');
		expect(entry).toBeTruthy();
		expect(new TextDecoder().decode(entry!.data)).toBe('small');
	});

	it('isolates persisted files per user', async () => {
		await uploadToStore('/mnt/uploads/mine.txt', 'user-1 data');

		user.set({ id: 'user-2' } as never);
		const worker = new FakeWorker();
		await hydratePyodideFiles(runtime(worker));

		expect(worker.files.has('/mnt/uploads/mine.txt')).toBe(false);
	});
});

async function uploadToStore(path: string, text: string) {
	const dir = path.slice(0, path.lastIndexOf('/'));
	const worker = new FakeWorker();
	worker.dirs.add(dir);
	await hydratePyodideFiles(runtime(worker));
	worker.files.set(path, { data: new TextEncoder().encode(text), mtime: ++worker.mtime });
	await flushPyodideFiles(runtime(worker));
}
