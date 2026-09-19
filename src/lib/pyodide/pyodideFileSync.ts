import { get } from 'svelte/store';

import { config, user } from '$lib/stores';
import { readAllFiles, putFiles, removePaths, type StoredFile } from './pyodideFileStore';

/**
 * Bridges a Pyodide runtime's in-memory `/mnt/uploads` tree to the parent-side
 * IndexedDB store.
 *
 * `hydrate` copies the store into a freshly created runtime; `flush` copies the
 * runtime's changes back. Flush deletes are snapshot-scoped: a store path is
 * removed only when it was present in this runtime's hydration snapshot and is
 * now absent from its FS. That prevents a partial hydration or a runtime that
 * never hydrated (e.g. created before persistence was enabled) from deleting
 * unrelated files.
 */

export const UPLOADS_DIR = '/mnt/uploads';

const DEFAULT_MESSAGE_TIMEOUT_MS = 30000;
let messageTimeoutMs = DEFAULT_MESSAGE_TIMEOUT_MS;
// Shorter than the interpreter load budget, so a wedged hydration cannot use up
// the whole pre-execution timeout.
const DEFAULT_HYDRATION_TIMEOUT_MS = 15000;
let hydrationTimeoutMs = DEFAULT_HYDRATION_TIMEOUT_MS;

interface RuntimeLike {
	readonly worker: Worker;
	/** Optional: when present, the runtime is held busy for the whole pass so
	 * it cannot be LRU-evicted (and terminated) mid-sync. */
	acquire?(): void;
	release?(): void;
}

interface TreeEntry {
	path: string;
	type: 'file' | 'directory';
	size?: number;
	mtime: number;
}

/** Last known FS state per runtime, used to compute flush deltas. */
const snapshots = new WeakMap<Worker, Map<string, number>>();
/** Serializes hydrate/flush passes per runtime so they cannot interleave. */
const queues = new WeakMap<Worker, Promise<unknown>>();
let requestId = 0;

function persistenceEnabled(): boolean {
	return get(config)?.features?.enable_pyodide_file_persistence === true;
}

function enqueue<T>(worker: Worker, task: () => Promise<T>): Promise<T> {
	const previous = queues.get(worker) ?? Promise.resolve();
	const next = previous.then(task, task);
	// Keep the chain alive for error-free sequencing; result is returned to caller.
	queues.set(
		worker,
		next.catch(() => undefined)
	);
	return next;
}

/**
 * Send one `fs:*` request and await its response.
 *
 * The sandbox reports a handler failure as `{ id, stderr }` with no `error`
 * field, so both shapes are treated as a rejection. Callers rely on this to
 * distinguish "did not happen" from "happened" — most importantly so a failed
 * hydration never marks paths as restored (which would make the next flush
 * delete them).
 */
function request<T = Record<string, unknown>>(
	worker: Worker,
	message: Record<string, unknown>,
	timeoutMs = messageTimeoutMs
): Promise<T> {
	const id = `pfs-${++requestId}`;
	return new Promise<T>((resolve, reject) => {
		let settled = false;
		const timeout = setTimeout(() => {
			settled = true;
			worker.removeEventListener('message', handler);
			reject(new Error('Pyodide file sync timeout'));
		}, timeoutMs);

		const release = () => {
			clearTimeout(timeout);
			worker.removeEventListener('message', handler);
		};

		function handler(event: MessageEvent) {
			if (settled) return;
			if (event.data?.id !== id) return;
			if (event.data?.type === 'status') return;
			settled = true;
			release();
			if (event.data?.error || event.data?.stderr) {
				reject(new Error(String(event.data.error ?? event.data.stderr)));
			} else {
				resolve(event.data as T);
			}
		}

		try {
			worker.addEventListener('message', handler);
			worker.postMessage({ ...message, id });
		} catch (error) {
			if (!settled) {
				settled = true;
				release();
				reject(error);
			}
		}
	});
}

/** Test-only override of the per-message timeout. */
export function __setPyodideFileSyncTimeoutForTests(
	timeoutMs: number | null,
	hydrationMs: number | null = null
): void {
	messageTimeoutMs = timeoutMs ?? DEFAULT_MESSAGE_TIMEOUT_MS;
	hydrationTimeoutMs = hydrationMs ?? timeoutMs ?? DEFAULT_HYDRATION_TIMEOUT_MS;
}

function parentDir(path: string): string {
	const index = path.lastIndexOf('/');
	return index <= 0 ? '/' : path.slice(0, index);
}

function baseName(path: string): string {
	return path.split('/').pop() ?? path;
}

/**
 * Run a sync pass with the runtime held busy, so an idle runtime cannot be
 * evicted (iframe removed, `fs:*` never answered) while we are talking to it.
 * Nested acquires are counted by the pool and are safe.
 */
async function guarded<T>(runtime: RuntimeLike, task: () => Promise<T>): Promise<T> {
	runtime.acquire?.();
	try {
		return await task();
	} finally {
		runtime.release?.();
	}
}

async function readTree(worker: Worker, timeoutMs = messageTimeoutMs): Promise<TreeEntry[]> {
	const res = await request<{ entries?: TreeEntry[] }>(
		worker,
		{
			type: 'fs:tree',
			path: UPLOADS_DIR
		},
		timeoutMs
	);
	if (!Array.isArray(res.entries)) {
		throw new Error('Pyodide file sync: malformed fs:tree response');
	}
	return res.entries;
}

function toSnapshot(entries: TreeEntry[]): Map<string, number> {
	const map = new Map<string, number>();
	for (const entry of entries) map.set(entry.path, entry.mtime);
	return map;
}

/**
 * Size cap for a single persisted file. Larger files remain readable/copyable in
 * the runtime but their previous stored copy is kept rather than deleted, so
 * growing a file past the cap does not silently lose the persisted version.
 */
const MAX_PERSISTED_FILE_BYTES = 32 * 1024 * 1024;

/**
 * Copy the persisted store into an empty runtime. Best-effort: any failure
 * (IndexedDB unavailable, timeout, runtime reset mid-hydration) resolves so a
 * caller's `ready` gate can never hang an execution.
 */
export async function hydratePyodideFiles(runtime: RuntimeLike): Promise<void> {
	if (!persistenceEnabled()) return;
	const worker = runtime.worker;

	// Hold the runtime busy so it cannot be evicted mid-hydration. Release at the
	// deadline even if the task is still draining (a request on a dead iframe
	// only settles on its own timeout) so a broken runtime is not pinned busy —
	// and thus un-evictable and un-resettable — for the whole drain.
	runtime.acquire?.();
	let released = false;
	const release = () => {
		if (released) return;
		released = true;
		runtime.release?.();
	};

	const task = enqueue(worker, async () => {
		try {
			const namespace = get(user)?.id;
			// Bound the whole task, not just the caller's gate: after the
			// deadline each remaining request would otherwise wait its full
			// timeout, holding the per-runtime queue (and thus later flushes)
			// for many timeouts on a dead iframe.
			const deadline = Date.now() + hydrationTimeoutMs;
			const ask = (message: Record<string, unknown>) => {
				const remaining = deadline - Date.now();
				if (remaining <= 0) return Promise.reject(new Error('Pyodide file sync: deadline'));
				return request(worker, message, Math.min(messageTimeoutMs, remaining));
			};

			const stored = await readAllFiles(namespace);
			if (stored.length === 0) {
				snapshots.set(worker, new Map());
				return;
			}

			// Record only what actually lands in the FS. If hydration is cut
			// short (deadline, reset, a failed upload), the un-restored paths
			// stay out of the snapshot so a later flush cannot mistake them for
			// deletions.
			const hydrated = new Map<string, number>();

			for (const entry of stored) {
				if (entry.type !== 'directory') continue;
				try {
					await ask({ type: 'fs:mkdir', path: entry.path });
					hydrated.set(entry.path, entry.mtime);
				} catch {
					// leave unhydrated; not added to the snapshot
				}
			}
			for (const entry of stored) {
				if (entry.type !== 'file' || !entry.data) continue;
				try {
					await ask({
						type: 'fs:upload',
						files: [{ name: baseName(entry.path), data: entry.data }],
						dir: parentDir(entry.path)
					});
					hydrated.set(entry.path, entry.mtime);
				} catch {
					// leave unhydrated; not added to the snapshot
				}
			}

			// Adopt the FS's real mtimes so the first flush does not rewrite
			// everything just because upload assigns a new mtime.
			try {
				const tree = await readTree(worker, Math.max(0, deadline - Date.now()));
				for (const entry of tree) {
					if (hydrated.has(entry.path)) hydrated.set(entry.path, entry.mtime);
				}
			} catch {
				// keep the stored mtimes
			}

			snapshots.set(worker, hydrated);
		} catch {
			// Best-effort: hydration must never fail the caller's ready gate.
		}
	});

	try {
		await withHydrationDeadline(task);
	} finally {
		release();
	}
	// The task may still be draining after a deadline; swallow its settlement.
	task.catch(() => undefined);
}

/**
 * Bound the whole hydration pass so a runtime that never answers cannot keep a
 * caller's `ready` gate pending. The task is not cancelled (there is no way to
 * interrupt a sandboxed run), but the caller releases its busy hold at the
 * deadline regardless.
 */
function withHydrationDeadline(task: Promise<unknown>): Promise<void> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	return Promise.race([
		task.then(() => undefined),
		new Promise<void>((resolve) => {
			timer = setTimeout(resolve, hydrationTimeoutMs);
		})
	]).finally(() => clearTimeout(timer));
}

/**
 * Persist a runtime's FS changes. Snapshot-scoped deletes plus add/edit writes.
 * Best-effort: failure (IndexedDB unavailable/quota, wedged runtime) must never
 * break the execution that triggered it.
 */
export async function flushPyodideFiles(runtime: RuntimeLike): Promise<void> {
	if (!persistenceEnabled()) return;
	const worker = runtime.worker;

	await guarded(runtime, () =>
		enqueue(worker, async () => {
			const namespace = get(user)?.id;
			const previous = snapshots.get(worker) ?? new Map<string, number>();

			let entries: TreeEntry[];
			try {
				entries = await readTree(worker);
			} catch {
				// Could not read the FS at all. Do not treat that as "empty":
				// leave the snapshot untouched so nothing is deleted and the
				// next flush retries.
				return;
			}

			const current = toSnapshot(entries);
			const writes: StoredFile[] = [];
			const deletes: string[] = [];
			// Paths whose contents could not be read this pass. They still exist
			// in the FS, so they must not be treated as deletions; leaving them
			// out of `current` makes the next flush retry them.
			const unreadable = new Set<string>();

			for (const entry of entries) {
				const changed = previous.get(entry.path) !== entry.mtime;
				if (entry.type === 'directory') {
					if (changed) writes.push({ path: entry.path, type: 'directory', mtime: entry.mtime });
					continue;
				}
				if (!changed) continue;
				// Oversized file: keep its last persisted copy instead of
				// reading/writing megabytes (and never delete it). If it shrinks
				// below the cap, its new mtime re-triggers this branch.
				if (entry.size != null && entry.size > MAX_PERSISTED_FILE_BYTES) continue;
				try {
					const res = await request<{ data?: ArrayBuffer }>(worker, {
						type: 'fs:read',
						path: entry.path
					});
					if (res.data) {
						writes.push({
							path: entry.path,
							type: 'file',
							data: res.data,
							mtime: entry.mtime
						});
					} else {
						unreadable.add(entry.path);
					}
				} catch {
					unreadable.add(entry.path);
				}
			}

			for (const path of previous.keys()) {
				if (!current.has(path) && !unreadable.has(path)) deletes.push(path);
			}

			const wrote = await putFiles(namespace, writes);
			const removed = await removePaths(namespace, deletes);

			// Only advance past the paths that actually persisted. A failed write
			// keeps its old (stale) mtime so the next flush retries it; a failed
			// delete keeps the path in the snapshot so it is retried too.
			if (!wrote) {
				for (const file of writes) current.set(file.path, previous.get(file.path) ?? -1);
			}
			if (!removed) {
				for (const path of deletes) current.set(path, previous.get(path) ?? -1);
			}

			for (const path of unreadable) current.delete(path);

			snapshots.set(worker, current);
		})
	).catch(() => undefined);
}
