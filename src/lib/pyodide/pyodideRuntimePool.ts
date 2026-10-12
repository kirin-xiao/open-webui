import { createPyodideWorker } from './createPyodideWorker';
import { hydratePyodideFiles } from './pyodideFileSync';

/**
 * Per-chat Pyodide runtime pool.
 *
 * Every execution must run against its own interpreter to isolate `sys.modules`
 * (module attribute mutations such as `X, Y = np.meshgrid = None, None` must not
 * leak across chats). A full interpreter per call is too expensive (~1-4s plus
 * imports), so runtimes are pooled per stable chat id and reused within a chat.
 *
 * Each live interpreter can retain 100-300 MB once scientific packages load, so
 * the pool is bounded by an LRU and disposes the least-recently-used idle
 * runtime. Runtimes with in-flight work are never evicted; a reset requested
 * while a runtime is busy is deferred until that work finishes.
 *
 * When file persistence is enabled, a new runtime hydrates `/mnt/uploads` from
 * IndexedDB. `ready` resolves when that best-effort pass finishes, so callers
 * wait for their files before the first execution.
 */

const DEFAULT_RUNTIME_KEY = '__default__';

const DEFAULT_MAX_RUNTIMES = 2;

export interface PyodideRuntime {
	/** Stable key this runtime is pooled under (chat id or a sentinel). */
	readonly chatId: string;
	/** The underlying worker or sandboxed iframe (Worker-shaped). */
	readonly worker: Worker;
	/**
	 * Resolves once any hydration of this runtime has finished (or immediately
	 * when persistence is off). Never rejects and never hangs: a reset during
	 * hydration still resolves.
	 */
	readonly ready: Promise<void>;
	/** Mark the runtime busy so it cannot be evicted mid-execution. */
	acquire(): void;
	/** Release a previous acquire, allowing eviction when idle. */
	release(): void;
	/**
	 * Drop this exact runtime. Terminates immediately when idle, otherwise
	 * defers until `release`. Targeting the handle (not a chat id) means a
	 * reset always affects the interpreter that actually misbehaved, even if
	 * the pooling mode changed in the meantime.
	 */
	reset(): void;
}

interface PoolEntry {
	chatId: string;
	worker: Worker;
	busy: number;
	lastUsed: number;
	/** Set when the runtime is dropped; terminated as soon as it goes idle. */
	disposed: boolean;
	ready: Promise<void>;
}

let pool = new Map<string, PoolEntry>();
let accessClock = 0;

let workerFactory: () => Worker = createPyodideWorker;
let maxRuntimes = DEFAULT_MAX_RUNTIMES;

function normalizeChatId(chatId?: string | null): string {
	return chatId ? String(chatId) : DEFAULT_RUNTIME_KEY;
}

function maybeEvict(): void {
	if (pool.size <= maxRuntimes) return;

	const candidates = [...pool.values()]
		.filter((entry) => entry.busy === 0)
		.sort((a, b) => a.lastUsed - b.lastUsed);

	let surplus = pool.size - maxRuntimes;
	for (const entry of candidates) {
		if (surplus <= 0) break;
		entry.worker.terminate();
		pool.delete(entry.chatId);
		surplus--;
	}
}

function dropEntry(entry: PoolEntry): void {
	entry.disposed = true;
	if (pool.get(entry.chatId) === entry) {
		pool.delete(entry.chatId);
	}
	if (entry.busy === 0) {
		entry.worker.terminate();
	}
}

function makeHandle(entry: PoolEntry): PyodideRuntime {
	return {
		chatId: entry.chatId,
		worker: entry.worker,
		// Getter, not a captured value: the pool assigns `entry.ready` after the
		// handle exists (hydration needs the handle's acquire/release).
		get ready(): Promise<void> {
			return entry.ready;
		},
		acquire() {
			entry.busy += 1;
			entry.lastUsed = ++accessClock;
		},
		release() {
			if (entry.busy === 0) return;
			entry.busy -= 1;
			entry.lastUsed = ++accessClock;

			// A reset/eviction requested while this runtime was busy is honoured
			// now that the last in-flight execution has finished.
			if (entry.busy === 0 && entry.disposed) {
				entry.worker.terminate();
				return;
			}

			if (pool.get(entry.chatId) === entry) maybeEvict();
		},
		reset() {
			dropEntry(entry);
		}
	};
}

/**
 * Get (or lazily create) the runtime for a chat. Pass a stable chat id, not the
 * socket session id, so the interpreter survives socket reconnects. An empty or
 * missing chat id falls back to a shared default runtime.
 */
export function getPyodideRuntime(chatId?: string | null): PyodideRuntime {
	const key = normalizeChatId(chatId);

	let entry = pool.get(key);
	if (!entry) {
		entry = {
			chatId: key,
			worker: workerFactory(),
			busy: 0,
			lastUsed: ++accessClock,
			disposed: false,
			ready: Promise.resolve()
		};
		pool.set(key, entry);

		const handle = makeHandle(entry);
		// Hold the runtime busy for the hydration pass so it cannot be evicted
		// (iframe removed, `fs:*` never answered) before its files are restored.
		entry.ready = hydratePyodideFiles(handle);
		maybeEvict();
		return handle;
	}

	entry.lastUsed = ++accessClock;
	return makeHandle(entry);
}

/**
 * Terminate and drop a chat's runtime. Persisted files survive because they live
 * in IndexedDB; in-memory state (variables, imported modules) is discarded.
 *
 * Prefer `runtime.reset()` when you hold the handle that timed out: re-deriving
 * the key here can target a different runtime if the pooling mode changed.
 * If the runtime is mid-execution the termination is deferred; dropping it
 * immediately would kill an in-flight run whose response the caller awaits.
 */
export function resetPyodideRuntime(chatId?: string | null): void {
	const key = normalizeChatId(chatId);
	const entry = pool.get(key);
	if (!entry) return;

	dropEntry(entry);
}

/** Terminate every pooled runtime (e.g. on app teardown / mode change). */
export function disposePyodideRuntimes(): void {
	for (const entry of [...pool.values()]) {
		entry.disposed = true;
		if (entry.busy === 0) {
			entry.worker.terminate();
		}
	}
	pool.clear();
}

/** Test-only hooks. */
export function __setPyodideWorkerFactoryForTests(
	factory: (() => Worker) | null,
	max = DEFAULT_MAX_RUNTIMES
): void {
	disposePyodideRuntimes();
	workerFactory = factory ?? createPyodideWorker;
	maxRuntimes = max;
	pool = new Map();
	accessClock = 0;
}
