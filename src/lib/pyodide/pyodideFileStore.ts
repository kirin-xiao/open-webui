/**
 * IndexedDB-backed store for the Pyodide `/mnt/uploads` tree.
 *
 * The code interpreter runs in a null-origin sandbox iframe, where IndexedDB is
 * unavailable (opaque origin), so persistence is owned by the parent page: it
 * reads a runtime's in-memory FS and writes the contents here, surviving page
 * reloads and browser restarts.
 *
 * Entries are namespaced per signed-in user so a shared browser profile does not
 * leak one user's files to another. When IndexedDB is unavailable (SSR, tests,
 * private browsing) every operation degrades to a no-op.
 */

const DB_NAME = 'open-webui-pyodide-files';
const STORE_NAME = 'files';
const DB_VERSION = 1;
const NAMESPACE_SEPARATOR = '\u0000';
const DEFAULT_NAMESPACE = 'default';

export interface StoredFile {
	path: string;
	type: 'file' | 'directory';
	data?: ArrayBuffer;
	mtime: number;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

/** Test-only reset of the cached connection. */
export function resetPyodideFileStoreForTests(): void {
	dbPromise = null;
}

function keyFor(namespace: string | undefined, path: string): string {
	return `${namespace || DEFAULT_NAMESPACE}${NAMESPACE_SEPARATOR}${path}`;
}

function prefixFor(namespace: string | undefined): string {
	return `${namespace || DEFAULT_NAMESPACE}${NAMESPACE_SEPARATOR}`;
}

function indexedDbAvailable(): boolean {
	try {
		return typeof indexedDB !== 'undefined' && indexedDB !== null;
	} catch {
		return false;
	}
}

function openDb(): Promise<IDBDatabase | null> {
	if (dbPromise) return dbPromise;
	if (!indexedDbAvailable()) {
		dbPromise = Promise.resolve(null);
		return dbPromise;
	}

	const pending = new Promise<IDBDatabase | null>((resolve) => {
		let request: IDBOpenDBRequest;
		try {
			request = indexedDB.open(DB_NAME, DB_VERSION);
		} catch {
			resolve(null);
			return;
		}

		request.onupgradeneeded = () => {
			const db = request.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				db.createObjectStore(STORE_NAME, { keyPath: 'key' });
			}
		};
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => resolve(null);
		// Another tab may hold an older connection open during an upgrade. Do
		// not cache that as "unavailable" forever; let a later call retry.
		request.onblocked = () => resolve(undefined as unknown as IDBDatabase);
	});

	dbPromise = pending.then((db) => {
		if (!db) {
			// Do not cache a failed/blocked open; retry on the next call.
			dbPromise = null;
			return null;
		}
		return db;
	});

	return dbPromise;
}

/** Read every stored entry for a namespace. Never rejects. */
export async function readAllFiles(namespace?: string): Promise<StoredFile[]> {
	const db = await openDb();
	if (!db) return [];

	return new Promise((resolve) => {
		try {
			const tx = db.transaction(STORE_NAME, 'readonly');
			const request = tx.objectStore(STORE_NAME).getAll();
			request.onsuccess = () => {
				const prefix = prefixFor(namespace);
				const rows = (request.result ?? []) as Array<StoredFile & { key: string }>;
				resolve(
					rows
						.filter((row) => typeof row.key === 'string' && row.key.startsWith(prefix))
						.map((row) => ({
							path: row.path,
							type: row.type,
							data: row.data,
							mtime: row.mtime
						}))
				);
			};
			request.onerror = () => resolve([]);
		} catch {
			resolve([]);
		}
	});
}

/** Insert or replace the given files. Resolves `false` if the write failed. */
export async function putFiles(
	namespace: string | undefined,
	files: StoredFile[]
): Promise<boolean> {
	if (files.length === 0) return true;
	return runWrite((store) => {
		for (const file of files) {
			store.put({
				key: keyFor(namespace, file.path),
				path: file.path,
				type: file.type,
				data: file.data,
				mtime: file.mtime
			});
		}
	});
}

/** Delete the given paths. Resolves `false` if the write failed. */
export async function removePaths(
	namespace: string | undefined,
	paths: string[]
): Promise<boolean> {
	if (paths.length === 0) return true;
	return runWrite((store) => {
		for (const path of paths) store.delete(keyFor(namespace, path));
	});
}

/** Delete every entry for a namespace. Never rejects. */
export async function clearAllFiles(namespace?: string): Promise<void> {
	const files = await readAllFiles(namespace);
	await removePaths(
		namespace,
		files.map((file) => file.path)
	);
}

function runWrite(run: (store: IDBObjectStore) => void): Promise<boolean> {
	return openDb().then(
		(db) =>
			new Promise<boolean>((resolve) => {
				if (!db) {
					resolve(false);
					return;
				}
				try {
					const tx = db.transaction(STORE_NAME, 'readwrite');
					run(tx.objectStore(STORE_NAME));
					tx.oncomplete = () => resolve(true);
					tx.onerror = () => resolve(false);
					tx.onabort = () => resolve(false);
				} catch {
					resolve(false);
				}
			})
	);
}
