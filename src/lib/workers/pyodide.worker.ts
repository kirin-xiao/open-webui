import { loadPyodide, type PyodideInterface } from 'pyodide';
import {
	findLiteralDynamicImports,
	parseMissingModule,
	resolveImportToPackage
} from '../pyodide/pyodidePackages';

declare global {
	interface Window {
		stdout: string | null;
		stderr: string | null;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		result: any;
		pyodide: PyodideInterface;
		packages: string[];
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		[key: string]: any;
	}
}

// ---------------------------------------------------------------------------
// Pyodide bootstrap
// ---------------------------------------------------------------------------

let pyodideReady: Promise<void> | null = null;

async function loadPyodideAndPackages(packages: string[] = []) {
	self.stdout = null;
	self.stderr = null;
	self.result = null;

	self.pyodide = await loadPyodide({
		indexURL: '/pyodide/',
		stdout: (text) => {
			console.log('Python output:', text);

			if (self.stdout) {
				self.stdout += `${text}\n`;
			} else {
				self.stdout = `${text}\n`;
			}
		},
		stderr: (text) => {
			console.log('An error occurred:', text);
			if (self.stderr) {
				self.stderr += `${text}\n`;
			} else {
				self.stderr = `${text}\n`;
			}
		},
		packages: ['micropip']
	});

	// Create the upload directory and mount IDBFS for persistence
	const uploadDir = '/mnt/uploads';
	self.pyodide.FS.mkdirTree(uploadDir);
	self.pyodide.FS.mount(self.pyodide.FS.filesystems.IDBFS, {}, '/mnt');

	// Load persisted files from IndexedDB
	await new Promise<void>((resolve) => {
		(self.pyodide.FS as any).syncfs(true, (err: Error | null) => {
			if (err) {
				console.error('Error syncing from IndexedDB:', err);
			}
			// Always resolve — missing data is fine on first run
			resolve();
		});
	});

	// Ensure /mnt/uploads still exists after sync (first-time init)
	try {
		self.pyodide.FS.stat(uploadDir);
	} catch {
		self.pyodide.FS.mkdirTree(uploadDir);
	}

	const micropip = self.pyodide.pyimport('micropip');
	await micropip.install(packages);
}

/**
 * Ensure Pyodide is loaded. On the first call, loads and installs packages.
 * Subsequent calls reuse the already-loaded instance (persistent worker).
 */
async function ensurePyodide(packages: string[] = []) {
	if (!pyodideReady) {
		pyodideReady = loadPyodideAndPackages(packages);
	}
	await pyodideReady;

	// Install any additional packages not loaded on init
	if (packages.length > 0 && self.pyodide) {
		const micropip = self.pyodide.pyimport('micropip');
		await micropip.install(packages);
	}
}

/**
 * Persist the in-memory FS to IndexedDB (fire-and-forget with logging).
 */
function persistFS() {
	if (!self.pyodide) return;
	(self.pyodide.FS as any).syncfs(false, (err: Error | null) => {
		if (err) {
			console.error('Error syncing to IndexedDB:', err);
		} else {
			console.log('Successfully synced to IndexedDB.');
		}
	});
}

// ---------------------------------------------------------------------------
// FS operations
// ---------------------------------------------------------------------------

function fsUploadFiles(files: { name: string; data: ArrayBuffer }[], dir = '/mnt/uploads') {
	try {
		self.pyodide.FS.stat(dir);
	} catch {
		self.pyodide.FS.mkdirTree(dir);
	}

	for (const file of files) {
		self.pyodide.FS.writeFile(`${dir}/${file.name}`, new Uint8Array(file.data));
	}
}

function fsList(path: string) {
	const entries: { name: string; type: 'file' | 'directory'; size: number }[] = [];
	try {
		const items = self.pyodide.FS.readdir(path).filter((n: string) => n !== '.' && n !== '..');
		for (const name of items) {
			try {
				const stat = self.pyodide.FS.stat(`${path}/${name}`);
				const isDir = self.pyodide.FS.isDir(stat.mode);
				entries.push({
					name,
					type: isDir ? 'directory' : 'file',
					size: isDir ? 0 : stat.size
				});
			} catch {
				// skip inaccessible entries
			}
		}
	} catch {
		// directory doesn't exist
	}
	return entries;
}

function fsRead(path: string): ArrayBuffer {
	const data: Uint8Array = (self.pyodide.FS as any).readFile(path) as Uint8Array;
	return data.buffer as ArrayBuffer;
}

function fsDelete(path: string) {
	try {
		const stat = self.pyodide.FS.stat(path);
		if (self.pyodide.FS.isDir(stat.mode)) {
			// Recursively delete directory contents
			const items = self.pyodide.FS.readdir(path).filter((n: string) => n !== '.' && n !== '..');
			for (const item of items) {
				fsDelete(`${path}/${item}`);
			}
			self.pyodide.FS.rmdir(path);
		} else {
			self.pyodide.FS.unlink(path);
		}
	} catch {
		// already gone
	}
}

function fsMkdir(path: string) {
	self.pyodide.FS.mkdirTree(path);
}

// ---------------------------------------------------------------------------
// Code execution
// ---------------------------------------------------------------------------

function appendOutput(current: string | null, line: string): string {
	return current ? `${current}${line}\n` : `${line}\n`;
}

type ExecutionPhase = 'loading' | 'packages' | 'executing';

/**
 * Report which phase an execution request is in. The host separates Pyodide
 * init / package downloads (`loading`, `packages`) from user-code execution
 * (`executing`) so a slow cold-cache install is not killed by the execution
 * timeout.
 */
function postStatus(id: string, phase: ExecutionPhase) {
	self.postMessage({ id, type: 'status', phase });
}

/**
 * Load packages that the code needs. `loadPackagesFromImports` only sees static
 * imports, so we additionally feed it string-literal dynamic imports
 * (`importlib.import_module("x")`, `__import__("x")`, `find_spec("x")`). Load
 * failures are returned rather than thrown so the user's code still runs and
 * can handle a missing package itself.
 */
async function loadPackagesForCode(code: string): Promise<string[]> {
	const loadErrors: string[] = [];

	const dynamicImports = findLiteralDynamicImports(code);
	if (dynamicImports.length > 0) {
		const synthetic = dynamicImports.map((name) => `import ${name}`).join('\n');
		try {
			await self.pyodide.loadPackagesFromImports(synthetic);
		} catch (error) {
			loadErrors.push(error instanceof Error ? error.message : String(error));
		}
	}

	try {
		await self.pyodide.loadPackagesFromImports(code, {
			errorCallback: (message: string) => loadErrors.push(message)
		});
	} catch (error) {
		loadErrors.push(error instanceof Error ? error.message : String(error));
	}

	return loadErrors;
}

/**
 * Run user code, and if it fails with an uncaught ModuleNotFoundError that maps
 * to a known Pyodide package, install that package and retry once. Otherwise
 * surface a precise error instead of a bare ModuleNotFoundError.
 */
async function runUserCode(id: string, code: string, loadErrors: string[]): Promise<unknown> {
	try {
		return await self.pyodide.runPythonAsync(code);
	} catch (error: unknown) {
		const message = error instanceof Error ? error.message : String(error);
		const missing = parseMissingModule(message);
		if (!missing) throw error;

		// `lockfile` is exposed by the Pyodide API but not in the public types.
		const lockfile = (
			self.pyodide as unknown as {
				lockfile?: { packages?: Record<string, { imports?: string[] }> };
			}
		).lockfile;
		const lockPkg = resolveImportToPackage(missing, lockfile?.packages);
		// Lock-listed packages resolve from the local/CDN index; anything else is
		// a best-effort PyPI install by its import name.
		const pkgToInstall = lockPkg ?? missing;

		try {
			postStatus(id, 'packages');
			const micropip = self.pyodide.pyimport('micropip');
			await micropip.install(pkgToInstall);
			self.stdout = appendOutput(
				self.stdout,
				`[open-webui] installed missing package '${pkgToInstall}' for module '${missing}' and retrying`
			);
			postStatus(id, 'executing');
			return await self.pyodide.runPythonAsync(code);
		} catch (installError: unknown) {
			const detail = installError instanceof Error ? installError.message : String(installError);
			const loader = loadErrors.length > 0 ? ` Loader: ${loadErrors.join('; ')}.` : '';
			throw new Error(
				`Module '${missing}' is not available in this Pyodide environment and could not be installed as '${pkgToInstall}': ${detail}.${loader}`
			);
		}
	}
}

async function executeCode(
	id: string,
	code: string,
	files?: { name: string; data: ArrayBuffer }[]
) {
	self.stdout = null;
	self.stderr = null;
	self.result = null;

	// Upload any accompanying files before execution
	if (files && files.length > 0) {
		fsUploadFiles(files);
		persistFS();
	}

	try {
		// Auto-load imported packages (static imports plus string-literal
		// dynamic imports). Load failures are collected, not thrown.
		postStatus(id, 'loading');
		const loadErrors = await loadPackagesForCode(code);

		// check if matplotlib is imported in the code
		if (code.includes('matplotlib')) {
			try {
				// Override plt.show() to return base64 image
				await self.pyodide.runPythonAsync(`import base64
import os
from io import BytesIO

# before importing matplotlib
# to avoid the wasm backend (which needs js.document', not available in worker)
os.environ["MPLBACKEND"] = "AGG"

import matplotlib.pyplot

_old_show = matplotlib.pyplot.show
assert _old_show, "matplotlib.pyplot.show"

def show(*, block=None):
	buf = BytesIO()
	matplotlib.pyplot.savefig(buf, format="png")
	buf.seek(0)
	# encode to a base64 str
	img_str = base64.b64encode(buf.read()).decode('utf-8')
	matplotlib.pyplot.clf()
	buf.close()
	print(f"data:image/png;base64,{img_str}")

matplotlib.pyplot.show = show`);
			} catch {
				// matplotlib unavailable; skip the patch so the user's code
				// runs and can handle the missing package itself.
			}
		}

		postStatus(id, 'executing');
		self.result = await runUserCode(id, code, loadErrors);

		// Safely process and recursively serialize the result
		self.result = processResult(self.result);

		console.log('Python result:', self.result);

		// Persist any files the code may have written
		persistFS();
	} catch (error: unknown) {
		self.stderr = error instanceof Error ? error.message : String(error);
	}

	self.postMessage({ id, result: self.result, stdout: self.stdout, stderr: self.stderr });
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async (event) => {
	const data = event.data;
	const { id, type } = data;

	// Backward compatibility: messages without a `type` field are execute requests
	if (!type || type === 'execute') {
		const { code, files, ...context } = data;

		// Copy context keys (packages, etc.) into worker scope
		for (const key of Object.keys(context)) {
			if (key !== 'id' && key !== 'type') {
				self[key] = context[key];
			}
		}

		await ensurePyodide(self.packages);
		await executeCode(id, code, files);
		return;
	}

	// FS operations require Pyodide to be loaded
	await ensurePyodide();

	switch (type) {
		case 'fs:upload': {
			const { files, dir } = data;
			fsUploadFiles(files, dir);
			persistFS();
			self.postMessage({ id, type: 'fs:upload', success: true });
			break;
		}

		case 'fs:list': {
			const entries = fsList(data.path);
			self.postMessage({ id, type: 'fs:list', entries });
			break;
		}

		case 'fs:read': {
			try {
				const buffer = fsRead(data.path);
				self.postMessage({ id, type: 'fs:read', data: buffer }, { transfer: [buffer] });
			} catch (err: unknown) {
				self.postMessage({
					id,
					type: 'fs:read',
					error: err instanceof Error ? err.message : String(err)
				});
			}
			break;
		}

		case 'fs:delete': {
			fsDelete(data.path);
			persistFS();
			self.postMessage({ id, type: 'fs:delete', success: true });
			break;
		}

		case 'fs:mkdir': {
			fsMkdir(data.path);
			persistFS();
			self.postMessage({ id, type: 'fs:mkdir', success: true });
			break;
		}

		case 'fs:sync': {
			// Re-read from IndexedDB into memory to pick up externally written files
			(self.pyodide.FS as any).syncfs(true, (err: Error | null) => {
				if (err) {
					console.error('Error syncing from IndexedDB:', err);
				}
				self.postMessage({ id, type: 'fs:sync', success: !err });
			});
			break;
		}

		default:
			console.warn('Unknown message type:', type);
	}
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function processResult(result: any): any {
	// Catch and always return JSON-safe string representations
	try {
		if (result == null) {
			// Handle null and undefined
			return null;
		}
		if (typeof result === 'string' || typeof result === 'number' || typeof result === 'boolean') {
			// Handle primitive types directly
			return result;
		}
		if (typeof result === 'bigint') {
			// Convert BigInt to a string for JSON-safe representation
			return result.toString();
		}
		if (Array.isArray(result)) {
			// If it's an array, recursively process items
			return result.map((item) => processResult(item));
		}
		if (typeof result.toJs === 'function') {
			// If it's a Pyodide proxy object (e.g., Pandas DF, Numpy Array), convert to JS and process recursively
			return processResult(result.toJs());
		}
		if (typeof result === 'object') {
			// Convert JS objects to a recursively serialized representation
			const processedObject: { [key: string]: any } = {};
			for (const key in result) {
				if (Object.prototype.hasOwnProperty.call(result, key)) {
					processedObject[key] = processResult(result[key]);
				}
			}
			return processedObject;
		}
		// Stringify anything that's left (e.g., Proxy objects that cannot be directly processed)
		return JSON.stringify(result);
	} catch (err: unknown) {
		// In case something unexpected happens, we return a stringified fallback
		return `[processResult error]: ${err instanceof Error ? err.message : String(err)}`;
	}
}

export default {};
