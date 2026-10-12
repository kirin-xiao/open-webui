/**
 * Helpers for preparing the Pyodide lockfile.
 *
 * The default (non-slim) build vendors only a curated subset of wheels in
 * `static/pyodide/`, while `pyodide-lock.json` advertises the whole Pyodide
 * distribution. For every lock entry whose wheel is not present locally we
 * rewrite `file_name` to the jsDelivr copy of the same official wheel, so the
 * browser runtime never requests a missing local wheel (404) and the `sha256`
 * integrity check still matches (jsDelivr serves the identical file).
 */

export const pyodideCdnBase = (version) => `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;

export function isRemoteFileName(fileName) {
	return typeof fileName === 'string' && /^https?:\/\//i.test(fileName);
}

export function baseName(fileName) {
	return typeof fileName === 'string' ? fileName.split('/').pop() : fileName;
}

/**
 * Rewrite `file_name` to the jsDelivr CDN URL for every package whose wheel is
 * not in `presentFileNames`.
 *
 * @returns {{ rewritten: number, unresolved: string[] }} counts for build logs.
 *   `unresolved` lists entries with no `file_name` at all.
 */
export function applyCdnFallback(lock, presentFileNames, version) {
	const present = new Set(presentFileNames);
	const baseUrl = pyodideCdnBase(version);
	let rewritten = 0;
	const unresolved = [];

	for (const [key, pkg] of Object.entries(lock.packages ?? {})) {
		const fileName = pkg.file_name;
		if (!fileName) {
			unresolved.push(key);
			continue;
		}
		if (isRemoteFileName(fileName)) continue;
		if (present.has(baseName(fileName))) continue;
		pkg.file_name = baseUrl + baseName(fileName);
		rewritten += 1;
	}

	return { rewritten, unresolved };
}

/**
 * Curated packages that have no entry in the lock at all. These cannot be
 * auto-loaded by `loadPackagesFromImports` and will fall back to PyPI at
 * runtime, so the build should warn about them.
 */
export function findMissingCurated(lock, curated) {
	return curated.filter((name) => {
		if (lock.packages?.[name]) return false;
		return !lock.packages?.[name.replace(/-/g, '_')];
	});
}

/**
 * Lock entries that are not resolvable: no `file_name`, or a local file name
 * that is not present. Remote entries are considered resolvable.
 */
export function findUnresolvedPackages(lock, presentFileNames) {
	const present = new Set(presentFileNames);
	const unresolved = [];
	for (const [key, pkg] of Object.entries(lock.packages ?? {})) {
		const fileName = pkg.file_name;
		if (!fileName || (!isRemoteFileName(fileName) && !present.has(baseName(fileName)))) {
			unresolved.push(key);
		}
	}
	return unresolved;
}
