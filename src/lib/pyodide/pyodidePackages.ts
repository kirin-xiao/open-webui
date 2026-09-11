/**
 * Pure helpers shared by the two Pyodide runtimes (the persistent worker and
 * the iframe sandbox) and by unit tests.
 *
 * These functions must stay self-contained (no imports, no external
 * references), because `pyodideSandboxHost.ts` injects their source into the
 * sandboxed script via `Function.prototype.toString()`.
 */

export interface LockPackageLike {
	imports?: string[];
}

/**
 * Extract the top-level module name from a Python `ModuleNotFoundError`
 * message (e.g. `No module named 'pandas.core'` -> `pandas`).
 */
export function parseMissingModule(message: string): string | null {
	const match = /No module named ['"]([^'"]+)['"]/.exec(message ?? '');
	if (!match || !match[1]) return null;
	return match[1].split('.')[0];
}

/**
 * Map an import name to a Pyodide lock package key using each package's
 * `imports` field (e.g. `sklearn` -> `scikit-learn`, `bs4` -> `beautifulsoup4`,
 * `PIL` -> `pillow`). Returns null when the module is not in the lock.
 */
export function resolveImportToPackage(
	importName: string,
	lockPackages: Record<string, LockPackageLike> | null | undefined
): string | null {
	if (!importName || !lockPackages) return null;
	const normalized = importName.replace(/-/g, '_');

	for (const [name, pkg] of Object.entries(lockPackages)) {
		if (name === importName || name === normalized) return name;
		const imports = pkg?.imports;
		if (Array.isArray(imports) && imports.includes(importName)) return name;
	}

	return null;
}

/**
 * Find string-literal dynamic imports that `loadPackagesFromImports` cannot see
 * because it only inspects static import statements. Covers the common
 * availability-probe patterns.
 */
export function findLiteralDynamicImports(code: string): string[] {
	const names = new Set<string>();
	const patterns = [
		/importlib\s*\.\s*import_module\s*\(\s*(['"])([\w.]+)\1/g,
		/__import__\s*\(\s*(['"])([\w.]+)\1/g,
		/(?:importlib\s*\.\s*util\s*\.\s*)?find_spec\s*\(\s*(['"])([\w.]+)\1/g
	];

	for (const pattern of patterns) {
		let match: RegExpExecArray | null;
		while ((match = pattern.exec(code)) !== null) {
			if (match[2]) names.add(match[2].split('.')[0]);
		}
	}

	return [...names];
}
