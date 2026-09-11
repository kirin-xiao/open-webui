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
 * availability-probe patterns. Comments and docstrings are ignored so they
 * cannot trigger false package loads.
 *
 * This function MUST stay self-contained (no references to other module-level
 * bindings): `pyodideSandboxHost.ts` injects its `toString()` output into the
 * sandbox, and production minification renames module-level references but
 * cannot rewrite them inside the stringified source.
 */
export function findLiteralDynamicImports(code: string): string[] {
	// Blank out `#` comments and triple-quoted strings (docstrings / multiline
	// strings) so the patterns below don't match an import merely mentioned in
	// prose. Regular single/double-quoted strings are preserved because the
	// probe patterns rely on them.
	function stripCommentsAndDocstrings(source: string): string {
		const out: string[] = [];
		let i = 0;
		const n = source.length;
		let quote: string | null = null;
		let triple = false;

		while (i < n) {
			const ch = source[i];

			if (!quote && (ch === '"' || ch === "'") && source[i + 1] === ch && source[i + 2] === ch) {
				quote = ch;
				triple = true;
				out.push('   ');
				i += 3;
				continue;
			}

			if (triple && quote && ch === quote && source[i + 1] === quote && source[i + 2] === quote) {
				out.push('   ');
				i += 3;
				quote = null;
				triple = false;
				continue;
			}

			if (quote) {
				if (triple) {
					// Preserve escapes so `\"""` does not close the string early.
					if (ch === '\\' && i + 1 < n) {
						out.push(' ', ' ');
						i += 2;
						continue;
					}
					out.push(' ');
					i++;
					continue;
				}
				out.push(ch);
				if (ch === '\\' && i + 1 < n) {
					out.push(source[i + 1]);
					i += 2;
					continue;
				}
				if (ch === quote) quote = null;
				i++;
				continue;
			}

			if (ch === '"' || ch === "'") {
				quote = ch;
				triple = false;
				out.push(ch);
				i++;
				continue;
			}

			if (ch === '#') {
				while (i < n && source[i] !== '\n') {
					out.push(' ');
					i++;
				}
				continue;
			}

			out.push(ch);
			i++;
		}

		return out.join('');
	}

	const names = new Set<string>();
	const patterns = [
		/(?:^|[^\w.])(?:importlib\s*\.\s*)?import_module\s*\(\s*(['"])([\w.]+)\1/g,
		/__import__\s*\(\s*(['"])([\w.]+)\1/g,
		/(?:^|[^\w.])(?:importlib\s*\.\s*util\s*\.\s*)?find_spec\s*\(\s*(['"])([\w.]+)\1/g
	];
	const source = stripCommentsAndDocstrings(code);

	for (const pattern of patterns) {
		let match: RegExpExecArray | null;
		while ((match = pattern.exec(source)) !== null) {
			if (match[2]) names.add(match[2].split('.')[0]);
		}
	}

	return [...names];
}
