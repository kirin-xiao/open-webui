import { afterEach, describe, expect, it, vi } from 'vitest';
import { sandboxScript } from './pyodideSandboxHost';

describe('pyodide sandbox script', () => {
	it('inlines the shared package helpers so it cannot drift from the worker', () => {
		expect(sandboxScript).toContain('function parseMissingModule');
		expect(sandboxScript).toContain('function resolveImportToPackage');
		expect(sandboxScript).toContain('function findLiteralDynamicImports');
	});

	it('is syntactically valid after helper injection', () => {
		expect(() => new Function(sandboxScript)).not.toThrow();
	});

	it('binds the helpers under their original names even when the build minifies (renames) them', async () => {
		// Simulate a production esbuild minification where `Function.prototype.toString()`
		// returns a renamed function declaration. The generated sandbox script must still
		// declare `const findLiteralDynamicImports = function <mangled>(...) {...}` so the
		// call sites keep resolving.
		const originalToString = Function.prototype.toString;
		const spy = vi.spyOn(Function.prototype, 'toString').mockImplementation(function (
			this: object
		) {
			const body = originalToString
				.call(this)
				.replace(/^function\s+\w+\(/, 'function MangledName(');
			return body;
		});

		try {
			vi.resetModules();
			const mod = await import('./pyodideSandboxHost');
			const script: string = mod.sandboxScript;

			expect(script).toContain('const findLiteralDynamicImports = function MangledName(');
			expect(script).toContain('const parseMissingModule = function MangledName(');
			expect(script).toContain('const resolveImportToPackage = function MangledName(');
			expect(() => new Function(script)).not.toThrow();
		} finally {
			spy.mockRestore();
		}
	});
});

afterEach(() => {
	vi.restoreAllMocks();
});
