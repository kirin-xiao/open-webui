import { describe, expect, it } from 'vitest';
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
});
