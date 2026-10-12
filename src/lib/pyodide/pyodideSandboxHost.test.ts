import { afterEach, describe, expect, it, vi } from 'vitest';
import { sandboxScript, sandboxHtml, matplotlibPatchPython } from './pyodideSandboxHost';

describe('pyodide sandbox script', () => {
	it('inlines the shared package helpers so it cannot drift from the worker', () => {
		expect(sandboxScript).toContain('function parseMissingModule');
		expect(sandboxScript).toContain('function resolveImportToPackage');
		expect(sandboxScript).toContain('function findLiteralDynamicImports');
		expect(sandboxScript).toContain('function summarizeInstallFailure');
	});

	it('is syntactically valid after helper injection', () => {
		expect(() => new Function(sandboxScript)).not.toThrow();
	});

	it('installs vendored matplotlib fonts before patching, and the sandbox HTML advertises their URLs', () => {
		// The bundled DejaVu/STIX faces have no CJK coverage, so the sandbox must
		// fetch the vendored font, and the URL list must actually reach it. The
		// HTML injection is the wiring that a reader-only assertion would miss.
		expect(sandboxScript).toContain('async function installMatplotlibFonts');
		expect(sandboxScript).toContain('__MATPLOTLIB_FONT_URLS__');
		expect(sandboxScript).toContain('await installMatplotlibFonts()');
		// Ordering: install must precede the patch that registers the files.
		expect(sandboxScript.indexOf('await installMatplotlibFonts()')).toBeLessThan(
			sandboxScript.indexOf('await patchMatplotlib()')
		);
		expect(sandboxHtml).toContain('__MATPLOTLIB_FONT_URLS__');
		expect(sandboxHtml).toContain('/pyodide/fonts/NotoSansSC-Regular.otf');
	});

	it('registers fonts by appending the family while preserving Latin defaults and the show() patch', () => {
		// The font setup and the show() patch share one Python program; assert on
		// the real emitted source rather than the surrounding JS string.
		expect(matplotlibPatchPython).toContain('_fm.fontManager.addfont(_path)');
		// Append, not replace: read the existing list first.
		expect(matplotlibPatchPython).toContain(
			'_families = list(matplotlib.rcParams.get("font.family", []))'
		);
		expect(matplotlibPatchPython).toContain('_families.append(_family)');
		expect(matplotlibPatchPython).toContain('matplotlib.rcParams["font.family"] = _families');
		expect(matplotlibPatchPython).toContain('matplotlib.rcParams["axes.unicode_minus"] = False');
		// Idempotency: addfont is unconditional in matplotlib, so registering on
		// every run would leak duplicate entries in a pooled interpreter.
		expect(matplotlibPatchPython).toContain('_openwebui_fonts_registered');
		// The existing show() patch must survive the added font setup.
		expect(matplotlibPatchPython).toContain('matplotlib.pyplot.show = show');
	});

	it('emits structurally valid Python: block statements are followed by a more-indented body', () => {
		// The JS `new Function` test above only parses JavaScript; the emitted
		// Python is never parsed by anything, so a broken indent would ship
		// silently. Guard the one failure mode that matters here: every line
		// opening a block (ending in ':') must be followed by a more-indented,
		// non-blank line (and indentation must be tab-consistent).
		const lines = matplotlibPatchPython.split('\n');
		const indentOf = (line: string) => line.match(/^\t*/)?.[0].length ?? 0;

		for (let i = 0; i < lines.length; i++) {
			const line = lines[i];
			const isBlockOpener =
				/:$/.test(line.trim()) && !/^\s*(#|['"])/.test(line) && !line.includes('=');
			if (!isBlockOpener) continue;
			const openerIndent = indentOf(line);
			const next = lines
				.slice(i + 1)
				.find((candidate) => candidate.trim() !== '' && !candidate.trim().startsWith('#'));
			if (next === undefined) continue;
			expect(indentOf(next)).toBeGreaterThan(openerIndent);
		}
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
			expect(script).toContain('const summarizeInstallFailure = function MangledName(');
			expect(() => new Function(script)).not.toThrow();
		} finally {
			spy.mockRestore();
		}
	});
});

afterEach(() => {
	vi.restoreAllMocks();
});
