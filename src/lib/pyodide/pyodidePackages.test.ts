import { describe, expect, it } from 'vitest';
import {
	findLiteralDynamicImports,
	parseMissingModule,
	resolveImportToPackage,
	summarizeInstallFailure
} from './pyodidePackages';

describe('parseMissingModule', () => {
	it('extracts the top-level module from a Python error message', () => {
		expect(parseMissingModule("ModuleNotFoundError: No module named 'pandas'")).toBe('pandas');
		expect(parseMissingModule('No module named "pandas.core"')).toBe('pandas');
	});

	it('returns null for unrelated messages', () => {
		expect(parseMissingModule('NameError: name x is not defined')).toBe(null);
		expect(parseMissingModule('')).toBe(null);
	});
});

describe('resolveImportToPackage', () => {
	const lockPackages = {
		pandas: { imports: ['pandas'] },
		'scikit-learn': { imports: ['sklearn'] },
		beautifulsoup4: { imports: ['bs4'] },
		pillow: { imports: ['PIL'] }
	};

	it('resolves aliased import names via the lock imports field', () => {
		expect(resolveImportToPackage('sklearn', lockPackages)).toBe('scikit-learn');
		expect(resolveImportToPackage('bs4', lockPackages)).toBe('beautifulsoup4');
		expect(resolveImportToPackage('PIL', lockPackages)).toBe('pillow');
		expect(resolveImportToPackage('pandas', lockPackages)).toBe('pandas');
	});

	it('returns null for unknown modules or missing lock', () => {
		expect(resolveImportToPackage('seaborn', lockPackages)).toBe(null);
		expect(resolveImportToPackage('pandas', null)).toBe(null);
	});
});

describe('summarizeInstallFailure', () => {
	it('leads with the blocking dependency when micropip reports a missing pure-Python wheel', () => {
		const detail =
			"Traceback (most recent call last):\n  ...\nValueError: Can't find a pure Python 3 wheel for 'curl-cffi>=0.15'.";
		const message = summarizeInstallFailure('yfinance', 'yfinance', detail, []);

		expect(message).toContain("Module 'yfinance' cannot be installed");
		expect(message).toContain("its dependency 'curl-cffi' has no pure-Python wheel");
		expect(message.indexOf('curl-cffi')).toBeLessThan(message.indexOf('Traceback'));
	});

	it('strips version specifiers and extras from the blocking requirement', () => {
		const message = summarizeInstallFailure(
			'foo',
			'foo',
			"ValueError: Can't find a pure Python 3 wheel for 'requests[security]>=2'.",
			[]
		);
		const lead = message.split(' Detail: ')[0];

		expect(lead).toContain("its dependency 'requests' has no pure-Python wheel");
		expect(lead).not.toContain('[security]');
	});

	it('does not call the requested package its own dependency', () => {
		const message = summarizeInstallFailure(
			'opencv-python',
			'opencv-python',
			"ValueError: Can't find a pure Python 3 wheel for 'opencv-python'.",
			[]
		);

		expect(message).toContain("Module 'opencv-python' cannot be installed");
		expect(message).toContain('it has no pure-Python wheel');
		expect(message).not.toContain('its dependency');
	});

	it('falls back to the not-available framing for unrelated install errors', () => {
		const message = summarizeInstallFailure('yfinance', 'yfinance', 'HTTP 404', []);

		expect(message).toContain("Module 'yfinance' is not available in this Pyodide environment");
		expect(message).toContain('could not be installed as');
		expect(message).toContain('Detail: HTTP 404.');
	});

	it('omits an empty Detail segment', () => {
		const message = summarizeInstallFailure('yfinance', 'yfinance', '', []);

		expect(message).not.toContain('Detail:');
	});

	it('appends loader diagnostics without dropping the primary message', () => {
		const message = summarizeInstallFailure('yfinance', 'yfinance', 'ValueError: boom', [
			'Failed to load yfinance'
		]);

		expect(message).toContain('Detail: ValueError: boom.');
		expect(message).toContain('Loader: Failed to load yfinance.');
	});

	it('is self-contained when stringified (sandbox injection contract)', () => {
		const isolated = new Function(
			`return (${summarizeInstallFailure.toString()})`
		)() as typeof summarizeInstallFailure;
		expect(
			isolated(
				'akshare',
				'akshare',
				"Can't find a pure Python 3 wheel for 'mini-racer>=0.12.4; platform_system != \"Linux\"'.",
				[]
			)
		).toContain("its dependency 'mini-racer' has no pure-Python wheel");
	});
});

describe('findLiteralDynamicImports', () => {
	it('finds importlib.import_module / __import__ / find_spec string literals', () => {
		const code = `
import importlib
importlib.import_module("pandas")
i = importlib.import_module('numpy.core')
__import__("scipy")
importlib.util.find_spec("matplotlib")
find_spec("sklearn")
`;

		expect(findLiteralDynamicImports(code).sort()).toEqual(
			['matplotlib', 'numpy', 'pandas', 'scipy', 'sklearn'].sort()
		);
	});

	it('matches import_module even when imported via from-import', () => {
		expect(
			findLiteralDynamicImports('from importlib import import_module\nimport_module("pandas")')
		).toEqual(['pandas']);
	});

	it('ignores computed module names', () => {
		expect(findLiteralDynamicImports('importlib.import_module(name)')).toEqual([]);
	});

	it('does not over-match find_spec when preceded by a word char or dot', () => {
		expect(findLiteralDynamicImports('myfind_spec("numpy")\nloader.find_spec("numpy")')).toEqual(
			[]
		);
	});

	it('does not over-match import_module / importlib.util.find_spec either', () => {
		expect(
			findLiteralDynamicImports(
				'notimportlib.import_module("numpy")\nmyimportlib.util.find_spec("numpy")'
			)
		).toEqual([]);
	});

	it('ignores probe patterns inside comments and docstrings', () => {
		const code = `# __import__("scipy") in a comment
"""__import__("requests") in a docstring"""
'''
__import__("torch") across lines
'''
importlib.import_module("pandas")
`;
		expect(findLiteralDynamicImports(code)).toEqual(['pandas']);
	});

	it('keeps # inside a string literal', () => {
		expect(findLiteralDynamicImports('x = "a#b"\nimportlib.import_module("numpy")')).toEqual([
			'numpy'
		]);
	});

	it('does not treat an escaped triple-quote as the end of a docstring', () => {
		const code =
			'doc = """say \\"""\n__import__("scipy")\n"""\nimportlib.import_module("pandas")\n';
		expect(findLiteralDynamicImports(code)).toEqual(['pandas']);
	});

	it('is self-contained when stringified (sandbox injection contract)', () => {
		// pyodideSandboxHost.ts injects findLiteralDynamicImports via toString().
		// If it references another module-level binding, production minification
		// renames that reference but not the stringified copy, producing a
		// ReferenceError. Evaluating the stringified function in isolation (where
		// no module bindings exist) catches that: the old cross-helper call threw
		// "stripCommentsAndDocstrings is not defined" here.
		const isolated = new Function(
			`return (${findLiteralDynamicImports.toString()})`
		)() as typeof findLiteralDynamicImports;
		expect(isolated('importlib.import_module("numpy")')).toEqual(['numpy']);
		expect(isolated('# __import__("scipy")\nfind_spec("sklearn")')).toEqual(['sklearn']);
	});
});
