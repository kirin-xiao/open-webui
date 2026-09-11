import { describe, expect, it } from 'vitest';
import {
	findLiteralDynamicImports,
	parseMissingModule,
	resolveImportToPackage
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
