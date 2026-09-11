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

	it('ignores computed module names', () => {
		expect(findLiteralDynamicImports('importlib.import_module(name)')).toEqual([]);
	});
});
