import { describe, expect, it } from 'vitest';
import {
	applyCdnFallback,
	baseName,
	findMissingCurated,
	findUnresolvedPackages,
	isRemoteFileName,
	pyodideCdnBase
} from './pyodide-lock-utils.js';

const makeLock = (packages) => ({ packages });

describe('applyCdnFallback', () => {
	it('keeps locally present wheels and rewrites missing ones to the CDN', () => {
		const lock = makeLock({
			numpy: { file_name: 'numpy-2.4.3-pyemscripten_2026_0_wasm32.whl' },
			networkx: { file_name: 'networkx-3.6.1-py3-none-any.whl' }
		});

		const result = applyCdnFallback(
			lock,
			['numpy-2.4.3-pyemscripten_2026_0_wasm32.whl'],
			'314.0.3'
		);

		expect(result.rewritten).toBe(1);
		expect(lock.packages.numpy.file_name).toBe('numpy-2.4.3-pyemscripten_2026_0_wasm32.whl');
		expect(lock.packages.networkx.file_name).toBe(
			`${pyodideCdnBase('314.0.3')}networkx-3.6.1-py3-none-any.whl`
		);
	});

	it('leaves already-remote file names untouched', () => {
		const lock = makeLock({
			numpy: { file_name: 'https://cdn.jsdelivr.net/pyodide/v314.0.3/full/numpy-2.4.3.whl' }
		});

		expect(applyCdnFallback(lock, [], '314.0.3').rewritten).toBe(0);
		expect(lock.packages.numpy.file_name).toBe(
			'https://cdn.jsdelivr.net/pyodide/v314.0.3/full/numpy-2.4.3.whl'
		);
	});

	it('reports entries with no file_name as unresolved', () => {
		const lock = makeLock({ empty: { file_name: null } });

		const result = applyCdnFallback(lock, [], '314.0.3');

		expect(result.rewritten).toBe(0);
		expect(result.unresolved).toEqual(['empty']);
	});
});

describe('findMissingCurated', () => {
	it('matches normalized (hyphen/underscore) lock keys', () => {
		const lock = makeLock({
			'scikit-learn': { file_name: 'scikit_learn.whl' },
			pillow: { file_name: 'pillow.whl' }
		});

		expect(findMissingCurated(lock, ['scikit-learn', 'pillow', 'seaborn'])).toEqual(['seaborn']);
	});
});

describe('findUnresolvedPackages', () => {
	it('flags missing local files but accepts remote URLs', () => {
		const lock = makeLock({
			numpy: { file_name: 'numpy.whl' },
			networkx: { file_name: 'https://cdn.jsdelivr.net/pyodide/v314.0.3/full/networkx.whl' },
			ghost: { file_name: 'ghost.whl' }
		});

		expect(findUnresolvedPackages(lock, ['numpy.whl'])).toEqual(['ghost']);
	});
});

describe('helpers', () => {
	it('baseName strips directories', () => {
		expect(baseName('a/b/c.whl')).toBe('c.whl');
		expect(baseName(null)).toBe(null);
	});

	it('isRemoteFileName only accepts http(s)', () => {
		expect(isRemoteFileName('https://example.com/a.whl')).toBe(true);
		expect(isRemoteFileName('a.whl')).toBe(false);
		expect(isRemoteFileName(null)).toBe(false);
	});
});
