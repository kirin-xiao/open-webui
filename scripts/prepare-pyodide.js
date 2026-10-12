// Curated set of wheels that are downloaded and vendored into static/pyodide/
// for offline use. Only list packages that exist in the Pyodide distribution
// (or in `pypiPackages` below); `seaborn`/`openpyxl` are intentionally absent
// because the Pyodide lock does not contain them (and openpyxl also needs
// `et_xmlfile`). The runtime resolves such packages from PyPI on demand.
const packages = [
	'micropip',
	'packaging',
	'requests',
	'beautifulsoup4',
	'numpy',
	'pandas',
	'matplotlib',
	'scikit-learn',
	'scipy',
	'regex',
	'sympy',
	'tiktoken',
	'pytz',
	'black',
	'openai',
	'lxml'
];

// Pure-Python packages whose wheels must be downloaded from PyPI and saved into
// static/pyodide/ so that the browser can install them offline via micropip.
// Packages already provided by the Pyodide distribution (click, platformdirs,
// typing_extensions, etc.) do NOT need to be listed here.
// Spell them canonically (dashed): that is the only form pyodide resolves lock entries by.
const pypiPackages = [
	'black',
	'pathspec',
	'mypy-extensions',
	'pytokens',
	'openpyxl',
	'et-xmlfile',
	'seaborn',
	'python-pptx',
	'python-docx',
	'xlsxwriter'
];

const pypiDepends = {
	black: ['click', 'mypy-extensions', 'packaging', 'pathspec', 'platformdirs', 'pytokens'],
	openpyxl: ['et-xmlfile'],
	seaborn: ['matplotlib', 'numpy', 'pandas'],
	'python-pptx': ['lxml', 'pillow', 'xlsxwriter', 'typing-extensions'],
	'python-docx': ['lxml', 'typing-extensions']
};

// Fonts vendored into static/pyodide/fonts/ so matplotlib can render non-Latin
// labels. The bundled DejaVu/STIX/Computer Modern faces have no CJK coverage, so
// Chinese, Japanese, and Korean text would otherwise render as missing-glyph
// boxes ("tofu"). matplotlib's AGG backend can only use fonts registered with
// its own FontManager -- there is no browser font fallback inside wasm -- so the
// file must be present locally.
//
// Noto Sans SC covers Simplified Chinese, Traditional Chinese, and Japanese.
// It contains no Hangul, so Korean is not covered; use the full pan-CJK face
// (Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf) if that is needed.
const fonts = [
	{
		name: 'NotoSansSC-Regular.otf',
		// Pinned tag: jsDelivr resolves this tag even though `main` also works.
		url: 'https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@Sans2.004/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf'
	}
];
const fontsDir = 'static/pyodide/fonts';

import { loadPyodide } from 'pyodide';
import { setGlobalDispatcher, ProxyAgent } from 'undici';
import { writeFile, readFile, copyFile, readdir, rmdir, access, mkdir, rm } from 'fs/promises';
import {
	applyCdnFallback,
	findMissingCurated,
	findUnresolvedPackages
} from './pyodide-lock-utils.js';

/**
 * Loading network proxy configurations from the environment variables.
 * And the proxy config with lowercase name has the highest priority to use.
 */
function initNetworkProxyFromEnv() {
	// we assume all subsequent requests in this script are HTTPS:
	// https://cdn.jsdelivr.net
	// https://pypi.org
	// https://files.pythonhosted.org
	const allProxy = process.env.all_proxy || process.env.ALL_PROXY;
	const httpsProxy = process.env.https_proxy || process.env.HTTPS_PROXY;
	const httpProxy = process.env.http_proxy || process.env.HTTP_PROXY;
	const preferedProxy = httpsProxy || allProxy || httpProxy;
	/**
	 * use only http(s) proxy because socks5 proxy is not supported currently:
	 * @see https://github.com/nodejs/undici/issues/2224
	 */
	if (!preferedProxy || !preferedProxy.startsWith('http')) return;
	let preferedProxyURL;
	try {
		preferedProxyURL = new URL(preferedProxy).toString();
	} catch {
		console.warn(`Invalid network proxy URL: "${preferedProxy}"`);
		return;
	}
	const dispatcher = new ProxyAgent({ uri: preferedProxyURL });
	setGlobalDispatcher(dispatcher);
	console.log(`Initialized network proxy "${preferedProxy}" from env`);
}

async function downloadPackages() {
	console.log('Setting up pyodide + micropip');

	let pyodide;
	try {
		pyodide = await loadPyodide({
			packageCacheDir: 'static/pyodide'
		});
	} catch (err) {
		console.error('Failed to load Pyodide:', err);
		return;
	}

	const packageJson = JSON.parse(await readFile('package.json'));
	const pyodideVersion = packageJson.dependencies.pyodide.replace('^', '');

	try {
		const pyodidePackageJson = JSON.parse(await readFile('static/pyodide/package.json'));
		const pyodidePackageVersion = pyodidePackageJson.version.replace('^', '');

		if (pyodideVersion !== pyodidePackageVersion) {
			console.log('Pyodide version mismatch, removing static/pyodide directory');
			await rmdir('static/pyodide', { recursive: true });
		}
	} catch (err) {
		console.log('Pyodide package not found, proceeding with download.', err);
	}

	try {
		console.log('Loading micropip package');
		await pyodide.loadPackage('micropip');

		const micropip = pyodide.pyimport('micropip');
		console.log('Downloading Pyodide packages:', packages);

		try {
			for (const pkg of packages) {
				console.log(`Installing package: ${pkg}`);
				await micropip.install(pkg);
			}
		} catch (err) {
			console.error('Package installation failed:', err);
			return;
		}

		console.log('Pyodide packages downloaded, freezing into lock file');

		try {
			const lockFile = await micropip.freeze();
			await writeFile('static/pyodide/pyodide-lock.json', lockFile);
		} catch (err) {
			console.error('Failed to write lock file:', err);
		}
	} catch (err) {
		console.error('Failed to load or install micropip:', err);
	}
}

async function copyPyodide() {
	console.log('Copying Pyodide files into static directory');
	// Copy all files from node_modules/pyodide to static/pyodide
	for await (const entry of await readdir('node_modules/pyodide')) {
		await copyFile(`node_modules/pyodide/${entry}`, `static/pyodide/${entry}`);
	}
}

/**
 * Download pure-Python wheels from PyPI and save them into static/pyodide/.
 * Also injects entries into pyodide-lock.json so that micropip resolves these
 * packages from the local server instead of fetching them from the internet.
 */
async function downloadPyPIWheels() {
	const lockPath = 'static/pyodide/pyodide-lock.json';
	let lockData;
	try {
		lockData = JSON.parse(await readFile(lockPath, 'utf-8'));
	} catch {
		console.warn('Could not read pyodide-lock.json, skipping PyPI wheel download');
		return;
	}

	for (const pkg of pypiPackages) {
		console.log(`Fetching PyPI metadata for: ${pkg}`);
		const res = await fetch(`https://pypi.org/pypi/${pkg}/json`);
		if (!res.ok) {
			console.error(`Failed to fetch PyPI metadata for ${pkg}: ${res.status}`);
			continue;
		}
		const meta = await res.json();
		const version = meta.info.version;
		const files = meta.urls || [];
		// Find the pure-Python wheel (py3-none-any)
		const wheel = files.find(
			(f) => f.filename.endsWith('.whl') && f.filename.includes('py3-none-any')
		);
		if (!wheel) {
			console.warn(`No pure-Python wheel found for ${pkg}==${version}, skipping`);
			continue;
		}
		const dest = `static/pyodide/${wheel.filename}`;
		// Download wheel if not already present
		try {
			await access(dest);
			console.log(`  Already exists: ${wheel.filename}`);
		} catch {
			console.log(`  Downloading: ${wheel.filename}`);
			const wheelRes = await fetch(wheel.url);
			if (!wheelRes.ok) {
				console.error(`  Failed to download ${wheel.filename}: ${wheelRes.status}`);
				continue;
			}
			const buffer = Buffer.from(await wheelRes.arrayBuffer());
			await writeFile(dest, buffer);
			console.log(`  Saved: ${dest} (${buffer.length} bytes)`);
		}

		// Inject into pyodide-lock.json so micropip resolves locally
		if (!lockData.packages[pkg]) {
			lockData.packages[pkg] = {
				name: pkg,
				version: version,
				file_name: wheel.filename,
				install_dir: 'site',
				sha256: wheel.digests?.sha256 || '',
				package_type: 'package',
				imports: [pkg.replace(/-/g, '_')],
				depends: pypiDepends[pkg] || []
			};
			console.log(`  Added ${pkg}==${version} to pyodide-lock.json`);
		}
	}

	await writeFile(lockPath, JSON.stringify(lockData, null, 2));
	console.log('Updated pyodide-lock.json with PyPI packages');
}

/**
 * Download the vendored matplotlib fonts into static/pyodide/fonts/.
 *
 * Best-effort: a failed download warns and continues, so a network hiccup never
 * breaks a build. The runtime treats the font as optional too.
 */
async function downloadFonts() {
	await mkdir(fontsDir, { recursive: true });

	for (const font of fonts) {
		const dest = `${fontsDir}/${font.name}`;
		try {
			await access(dest);
			console.log(`  Already exists: ${font.name}`);
			continue;
		} catch {
			// Not cached yet; fall through to the download.
		}

		console.log(`  Downloading: ${font.name}`);
		try {
			const res = await fetch(font.url);
			if (!res.ok) {
				console.warn(
					`  Failed to download ${font.name}: ${res.status}; CJK plot labels will render as boxes`
				);
				continue;
			}
			const buffer = Buffer.from(await res.arrayBuffer());
			await writeFile(dest, buffer);
			console.log(`  Saved: ${dest} (${buffer.length} bytes)`);
		} catch (err) {
			console.warn(
				`  Failed to download ${font.name}: ${err}; CJK plot labels will render as boxes`
			);
		}
	}
}

/**
 * Make every lock entry resolvable in the default (non-slim) build: wheels that
 * were not vendored into static/pyodide/ are served from the jsDelivr CDN
 * instead of 404-ing against the local mount. Also reports curated packages
 * that are missing from the lock entirely.
 */
async function finalizeLockWithCdnFallback() {
	const { version } = JSON.parse(await readFile('node_modules/pyodide/package.json', 'utf-8'));
	const lockPath = 'static/pyodide/pyodide-lock.json';
	const lockData = JSON.parse(await readFile(lockPath, 'utf-8'));
	const presentFileNames = await readdir('static/pyodide');

	const { rewritten } = applyCdnFallback(lockData, presentFileNames, version);
	const missingCurated = findMissingCurated(lockData, packages);
	if (missingCurated.length > 0) {
		console.warn(
			`[pyodide] curated packages missing from the lockfile (will be resolved from PyPI on demand): ${missingCurated.join(', ')}`
		);
	}
	const unresolved = findUnresolvedPackages(lockData, presentFileNames);
	if (unresolved.length > 0) {
		console.warn(
			`[pyodide] lock entries still unresolvable after CDN fallback: ${unresolved.join(', ')}`
		);
	}

	await writeFile(lockPath, JSON.stringify(lockData, null, 2));
	console.log(
		`[pyodide] rewrote ${rewritten} non-vendored lock entries to the jsDelivr CDN (${presentFileNames.length} local files present)`
	);
}

initNetworkProxyFromEnv();
if (process.env.USE_SLIM === 'true') {
	// Rebuild generated assets so a previous full build cannot leave bundled wheels behind.
	await rm('static/pyodide', { recursive: true, force: true });
	await mkdir('static/pyodide', { recursive: true });
	await copyPyodide();

	const { version } = JSON.parse(await readFile('node_modules/pyodide/package.json', 'utf-8'));
	const lockPath = 'static/pyodide/pyodide-lock.json';
	const lockData = JSON.parse(await readFile(lockPath, 'utf-8'));
	for (const pkg of Object.values(lockData.packages)) {
		pkg.file_name = new URL(
			pkg.file_name,
			`https://cdn.jsdelivr.net/pyodide/v${version}/full/`
		).href;
	}
	await writeFile(lockPath, JSON.stringify(lockData, null, 2));
} else {
	await downloadPackages();
	await copyPyodide();
	await downloadPyPIWheels();
	await finalizeLockWithCdnFallback();
	// After downloadPackages(), whose version-mismatch cleanup removes the whole
	// static/pyodide directory (fonts included).
	await downloadFonts();
}
