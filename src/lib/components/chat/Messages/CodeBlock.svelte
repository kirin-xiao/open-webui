<script lang="ts">
	import hljs from 'highlight.js';
	import { toast } from 'svelte-sonner';
	import { getContext, onMount, tick } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import { config } from '$lib/stores';

	import { getPyodideRuntime } from '$lib/pyodide/pyodideRuntimePool';
	import { executeCode } from '$lib/apis/utils';
	import {
		copyToClipboard,
		initMermaid,
		renderMermaidDiagram,
		renderVegaVisualization,
		unescapeHtml
	} from '$lib/utils';

	import 'highlight.js/styles/github-dark.min.css';
	import equal from 'fast-deep-equal';

	import CodeEditor from '$lib/components/common/CodeEditor.svelte';
	import DiffBlock from './DiffBlock.svelte';
	import SvgPanZoom from '$lib/components/common/SVGPanZoom.svelte';

	import ChevronUp from '$lib/components/icons/ChevronUp.svelte';
	import ChevronUpDown from '$lib/components/icons/ChevronUpDown.svelte';
	import CommandLine from '$lib/components/icons/CommandLine.svelte';
	import Cube from '$lib/components/icons/Cube.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let id = '';
	export let edit = true;
	export let chatId = '';

	export let onSave = (e) => {};
	export let onUpdate = (e, codeBlockId = '') => {};
	export let onPreview = (e) => {};

	export let save = false;
	export let run = true;
	export let preview = false;
	export let collapsed = false;

	export let token;
	export let lang = '';
	export let code = '';
	export let attributes = {};

	export let className = '';
	export let editorClassName = '';
	export let stickyButtonsClassName = 'top-0';

	let _code = '';
	$: _code = code;
	$: isDiff = ['diff', 'patch'].includes(lang.trim().toLowerCase());
	let editingDiff = false;

	let _token = null;

	let renderHTML = null;
	let renderError = null;

	let highlightedCode = null;
	let executing = false;

	let stdout = null;
	let stderr = null;
	let result = null;
	let files = null;

	let copied = false;
	let saved = false;

	const collapseCodeBlock = () => {
		collapsed = !collapsed;
	};

	const saveCode = () => {
		saved = true;

		code = _code;
		onSave(code);

		setTimeout(() => {
			saved = false;
		}, 1000);
	};

	const copyCode = async () => {
		copied = true;
		await copyToClipboard(_code);

		setTimeout(() => {
			copied = false;
		}, 1000);
	};

	const previewCode = () => {
		onPreview(code);
	};

	const checkPythonCode = (str) => {
		// Check if the string contains typical Python syntax characters
		const pythonSyntax = [
			'def ',
			'else:',
			'elif ',
			'try:',
			'except:',
			'finally:',
			'yield ',
			'lambda ',
			'assert ',
			'nonlocal ',
			'del ',
			'True',
			'False',
			'None',
			' and ',
			' or ',
			' not ',
			' in ',
			' is ',
			' with '
		];

		for (let syntax of pythonSyntax) {
			if (str.includes(syntax)) {
				return true;
			}
		}

		// If none of the above conditions met, it's probably not Python code
		return false;
	};

	const executePython = async (code) => {
		result = null;
		stdout = null;
		stderr = null;

		executing = true;

		if ($config?.code?.engine === 'jupyter') {
			const output = await executeCode(localStorage.token, code).catch((error) => {
				toast.error(`${error}`);
				return null;
			});

			if (output) {
				if (output['stdout']) {
					stdout = output['stdout'];
					const stdoutLines = stdout.split('\n');

					for (const [idx, line] of stdoutLines.entries()) {
						if (line.startsWith('data:image/png;base64')) {
							if (files) {
								files.push({
									type: 'image/png',
									data: line
								});
							} else {
								files = [
									{
										type: 'image/png',
										data: line
									}
								];
							}

							if (stdout.includes(`${line}\n`)) {
								stdout = stdout.replace(`${line}\n`, ``);
							} else if (stdout.includes(`${line}`)) {
								stdout = stdout.replace(`${line}`, ``);
							}
						}
					}
				}

				if (output['result']) {
					result = output['result'];
					const resultLines = result.split('\n');

					for (const [idx, line] of resultLines.entries()) {
						if (line.startsWith('data:image/png;base64')) {
							if (files) {
								files.push({
									type: 'image/png',
									data: line
								});
							} else {
								files = [
									{
										type: 'image/png',
										data: line
									}
								];
							}

							if (result.includes(`${line}\n`)) {
								result = result.replace(`${line}\n`, ``);
							} else if (result.includes(`${line}`)) {
								result = result.replace(`${line}`, ``);
							}
						}
					}
				}

				output['stderr'] && (stderr = output['stderr']);
			}

			executing = false;
		} else {
			executePythonAsWorker(code);
		}
	};

	const LOAD_TIMEOUT_MS = 300000;
	const EXECUTION_TIMEOUT_MS = 60000;

	const executePythonAsWorker = async (code) => {
		// Packages are resolved by the worker from `pyodide-lock.json` via
		// loadPackagesFromImports, so no hardcoded import->package list is needed.

		// Each chat gets its own interpreter so module state cannot leak across
		// chats. The runtime is pooled and reused within the chat, so files
		// written here remain visible in PyodideFileNav.
		const runtime = getPyodideRuntime(chatId);
		const worker = runtime.worker;
		runtime.acquire();

		let settled = false;
		let loadTimeoutId = null;
		let executionTimeoutId = null;

		const finish = (error) => {
			if (settled) return;
			settled = true;

			clearTimeout(loadTimeoutId);
			clearTimeout(executionTimeoutId);
			worker.removeEventListener('message', handler);
			worker.removeEventListener('error', onError);
			runtime.release();

			if (error) {
				stderr = error;
			}

			executing = false;
		};

		// A timed-out run cannot be interrupted in-process, so drop the exact
		// interpreter that hung (not by chat id, which could now resolve to a
		// different runtime). IDBFS files survive because they live in IndexedDB;
		// a fresh interpreter reloads them on the next run.
		const terminateWorker = () => {
			runtime.reset();
		};

		// Loading (Pyodide init + package downloads) gets its own generous budget
		// so cold-cache installs are not misreported as an execution timeout. The
		// execution limit is only started once user code actually begins to run.
		const armLoadTimeout = () => {
			clearTimeout(loadTimeoutId);
			clearTimeout(executionTimeoutId);
			executionTimeoutId = null;
			loadTimeoutId = setTimeout(() => {
				finish('Package Loading Time Limit Exceeded');
				terminateWorker();
			}, LOAD_TIMEOUT_MS);
		};

		const armExecutionTimeout = () => {
			clearTimeout(loadTimeoutId);
			loadTimeoutId = null;
			clearTimeout(executionTimeoutId);
			executionTimeoutId = setTimeout(() => {
				finish('Execution Time Limit Exceeded');
				terminateWorker();
			}, EXECUTION_TIMEOUT_MS);
		};

		const handler = (event) => {
			// Ignore messages from other requests on the shared worker
			if (event.data?.id !== id) return;

			const { id: _id, ...data } = event.data;

			// Status/progress messages are not completion. `executing` starts the
			// execution limit; `loading`/`packages` (e.g. an install-retry) fall
			// back to the load budget.
			if (data.type === 'status') {
				if (data.phase === 'executing') {
					armExecutionTimeout();
				} else {
					armLoadTimeout();
				}
				return;
			}

			console.log('pyodideWorker.onmessage', event);

			console.log(_id, data);

			if (data['stdout']) {
				stdout = data['stdout'];
				const stdoutLines = stdout.split('\n');

				for (const [idx, line] of stdoutLines.entries()) {
					if (line.startsWith('data:image/png;base64')) {
						if (files) {
							files.push({
								type: 'image/png',
								data: line
							});
						} else {
							files = [
								{
									type: 'image/png',
									data: line
								}
							];
						}

						if (stdout.includes(`${line}\n`)) {
							stdout = stdout.replace(`${line}\n`, ``);
						} else if (stdout.includes(`${line}`)) {
							stdout = stdout.replace(`${line}`, ``);
						}
					}
				}
			}

			if (data['result']) {
				result = data['result'];
				const resultLines = result.split('\n');

				for (const [idx, line] of resultLines.entries()) {
					if (line.startsWith('data:image/png;base64')) {
						if (files) {
							files.push({
								type: 'image/png',
								data: line
							});
						} else {
							files = [
								{
									type: 'image/png',
									data: line
								}
							];
						}

						if (result.startsWith(`${line}\n`)) {
							result = result.replace(`${line}\n`, ``);
						} else if (result.startsWith(`${line}`)) {
							result = result.replace(`${line}`, ``);
						}
					}
				}
			}

			data['stderr'] && (stderr = data['stderr']);
			data['result'] && (result = data['result']);

			finish();

			// Signal PyodideFileNav to auto-refresh after execution
			window.dispatchEvent(new Event('pyodide:files'));
		};

		const onError = (event) => {
			console.log('pyodideWorker.onerror', event);
			finish();
		};

		worker.addEventListener('message', handler);
		worker.addEventListener('error', onError);

		try {
			worker.postMessage({
				id: id,
				code: code
			});
		} catch (e) {
			// A throwing postMessage (e.g. DataCloneError) would otherwise skip
			// finish() and pin the runtime's busy counter forever.
			console.error('Failed to post execution to Pyodide runtime:', e);
			finish(e instanceof Error ? e.message : String(e));
			return;
		}

		armLoadTimeout();
	};

	let mermaid = null;
	const renderMermaid = async (code) => {
		if (!mermaid) {
			mermaid = await initMermaid();
		}
		return await renderMermaidDiagram(mermaid, code);
	};

	const render = async () => {
		onUpdate(token, id);
		if (lang === 'mermaid' && (token?.raw ?? '').slice(-4).includes('```')) {
			try {
				renderHTML = await renderMermaid(code);
			} catch (error) {
				console.error('Failed to render mermaid diagram:', error);
				const errorMsg = error instanceof Error ? error.message : String(error);
				renderError = $i18n.t('Failed to render diagram') + `: ${errorMsg}`;
				renderHTML = null;
			}
		} else if (
			(lang === 'vega' || lang === 'vega-lite') &&
			(token?.raw ?? '').slice(-4).includes('```')
		) {
			try {
				renderHTML = await renderVegaVisualization(code, lang);
			} catch (error) {
				console.error('Failed to render Vega visualization:', error);
				const errorMsg = error instanceof Error ? error.message : String(error);
				renderError = $i18n.t('Failed to render visualization') + `: ${errorMsg}`;
				renderHTML = null;
			}
		}
	};

	$: if (token) {
		if (token.text !== _token?.text || token.raw !== _token?.raw) {
			_token = token;
		} else if (!equal(token, _token)) {
			_token = token;
		}
	}

	$: if (_token) {
		render();
	}

	$: if (attributes) {
		onAttributesUpdate();
	}

	const onAttributesUpdate = () => {
		if (attributes?.output) {
			try {
				const output = JSON.parse(unescapeHtml(attributes.output));
				stdout = output.stdout;
				stderr = output.stderr;
				result = output.result;
			} catch (error) {
				console.error('Error:', error);
			}
		}
	};

	onMount(async () => {
		if (token) {
			onUpdate(token, id);
		}
	});
</script>

<div>
	<div
		class="relative {className} flex flex-col rounded-2xl border border-gray-100/30 dark:border-gray-850/30 my-0.5 overflow-clip"
		dir="ltr"
	>
		{#if ['mermaid', 'vega', 'vega-lite'].includes(lang)}
			{#if renderHTML}
				<SvgPanZoom
					className=" rounded-2xl max-h-fit overflow-hidden"
					svg={renderHTML}
					content={_token.text}
				/>
			{:else}
				<div class="p-3">
					{#if renderError}
						<div
							class="flex gap-2.5 border px-4 py-3 border-red-600/10 bg-red-600/10 rounded-2xl mb-2"
						>
							{renderError}
						</div>
					{/if}
					<pre>{code}</pre>
				</div>
			{/if}
		{:else}
			<div
				class="sticky {stickyButtonsClassName} left-0 right-0 py-1.5 px-3.5 gap-2 flex items-center justify-end w-full z-10 text-xs text-black dark:text-white bg-white dark:bg-black"
			>
				<div class="flex-1 truncate">
					<Tooltip content={lang} placement="top-start">
						<span class=" truncate text-ellipsis">
							{lang}
						</span>
					</Tooltip>
				</div>

				<div class="flex items-center gap-0.5 shrink-0">
					{#if isDiff && edit}
						<button
							class="bg-none border-none transition rounded-md px-1.5 py-0.5 bg-white dark:bg-black"
							aria-pressed={editingDiff}
							on:click={() => {
								editingDiff = !editingDiff;
								collapsed = false;
							}}
						>
							{editingDiff ? $i18n.t('Done') : $i18n.t('Edit')}
						</button>
					{/if}
					<button
						class="flex gap-1 items-center bg-none border-none transition rounded-md px-1.5 py-0.5 bg-white dark:bg-black"
						on:click={collapseCodeBlock}
					>
						<div class=" -translate-y-[0.5px]">
							<ChevronUpDown className="size-3" />
						</div>

						<div>
							{collapsed ? $i18n.t('Expand') : $i18n.t('Collapse')}
						</div>
					</button>

					{#if ($config?.features?.enable_code_execution ?? true) && (lang.toLowerCase() === 'python' || lang.toLowerCase() === 'py' || (lang === '' && checkPythonCode(code)))}
						{#if executing}
							<div
								class="run-code-button bg-none border-none p-0.5 cursor-not-allowed bg-white dark:bg-black"
							>
								{$i18n.t('Running')}
							</div>
						{:else if run}
							<button
								class="flex gap-1 items-center run-code-button bg-none border-none transition rounded-md px-1.5 py-0.5 bg-white dark:bg-black"
								on:click={async () => {
									code = _code;
									await tick();
									executePython(code);
								}}
							>
								<div>
									{$i18n.t('Run')}
								</div>
							</button>
						{/if}
					{/if}

					{#if save}
						<button
							class="save-code-button bg-none border-none transition rounded-md px-1.5 py-0.5 bg-white dark:bg-black"
							on:click={saveCode}
						>
							{saved ? $i18n.t('Saved') : $i18n.t('Save')}
						</button>
					{/if}

					<button
						class="copy-code-button bg-none border-none transition rounded-md px-1.5 py-0.5 bg-white dark:bg-black"
						on:click={copyCode}>{copied ? $i18n.t('Copied') : $i18n.t('Copy')}</button
					>

					{#if preview && ['html', 'svg'].includes(lang)}
						<button
							class="flex gap-1 items-center run-code-button bg-none border-none transition rounded-md px-1.5 py-0.5 bg-white dark:bg-black"
							on:click={previewCode}
						>
							<div>
								{$i18n.t('Preview')}
							</div>
						</button>
					{/if}
				</div>
			</div>

			<div
				class="language-{lang} rounded-t-2xl -mt-8 {editorClassName
					? editorClassName
					: executing || stdout || stderr || result
						? ''
						: 'rounded-b-2xl'} overflow-hidden"
			>
				<div class=" pt-6.5 bg-white dark:bg-black"></div>

				{#if !collapsed}
					{#if isDiff && !(edit && editingDiff)}
						<DiffBlock code={_code} />
					{:else if edit}
						<CodeEditor
							value={isDiff ? _code : code}
							{id}
							{lang}
							onSave={() => {
								saveCode();
							}}
							onChange={(value) => {
								_code = value;
							}}
						/>
					{:else}
						<pre
							class=" hljs p-4 px-5 overflow-x-auto"
							style="border-top-left-radius: 0px; border-top-right-radius: 0px; {(executing ||
								stdout ||
								stderr ||
								result) &&
								'border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;'}"><code
								class="language-{lang} rounded-t-none whitespace-pre text-sm"
								>{#if lang && hljs.getLanguage(lang)}{@html hljs.highlight(code, {
										language: lang,
										ignoreIllegals: true
									}).value}{:else}{code}{/if}</code
							></pre>
					{/if}
				{:else}
					<div
						class="bg-white dark:bg-black dark:text-white rounded-b-2xl! pt-1 pb-2 px-4 flex flex-col gap-2 text-xs"
					>
						<span class="text-gray-500 italic">
							{$i18n.t('{{COUNT}} hidden lines', {
								COUNT: (isDiff ? _code : code).split('\n').length
							})}
						</span>
					</div>
				{/if}
			</div>

			{#if !collapsed}
				<div
					id="plt-canvas-{id}"
					class="bg-gray-50 dark:bg-black dark:text-white max-w-full overflow-x-auto scrollbar-hidden"
				/>

				{#if executing || stdout || stderr || result || files}
					<div
						class="bg-gray-50 dark:bg-black dark:text-white rounded-b-2xl! pt-2 pb-3 px-3.5 flex flex-col gap-2"
					>
						{#if executing}
							<div class=" ">
								<div class=" text-gray-500 text-xs mb-1">{$i18n.t('STDOUT/STDERR')}</div>
								<div class="text-sm">{$i18n.t('Running...')}</div>
							</div>
						{:else}
							{#if stdout || stderr}
								<div class=" ">
									<div class=" text-gray-500 text-xs mb-1">{$i18n.t('STDOUT/STDERR')}</div>
									<div
										class="text-sm font-mono whitespace-pre-wrap {stdout?.split('\n')?.length > 100
											? `max-h-96`
											: ''}  overflow-y-auto"
									>
										{stdout || stderr}
									</div>
								</div>
							{/if}
							{#if result || files}
								<div class=" ">
									<div class=" text-gray-500 text-xs mb-1">{$i18n.t('RESULT')}</div>
									{#if result}
										<div class="text-sm">{`${JSON.stringify(result)}`}</div>
									{/if}
									{#if files}
										<div class="flex flex-col gap-2">
											{#each files as file}
												{#if file.type.startsWith('image')}
													<img src={file.data} alt="Output" class=" w-full max-w-[36rem]" />
												{/if}
											{/each}
										</div>
									{/if}
								</div>
							{/if}
						{/if}
					</div>
				{/if}
			{/if}
		{/if}
	</div>
</div>
