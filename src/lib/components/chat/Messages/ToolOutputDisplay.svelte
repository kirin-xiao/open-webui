<script lang="ts">
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	import { safeLinkUrl } from '$lib/utils';
	import { formatToolResult } from '$lib/utils/toolLabels.js';
	import Markdown from './Markdown.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let id = '';
	export let name = '';
	export let result = '';
	/** Parsed tool arguments. `ask_user` results need them for the question text. */
	export let args: Record<string, unknown> | null = null;
	export let chatId = '';
	export let messageId = '';
	export let done = true;

	const PREVIEW_LIMIT = 10000;
	/** Terminal output is capped before the "Show all" affordance would apply. */
	const OUTPUT_CAP = 50000;

	let expanded = false;
	$: if (!result) expanded = false;

	// Classification lives in `toolLabels.js` next to the argument formatting: it is
	// name-dependent, must stay unit-testable, and returns `null` for unknown or
	// mismatched payloads so the pretty-JSON fallback below still applies.
	$: view = formatToolResult(name, result, args);
	$: resultValue = parseJSONValue(result);
	$: fallbackText =
		resultValue !== null && typeof resultValue === 'object'
			? JSON.stringify(resultValue, null, 2)
			: String(resultValue);

	function parseJSONValue(value: unknown): unknown {
		let parsed: unknown = value;
		while (typeof parsed === 'string') {
			try {
				parsed = JSON.parse(parsed);
			} catch {
				break;
			}
		}
		return parsed;
	}

	function hostOf(url: string): string {
		try {
			return new URL(url).host;
		} catch {
			return url;
		}
	}

	function terminalText(section: string): string {
		if (section.length <= OUTPUT_CAP) return section;
		return `${section.slice(0, OUTPUT_CAP)}\n\n[Output truncated...]`;
	}

	/** A stream that is only whitespace is empty for display purposes. */
	function hasText(value: string): boolean {
		return value.trim().length > 0;
	}

	/** `execute_code` stdout carries `![Output Image](...)` once the backend uploads a plot. */
	function stdoutIsMarkdown(text: string): boolean {
		return /!\[[^\]]*\]\([^)]+\)/.test(text);
	}
</script>

{#if view?.kind === 'error'}
	<div
		class="text-xs whitespace-pre-wrap break-words font-mono bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300 rounded-lg p-2"
	>
		{view.text}
	</div>
{:else if view?.kind === 'search'}
	{#if (view.results ?? []).length === 0}
		<div class="text-xs text-gray-500 dark:text-gray-400 px-1">{$i18n.t('No results')}</div>
	{:else}
		<div class="space-y-1">
			{#each view.results ?? [] as item}
				<a
					href={safeLinkUrl(item.link)}
					target="_blank"
					rel="noopener noreferrer"
					class="block rounded-lg px-2 py-1.5 hover:bg-gray-50 dark:hover:bg-gray-850/50 transition no-underline"
				>
					<div class="flex items-baseline gap-2">
						<span class="text-xs font-medium text-gray-800 dark:text-gray-200 line-clamp-1">
							{item.title || item.link}
						</span>
						{#if item.link}
							<span class="shrink-0 text-[0.625rem] text-gray-400 dark:text-gray-500"
								>{hostOf(item.link)}</span
							>
						{/if}
					</div>
					{#if item.snippet}
						<div
							class="mt-0.5 text-[0.6875rem] leading-relaxed text-gray-500 dark:text-gray-400 line-clamp-2"
						>
							{item.snippet}
						</div>
					{/if}
				</a>
			{/each}
		</div>
	{/if}
{:else if view?.kind === 'terminal'}
	{@const stdout = terminalText(view.stdout ?? '')}
	{@const stderr = terminalText(view.stderr ?? '')}
	{@const result = terminalText(view.result ?? '')}
	<div class="space-y-2">
		{#if hasText(stdout)}
			<div>
				<div class="text-[0.6875rem] text-gray-500 dark:text-gray-400 mb-0.5">
					{$i18n.t('STDOUT')}
				</div>
				{#if stdoutIsMarkdown(stdout)}
					<div class="markdown-prose">
						<Markdown {id} {chatId} {messageId} content={stdout} {done} />
					</div>
				{:else}
					<pre
						class="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap break-words font-mono bg-gray-50 dark:bg-gray-900 rounded-lg p-2">{stdout}</pre>
				{/if}
			</div>
		{/if}
		{#if hasText(stderr)}
			<div>
				<div class="text-[0.6875rem] text-gray-500 dark:text-gray-400 mb-0.5">
					{$i18n.t('STDERR')}
				</div>
				<pre
					class="text-xs whitespace-pre-wrap break-words font-mono text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-950/30 rounded-lg p-2">{stderr}</pre>
			</div>
		{/if}
		{#if hasText(result)}
			<div>
				<div class="text-[0.6875rem] text-gray-500 dark:text-gray-400 mb-0.5">
					{$i18n.t('RESULT')}
				</div>
				<pre
					class="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap break-words font-mono bg-gray-50 dark:bg-gray-900 rounded-lg p-2">{result}</pre>
			</div>
		{/if}
		{#if !hasText(stdout) && !hasText(stderr) && !hasText(result)}
			<div class="text-xs text-gray-500 dark:text-gray-400 px-1">{$i18n.t('No output')}</div>
		{/if}
	</div>
{:else if view?.kind === 'markdown'}
	<div class="markdown-prose max-h-[32rem] overflow-y-auto">
		<Markdown {id} {chatId} {messageId} content={view.text ?? ''} {done} />
	</div>
{:else if view?.kind === 'qa'}
	<div class="space-y-2">
		{#if view.status === 'cancelled'}
			<div class="text-xs text-gray-500 dark:text-gray-400 px-1">
				{$i18n.t('Question cancelled')}
			</div>
		{:else if (view.questions ?? []).length === 0}
			<div class="text-xs text-gray-500 dark:text-gray-400 px-1">{$i18n.t('No answer')}</div>
		{:else}
			{#each view.questions ?? [] as question}
				<div class="space-y-0.5 border-l-2 border-gray-100 pl-2 dark:border-gray-850/50">
					{#if question.header}
						<div class="text-xs font-medium text-gray-800 dark:text-gray-200">
							{question.header}
						</div>
					{/if}
					{#if question.question && question.question !== question.header}
						<div class="text-xs text-gray-500 dark:text-gray-400">{question.question}</div>
					{/if}
					<div class="text-xs text-gray-600 dark:text-gray-300">
						{#if question.answer}
							{question.answer}
							{#if question.isOther}
								<span class="text-gray-400 dark:text-gray-500">({$i18n.t('Other')})</span>
							{/if}
						{:else}
							<span class="text-gray-400 dark:text-gray-500">{$i18n.t('No answer')}</span>
						{/if}
					</div>
				</div>
			{/each}
		{/if}
	</div>
{:else if resultValue !== null && typeof resultValue === 'object'}
	<pre
		class="text-xs text-gray-600 dark:text-gray-300 whitespace-pre font-mono bg-gray-50 dark:bg-gray-900 rounded-lg p-2 overflow-x-auto">{fallbackText}</pre>
{:else}
	{@const isTruncated = fallbackText.length > PREVIEW_LIMIT && !expanded}
	<pre
		class="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap break-words font-mono">{isTruncated
			? fallbackText.slice(0, PREVIEW_LIMIT)
			: fallbackText}</pre>
	{#if isTruncated}
		<button
			class="mt-1 text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition"
			on:click|stopPropagation={() => {
				expanded = true;
			}}
		>
			{$i18n.t('Show all ({{COUNT}} characters)', {
				COUNT: fallbackText.length.toLocaleString()
			})}
		</button>
	{/if}
{/if}
