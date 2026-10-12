<script lang="ts">
	import { getContext } from 'svelte';
	import type { ContextCheckpointRecord } from '$lib/utils/contextCompaction';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import CheckCircle from '$lib/components/icons/CheckCircle.svelte';
	import ChevronDown from '$lib/components/icons/ChevronDown.svelte';
	import ChevronRight from '$lib/components/icons/ChevronRight.svelte';
	import ArrowPath from '$lib/components/icons/ArrowPath.svelte';
	import XMark from '$lib/components/icons/XMark.svelte';

	const i18n: any = getContext('i18n');

	export let checkpoint: ContextCheckpointRecord | null = null;
	export let status: string | null = null;
	export let usage: {
		threshold?: number | null;
		percent?: number | null;
		tokens?: number | null;
		estimated_tokens?: number | null;
	} | null = null;
	export let messageId: string | null = null;
	export let model: string | null = null;
	export let disabled = false;
	export let onUndo: (messageId: string) => void = () => {};
	export let onRegenerate: () => void = () => {};

	let expanded = false;

	$: effectiveStatus = status ?? checkpoint?.status ?? 'completed';
	$: isRunning = effectiveStatus === 'running';
	$: isFailed = effectiveStatus === 'failed';
	$: summary = checkpoint?.summary ?? '';
	$: hasSummary = Boolean(String(summary).trim());
	$: contextHasThreshold = Number(usage?.threshold) > 0;
	$: contextPercent = contextHasThreshold ? Math.max(0, Math.round(usage?.percent ?? 0)) : null;
	$: contextTokens = usage?.tokens ?? usage?.estimated_tokens ?? null;
	$: usageLine = (() => {
		if (contextHasThreshold && contextPercent !== null) {
			return `${contextPercent}% · ${formatCount(contextTokens)}/${formatCount(usage?.threshold)} ${$i18n.t('tokens')}`;
		}
		if (contextTokens !== null) {
			return `${formatCount(contextTokens)} ${$i18n.t('tokens')}`;
		}
		return null;
	})();

	const formatCount = (value: number | null | undefined) => {
		if (value === null || value === undefined || Number.isNaN(Number(value))) {
			return '—';
		}
		const count = Number(value);
		if (count >= 1_000_000) {
			return `${(count / 1_000_000).toFixed(1)}M`;
		}
		if (count >= 1000) {
			return `${(count / 1000).toFixed(1)}k`;
		}
		return `${count}`;
	};
</script>

<div
	role="listitem"
	class="flex flex-col justify-between px-3.5 mb-3 w-full max-w-[58rem] mx-auto rounded-lg group message-listitem"
>
	<div
		class="rounded-lg border border-gray-100 bg-gray-50/60 px-3 py-2 text-xs dark:border-white/[0.06] dark:bg-white/[0.02]"
	>
		<div class="flex items-center gap-2">
			<span
				class="flex size-4 shrink-0 items-center justify-center text-gray-400 dark:text-gray-500"
			>
				{#if isRunning}
					<Spinner className="size-3.5" />
				{:else if isFailed}
					<XMark className="size-3.5" strokeWidth="1.8" />
				{:else}
					<CheckCircle className="size-3.5" strokeWidth="1.8" />
				{/if}
			</span>

			<button
				type="button"
				class="flex min-w-0 flex-1 items-center gap-2 text-left"
				aria-expanded={expanded}
				on:click={() => {
					if (hasSummary) {
						expanded = !expanded;
					}
				}}
			>
				<span class="shrink-0 font-medium text-gray-700 dark:text-gray-200">
					{$i18n.t('Context checkpoint')}
				</span>
				{#if isRunning}
					<span class="app-muted truncate">{$i18n.t('Compacting context...')}</span>
				{:else if isFailed}
					<span class="truncate text-red-500/80">
						{$i18n.t('Context compaction failed')}
					</span>
				{:else if hasSummary}
					<span class="app-muted truncate">
						{expanded ? $i18n.t('Hide summary') : $i18n.t('View summary')}
					</span>
					<span class="shrink-0 text-gray-400 dark:text-gray-600">
						{#if expanded}
							<ChevronDown className="size-3" strokeWidth="1.8" />
						{:else}
							<ChevronRight className="size-3" strokeWidth="1.8" />
						{/if}
					</span>
				{/if}
			</button>

			{#if usageLine}
				<span class="shrink-0 font-mono text-[0.625rem] text-gray-400 dark:text-gray-600">
					{usageLine}
				</span>
			{/if}
		</div>

		{#if expanded && hasSummary}
			<div
				class="mt-2 max-h-72 overflow-y-auto whitespace-pre-wrap break-words rounded-md bg-white/60 p-2 text-[0.6875rem] leading-relaxed text-gray-600 scrollbar-thin dark:bg-black/20 dark:text-gray-300"
			>
				{summary}
			</div>
		{/if}

		{#if !isRunning}
			<div class="mt-2 flex items-center gap-3 text-[0.625rem]">
				<button
					type="button"
					class="app-muted transition-colors hover:text-gray-900 dark:hover:text-white disabled:opacity-40"
					{disabled}
					on:click={() => onRegenerate()}
				>
					<span class="inline-flex items-center gap-1">
						<ArrowPath className="size-3" strokeWidth="1.8" />
						{$i18n.t('Regenerate summary')}
					</span>
				</button>
				{#if messageId && hasSummary}
					<button
						type="button"
						class="app-muted transition-colors hover:text-red-500 dark:hover:text-red-400 disabled:opacity-40"
						{disabled}
						on:click={() => onUndo(messageId)}
					>
						{$i18n.t('Undo compaction')}
					</button>
				{/if}
				{#if model}
					<span class="truncate text-gray-400 dark:text-gray-600">
						{$i18n.t('Compaction model')}: {model}
					</span>
				{/if}
			</div>
		{/if}
	</div>
</div>
