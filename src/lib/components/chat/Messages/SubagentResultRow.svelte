<script lang="ts">
	import { getContext } from 'svelte';
	import type { Readable } from 'svelte/store';

	import ChevronDown from '$lib/components/icons/ChevronDown.svelte';

	type SubagentResult = {
		// Current shape.
		sessionID?: string;
		childID?: string;
		childIDs?: string[];
		description?: string;
		status?: string;
		state?: string;
		output?: string;
		// Persisted legacy handle.
		delegation_id?: string;
		delegation_ids?: string[];
		subagent_chat_id?: string;
		subagent_chat_ids?: string[];
	};

	const i18n = getContext<Readable<{ t: (value: string) => string }>>('i18n');

	export let content: string;
	export let result: SubagentResult;
	/** When false (e.g. a read-only shared chat) the child id is plain text, not a link. */
	export let linkable = true;

	/** First non-empty string, ignoring whitespace-only values. */
	const firstString = (...values: unknown[]): string => {
		for (const value of values) {
			if (typeof value === 'string' && value.trim()) return value.trim();
		}
		return '';
	};

	/** First resolved child id from a mix of scalar and list fields. */
	const firstId = (...values: Array<string | string[] | undefined>): string => {
		for (const value of values) {
			if (typeof value === 'string' && value.trim()) return value.trim();
			if (Array.isArray(value)) {
				const found = value.find((entry) => typeof entry === 'string' && entry.trim());
				if (found) return found.trim();
			}
		}
		return '';
	};

	// Persisted completions carry the child id and state in the result envelope as
	// well as `meta`, so read the envelope attributes as a fallback. Read from the
	// prop (rather than a one-shot top-level const) so a re-rendered message updates.
	const envelopeAttr = (value: string, name: string): string => {
		const envelope = (value ?? '').match(/<subagent\b[^>]*>/i)?.[0] ?? '';
		return envelope.match(new RegExp(`${name}\\s*=\\s*"([^"]*)"`, 'i'))?.[1] ?? '';
	};

	let expanded = false;
	$: delegationIds = Array.isArray(result?.delegation_ids) ? result.delegation_ids : [];
	$: delegationLabel =
		result?.delegation_id ?? (delegationIds.length > 1 ? `${delegationIds.length} tasks` : '');
	$: childId = firstId(
		result?.sessionID,
		result?.childID,
		result?.subagent_chat_id,
		result?.childIDs,
		result?.subagent_chat_ids,
		envelopeAttr(content, 'sessionID')
	);
	$: status = firstString(result?.status, result?.state, envelopeAttr(content, 'state'));
	$: raw = (content ?? '').trim() || firstString(result?.output);
	$: body = raw.replace(/<\/?subagent\b[^>]*>/gi, '').trim();
	$: summary = (() => {
		const description = firstString(result?.description);
		if (description) {
			return description.length > 96 ? `${description.slice(0, 96)}...` : description;
		}
		const line = body
			.split('\n')
			.map((value) => value.trim())
			.find((value) => value && !value.startsWith('['));
		if (!line) return delegationLabel;
		return line.length > 96 ? `${line.slice(0, 96)}...` : line;
	})();
</script>

<div class="w-full min-w-0 pb-1">
	<div class="w-full min-w-0 flex items-center gap-2 text-gray-500 dark:text-gray-400">
		<span class="text-[0.75rem] font-normal shrink-0">
			{$i18n.t('Background subagent finished')}
		</span>
		{#if summary}
			{#if childId && linkable}
				<a
					href={`/c/${childId}`}
					title={childId}
					class="text-[0.75rem] truncate min-w-0 flex-1 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
				>
					{summary}
				</a>
			{:else}
				<span class="text-[0.75rem] truncate min-w-0 flex-1">{summary}</span>
			{/if}
		{/if}
		{#if status && status !== 'completed'}
			<span class="text-[0.6875rem] text-amber-600 dark:text-amber-500 shrink-0">
				{status}
			</span>
		{/if}
		{#if delegationLabel}
			<span
				class="hidden sm:inline text-[0.6875rem] font-mono text-gray-400 dark:text-gray-600 shrink-0"
			>
				{delegationLabel}
			</span>
		{/if}
		<button
			type="button"
			class="shrink-0 text-gray-400 dark:text-gray-600 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
			aria-expanded={expanded}
			aria-label={$i18n.t(expanded ? 'Collapse' : 'Expand')}
			on:click={() => (expanded = !expanded)}
		>
			<ChevronDown
				className="size-3 text-gray-400 dark:text-gray-600 shrink-0 transition-transform duration-150 {expanded
					? 'rotate-180'
					: ''}"
			/>
		</button>
	</div>
	{#if expanded}
		<div
			class="mt-2 ml-3 border-l border-gray-100 dark:border-white/8 pl-3 text-[0.78125rem] leading-relaxed text-gray-600 dark:text-gray-400 whitespace-pre-wrap break-words"
		>
			{body || raw}
		</div>
	{/if}
</div>
