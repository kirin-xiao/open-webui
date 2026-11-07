<script lang="ts">
	import { getContext } from 'svelte';
	import type { Readable } from 'svelte/store';

	import SubagentResultRow from './SubagentResultRow.svelte';
	import { isSubagentSteer, type SteerEntry } from './steers';

	const i18n = getContext<Readable<{ t: (value: string) => string }>>('i18n');

	export let steers: SteerEntry[] = [];
	/** Show the presentation-only note for a response generated before a steer. */
	export let showPreSteerNote = false;
	export let compactPreview = false;
	/** Shared/read-only chats render the child id as plain text. */
	export let linkable = true;

	$: visibleSteers = steers.filter((steer) => steer?.content?.trim());
</script>

{#if visibleSteers.length > 0 || showPreSteerNote}
	<div class="w-full min-w-0 {compactPreview ? '' : 'mt-1'} space-y-1" aria-live="polite">
		{#if showPreSteerNote}
			<p class="text-[0.75rem] leading-snug text-amber-600 dark:text-amber-500">
				{$i18n.t(
					'This response was generated before your steering message. A new response is coming.'
				)}
			</p>
		{/if}
		{#each visibleSteers as steer (steer.id)}
			{#if isSubagentSteer(steer)}
				<!-- A consumed child completion is a result, not a user steer: keep
				     the same row legacy chats use, including the child-chat link. -->
				<SubagentResultRow content={steer.content} result={{ source: 'subagent' }} {linkable} />
			{:else}
				<div
					class="flex min-w-0 items-start gap-1.5 rounded-lg border border-gray-100 bg-gray-50/60 px-2 py-1 text-[0.75rem] leading-snug text-gray-600 dark:border-white/8 dark:bg-white/[0.03] dark:text-gray-400"
				>
					<span class="shrink-0 font-medium text-gray-500 dark:text-gray-400">
						{steer.status === 'pending' ? $i18n.t('Steering queued') : $i18n.t('Steering message')}
					</span>
					<span class="min-w-0 whitespace-pre-wrap break-words">{steer.content}</span>
				</div>
			{/if}
		{/each}
	</div>
{/if}
