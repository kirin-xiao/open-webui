<script lang="ts">
	import { getContext } from 'svelte';
	import type { Readable } from 'svelte/store';

	import { settings } from '$lib/stores';

	type I18n = Readable<{ t: (value: string) => string }>;

	const i18n = getContext<I18n>('i18n');

	export let text = '';
	/** Waiting for the next model boundary. */
	export let pending = false;
</script>

<div class="w-full min-w-0 pb-1">
	<div class="flex {($settings?.chatBubble ?? true) ? 'justify-end' : 'w-full'}">
		<div
			class="rounded-3xl {($settings?.chatBubble ?? true)
				? 'max-w-[90%] px-4 py-1.5 bg-gray-50 dark:bg-gray-850'
				: 'w-full'} {pending ? 'opacity-60' : ''}"
		>
			<div
				class="whitespace-pre-wrap break-words text-[0.9375rem]"
				dir={$settings?.chatDirection ?? 'auto'}
			>
				{text}
			</div>
			{#if pending}
				<div class="mt-0.5 text-[0.6875rem] text-gray-400 dark:text-gray-500">
					{$i18n.t('Pending')}
				</div>
			{/if}
		</div>
	</div>
</div>
