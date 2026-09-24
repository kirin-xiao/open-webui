<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getModels } from '$lib/apis';
	import { getSubagentsConfig, setSubagentsConfig } from '$lib/apis/configs';
	import { getBaseModels } from '$lib/apis/models';
	import SettingsSelect from '$lib/components/common/SettingsSelect.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Switch from '$lib/components/common/Switch.svelte';

	const i18n = getContext('i18n');

	type ModelOption = {
		id: string;
		name: string;
		connection_type?: string;
	};

	let loading = true;
	let saving = false;
	let enabled = false;
	let model = '';
	let maxIterations = 30;
	let maxOutput = 30000;
	let systemPrompt = '';

	let workspaceModels: ModelOption[] = [];
	let baseModels: ModelOption[] = [];
	let models: ModelOption[] | null = null;
	$: modelOptions = models ?? [];

	onMount(async () => {
		try {
			const config = await getSubagentsConfig(localStorage.token);
			enabled = config?.ENABLE_SUBAGENTS ?? false;
			model = config?.SUBAGENTS_MODEL ?? '';
			maxIterations = Number(config?.SUBAGENTS_MAX_ITERATIONS) || 30;
			maxOutput = Number(config?.SUBAGENTS_MAX_OUTPUT) || 30000;
			systemPrompt = config?.SUBAGENTS_SYSTEM_PROMPT ?? '';
		} catch (error) {
			toast.error(`${error}`);
		} finally {
			loading = false;
		}

		try {
			workspaceModels = (await getBaseModels(localStorage.token)) ?? [];
			baseModels = (await getModels(localStorage.token, null, false)) ?? [];

			models = baseModels.map((m) => {
				const workspaceModel = workspaceModels.find((wm) => wm.id === m.id);

				if (workspaceModel) {
					return {
						...m,
						...workspaceModel
					};
				} else {
					return m;
				}
			});
		} catch (error) {
			console.error('Failed to load subagent models:', error);
			models = [];
		}
	});

	const save = async () => {
		saving = true;
		try {
			await setSubagentsConfig(localStorage.token, {
				ENABLE_SUBAGENTS: enabled,
				SUBAGENTS_MODEL: model,
				SUBAGENTS_MAX_ITERATIONS: maxIterations,
				SUBAGENTS_MAX_OUTPUT: maxOutput,
				SUBAGENTS_SYSTEM_PROMPT: systemPrompt
			});
			toast.success($i18n.t('Settings saved successfully!'));
		} catch (error) {
			toast.error(`${error}`);
		} finally {
			saving = false;
		}
	};
</script>

<form class="flex h-full flex-col justify-between text-sm" on:submit|preventDefault={save}>
	<h2 class="text-sm font-medium text-gray-900 dark:text-white mb-4">
		{$i18n.t('settings.admin.subagents.title')}
	</h2>

	<div class="flex-1 min-h-0 overflow-y-auto scrollbar-hover pr-1.5">
		{#if loading}
			<div class="flex justify-center py-8"><Spinner className="size-6" /></div>
		{:else}
			<div class="flex flex-col gap-2.5">
				<label class="flex cursor-pointer items-center justify-between">
					<span class="text-xs text-gray-600 dark:text-gray-400">
						{$i18n.t('settings.admin.subagents.enableSubAgents.label')}
					</span>
					<Switch bind:state={enabled} />
				</label>
				<p class="-mt-1 text-[0.6875rem] text-gray-400 dark:text-gray-600">
					{$i18n.t(
						'Allow the AI to delegate tasks to subagents. Each subagent creates a real chat with full tool access. Uses additional LLM calls.'
					)}
				</p>

				{#if enabled}
					<div>
						<label class="text-xs text-gray-600 dark:text-gray-400" for="sa-model">
							{$i18n.t('settings.admin.subagents.model.label')}
						</label>
						<div class="mt-1">
							<SettingsSelect
								id="sa-model"
								bind:value={model}
								className="w-full"
								placeholder={$i18n.t('Select a model')}
							>
								<option value="" selected>{$i18n.t('Current Model')}</option>
								{#each modelOptions as m}
									<option value={m.id} class="bg-gray-100 dark:bg-gray-700">
										{m.name}
										{m?.connection_type === 'local' ? `(${$i18n.t('Local')})` : ''}
									</option>
								{/each}
							</SettingsSelect>
						</div>
					</div>

					<div>
						<label class="text-xs text-gray-600 dark:text-gray-400" for="sa-iterations">
							{$i18n.t('settings.admin.subagents.maxIterations.label')}
						</label>
						<div class="mt-1 flex items-center gap-1.5">
							<input
								id="sa-iterations"
								type="number"
								bind:value={maxIterations}
								min="1"
								max="100"
								class="h-7 w-16 rounded-lg border border-gray-100/50 bg-gray-50/40 px-2 text-xs text-gray-700 outline-hidden transition-colors focus:border-blue-400 dark:border-white/[0.04] dark:bg-white/[0.03] dark:text-gray-300 dark:focus:border-blue-500"
							/>
							<span class="text-[0.6875rem] text-gray-400 dark:text-gray-600">
								{$i18n.t('tool loops per subagent')}
							</span>
						</div>
					</div>

					<div>
						<label class="text-xs text-gray-600 dark:text-gray-400" for="sa-output">
							{$i18n.t('settings.admin.subagents.maxOutput.label')}
						</label>
						<div class="mt-1 flex items-center gap-1.5">
							<input
								id="sa-output"
								type="number"
								bind:value={maxOutput}
								min="1000"
								max="100000"
								step="1000"
								class="h-7 w-20 rounded-lg border border-gray-100/50 bg-gray-50/40 px-2 text-xs text-gray-700 outline-hidden transition-colors focus:border-blue-400 dark:border-white/[0.04] dark:bg-white/[0.03] dark:text-gray-300 dark:focus:border-blue-500"
							/>
							<span class="text-[0.6875rem] text-gray-400 dark:text-gray-600">chars</span>
						</div>
					</div>

					<div>
						<label class="text-xs text-gray-600 dark:text-gray-400" for="sa-prompt">
							{$i18n.t('settings.admin.subagents.systemPrompt.label')}
						</label>
						<textarea
							id="sa-prompt"
							bind:value={systemPrompt}
							rows="4"
							placeholder={$i18n.t('You are a subagent...')}
							class="mt-1 w-full resize-y rounded-lg border border-gray-100/50 bg-gray-50/40 px-2 py-1.5 font-mono text-xs text-gray-700 outline-hidden transition-colors focus:border-blue-400 dark:border-white/[0.04] dark:bg-white/[0.03] dark:text-gray-300 dark:focus:border-blue-500"
						></textarea>
						<p class="mt-0.5 text-[0.6875rem] text-gray-400 dark:text-gray-600">
							{$i18n.t('settings.admin.subagents.systemPrompt.description')}
						</p>
					</div>
				{/if}
			</div>
		{/if}
	</div>

	{#if !loading}
		<div class="flex justify-end pt-6 text-sm font-normal">
			<button
				class="rounded-full bg-black px-3.5 py-1.5 text-sm font-normal text-white transition hover:bg-gray-900 disabled:opacity-50 dark:bg-white dark:text-black dark:hover:bg-gray-100"
				type="submit"
				disabled={saving}
			>
				{$i18n.t('Save')}
			</button>
		</div>
	{/if}
</form>
