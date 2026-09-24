<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';
	import { config, settings, user as _user, mobile, temporaryChatEnabled } from '$lib/stores';
	import { refreshChatList } from '$lib/stores/chatList';
	import { tick, getContext, onMount, onDestroy, createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();

	import { toast } from 'svelte-sonner';
	import { deleteChatMessageById, updateChatById } from '$lib/apis/chats';
	import {
		copyToClipboard,
		createMessagesList,
		extractCurlyBraceWords,
		getDeepestChildId
	} from '$lib/utils';
	import { getMessageCheckpoint } from '$lib/utils/contextCompaction';

	import Message from './Messages/Message.svelte';
	import ContextCheckpoint from './Messages/ContextCheckpoint.svelte';
	import Loader from '../common/Loader.svelte';
	import Spinner from '../common/Spinner.svelte';

	import ChatPlaceholder from './ChatPlaceholder.svelte';

	const i18n = getContext('i18n');

	export let className = 'h-full flex pt-18';

	export let chatId = '';
	export let user = $_user;

	export let prompt;
	export let history = {};
	export let selectedModels;
	export let atSelectedModel;

	let messages: any[] = [];

	export let setInputText: Function = () => {};

	export let sendMessage: Function;
	export let continueResponse: Function;
	export let regenerateResponse: Function;
	export let mergeResponses: Function;

	export let chatActionHandler: Function;
	export let showMessage: Function = () => {};
	export let submitMessage: Function = () => {};
	export let addMessages: Function = () => {};
	export let onToolCallResolved: Function = () => {};
	export let forkHandler: Function | null = null;

	export let readOnly = false;
	export let allowDelete = true;
	export let compactPreview = false;
	export let editCodeBlock = true;

	export let contextUsage: any = null;
	export let contextCompaction: any = null;
	export let contextCompactionEnabled = false;
	export let onUndoCheckpoint: (messageId: string) => void = () => {};
	export let onRegenerateCheckpoint: () => void = () => {};

	export let topPadding = false;
	export let bottomPadding = false;
	export let autoScroll;
	export let messagesContainerId = 'messages-container';

	export let onSelect = (e) => {};
	export let onInsertToNote: ((content: string) => void) | null = null;

	export let messagesCount: number | null = 8;
	let messagesLoading = false;

	const getMessagesContainer = () => document.getElementById(messagesContainerId);

	onDestroy(() => {
		cancelAnimationFrame(pendingRebuild);
	});

	const loadMoreMessages = async () => {
		const element = getMessagesContainer();
		const previousScrollHeight = element?.scrollHeight ?? 0;

		messagesLoading = true;
		messagesCount += 8;

		buildMessages();

		await tick();

		if (element) {
			element.scrollTop += element.scrollHeight - previousScrollHeight;
		}

		messagesLoading = false;
	};

	let pendingRebuild = null;
	let lastCurrentId = null;

	const buildMessages = () => {
		let _messages = [];

		let message = history.messages[history.currentId];
		const visitedMessageIds = new Set();

		while (message && (messagesCount !== null ? _messages.length < messagesCount : true)) {
			if (visitedMessageIds.has(message.id)) {
				console.warn('Circular dependency detected in message history', message.id);
				break;
			}
			visitedMessageIds.add(message.id);

			_messages.push(message);
			message = message.parentId !== null ? history.messages[message.parentId] : null;
		}

		messages = _messages.reverse();
	};

	// Throttle message list rebuilds to once per animation frame during streaming.
	// Structural changes (currentId change) always rebuild immediately.
	const handleHistoryChange = (currentId, _messages) => {
		if (!currentId) {
			messages = [];
			return;
		}

		const currentIdChanged = currentId !== lastCurrentId;
		lastCurrentId = currentId;

		if (currentIdChanged) {
			// Structural change: new chat, navigation, new message — rebuild immediately
			cancelAnimationFrame(pendingRebuild);
			pendingRebuild = null;
			buildMessages();
		} else if (_messages) {
			// Content update (streaming) — throttle to once per frame
			if (!pendingRebuild) {
				pendingRebuild = requestAnimationFrame(() => {
					pendingRebuild = null;
					buildMessages();
				});
			}
		}
	};

	$: handleHistoryChange(history.currentId, history.messages);

	// Newest checkpoint on the active branch, if any. Used to render the
	// compaction timeline row right before the boundary message that carries it.
	const findCheckpoint = (list: any[]) => {
		let found: { idx: number; messageId: string; record: any } | null = null;
		for (let idx = 0; idx < list.length; idx += 1) {
			const record = getMessageCheckpoint(list[idx]);
			if (record) {
				found = { idx, messageId: list[idx].id, record };
			}
		}
		return found;
	};

	$: compactionStatus = contextCompaction?.status ?? null;
	$: compactionRunning = compactionStatus === 'running';
	$: visibleCheckpoint = findCheckpoint(messages);
	// Only walk the full branch when the boundary is outside the loaded window.
	$: fullCheckpoint =
		visibleCheckpoint === null && (history as any)?.currentId
			? findCheckpoint(createMessagesList(history, (history as any).currentId))
			: null;
	$: globalCheckpoint = visibleCheckpoint ?? fullCheckpoint;
	// Show the row at the top of the list only when the boundary message itself is
	// outside the loaded window (or a failure has no checkpoint to attach to).
	$: showCheckpointHeader =
		visibleCheckpoint === null && (globalCheckpoint !== null || compactionStatus === 'failed');

	$: if (autoScroll && bottomPadding) {
		(async () => {
			await tick();
			scrollToBottom();
		})();
	}

	const scrollToBottom = () => {
		const element = getMessagesContainer();
		if (element) {
			element.scrollTop = element.scrollHeight;

			// Follow-up scroll to account for content-visibility: auto re-layouts
			requestAnimationFrame(() => {
				if (element) {
					element.scrollTop = element.scrollHeight;
				}
			});
		}
	};

	export const scrollToTop = async () => {
		messagesCount = null;
		buildMessages();
		await tick();

		const element = getMessagesContainer();
		if (!element) return;

		element.scrollTo({ top: 0, behavior: 'smooth' });
		requestAnimationFrame(() => {
			element.scrollTo({ top: 0, behavior: 'smooth' });
			requestAnimationFrame(() => {
				element.scrollTo({ top: 0, behavior: 'smooth' });
			});
		});
	};

	const updateChat = async () => {
		if (!$temporaryChatEnabled) {
			history = history;
			await tick();
			const res = await updateChatById(localStorage.token, chatId, {
				history: history,
				messages: messages
			});

			// Keep local plain-content edits aligned with the saved chat response.
			if (res?.chat?.history?.messages) {
				for (const [id, msg] of Object.entries(res.chat.history.messages)) {
					if (history.messages[id] && (msg as any).content) {
						history.messages[id].content = (msg as any).content;
					}
				}
				history = history;
			}

			await refreshChatList(localStorage.token);
		}
	};

	const gotoMessage = async (message, idx) => {
		// Determine the correct sibling list (either parent's children or root messages)
		let siblings;
		if (message.parentId !== null) {
			siblings = history.messages[message.parentId].childrenIds;
		} else {
			siblings = Object.values(history.messages)
				.filter((msg) => msg.parentId === null)
				.map((msg) => msg.id);
		}

		// Clamp index to a valid range
		idx = Math.max(0, Math.min(idx, siblings.length - 1));

		let messageId = siblings[idx];

		// If we're navigating to a different message
		if (message.id !== messageId) {
			history.currentId = getDeepestChildId(history, messageId);
		}

		await tick();

		// Optional auto-scroll
		if ($settings?.scrollOnBranchChange ?? true) {
			const element = getMessagesContainer();
			autoScroll = element
				? element.scrollHeight - element.scrollTop <= element.clientHeight + 50
				: false;

			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const showPreviousMessage = async (message) => {
		if (message.parentId !== null) {
			let messageId =
				history.messages[message.parentId].childrenIds[
					Math.max(history.messages[message.parentId].childrenIds.indexOf(message.id) - 1, 0)
				];

			if (message.id !== messageId) {
				history.currentId = getDeepestChildId(history, messageId);
			}
		} else {
			let childrenIds = Object.values(history.messages)
				.filter((message) => message.parentId === null)
				.map((message) => message.id);
			let messageId = childrenIds[Math.max(childrenIds.indexOf(message.id) - 1, 0)];

			if (message.id !== messageId) {
				history.currentId = getDeepestChildId(history, messageId);
			}
		}

		await tick();

		if ($settings?.scrollOnBranchChange ?? true) {
			const element = getMessagesContainer();
			autoScroll = element
				? element.scrollHeight - element.scrollTop <= element.clientHeight + 50
				: false;

			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const showNextMessage = async (message) => {
		if (message.parentId !== null) {
			let messageId =
				history.messages[message.parentId].childrenIds[
					Math.min(
						history.messages[message.parentId].childrenIds.indexOf(message.id) + 1,
						history.messages[message.parentId].childrenIds.length - 1
					)
				];

			if (message.id !== messageId) {
				history.currentId = getDeepestChildId(history, messageId);
			}
		} else {
			let childrenIds = Object.values(history.messages)
				.filter((message) => message.parentId === null)
				.map((message) => message.id);
			let messageId =
				childrenIds[Math.min(childrenIds.indexOf(message.id) + 1, childrenIds.length - 1)];

			if (message.id !== messageId) {
				history.currentId = getDeepestChildId(history, messageId);
			}
		}

		await tick();

		if ($settings?.scrollOnBranchChange ?? true) {
			const element = getMessagesContainer();
			autoScroll = element
				? element.scrollHeight - element.scrollTop <= element.clientHeight + 50
				: false;

			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const rateMessage = async (messageId, rating) => {
		history.messages[messageId].annotation = {
			...history.messages[messageId].annotation,
			rating: rating
		};

		await updateChat();
	};

	const editMessage = async (messageId, { content, files, output = undefined }, submit = true) => {
		if ((selectedModels ?? []).filter((id) => id).length === 0) {
			toast.error($i18n.t('Model not selected'));
			return;
		}
		if (history.messages[messageId].role === 'user') {
			if (submit) {
				// New user message
				let userPrompt = content;
				let userMessageId = uuidv4();

				let userMessage = {
					id: userMessageId,
					parentId: history.messages[messageId].parentId,
					childrenIds: [],
					role: 'user',
					content: userPrompt,
					...(files && { files: files }),
					models: selectedModels,
					timestamp: Math.floor(Date.now() / 1000) // Unix epoch
				};

				let messageParentId = history.messages[messageId].parentId;

				if (messageParentId !== null) {
					history.messages[messageParentId].childrenIds = [
						...history.messages[messageParentId].childrenIds,
						userMessageId
					];
				}

				history.messages[userMessageId] = userMessage;
				history.currentId = userMessageId;

				await tick();
				await sendMessage(history, userMessageId);
			} else {
				// Edit user message
				history.messages[messageId].content = content;
				history.messages[messageId].files = files;
				await updateChat();
			}
		} else {
			if (submit) {
				// New response message (Save As Copy)
				const responseMessageId = uuidv4();
				const message = history.messages[messageId];
				const parentId = message.parentId;

				const responseMessage = {
					...message,
					id: responseMessageId,
					parentId: parentId,
					childrenIds: [],
					files: undefined,
					content: output !== undefined ? '' : content,
					...(output !== undefined ? { output } : {}),
					timestamp: Math.floor(Date.now() / 1000) // Unix epoch
				};

				history.messages[responseMessageId] = responseMessage;
				history.currentId = responseMessageId;

				// Append messageId to childrenIds of parent message
				if (parentId !== null) {
					history.messages[parentId].childrenIds = [
						...history.messages[parentId].childrenIds,
						responseMessageId
					];
				}

				await updateChat();
			} else {
				// Edit response message
				if (content !== undefined) {
					history.messages[messageId].originalContent = history.messages[messageId].content;
					history.messages[messageId].content = content;
				}
				if (output !== undefined) {
					history.messages[messageId].output = output;
					history.messages[messageId].content = '';
				}
				await updateChat();
			}
		}
	};

	const actionMessage = async (actionId, message, event = null) => {
		await chatActionHandler(chatId, actionId, message.model, message.id, event);
	};

	const saveMessage = async (messageId, message) => {
		if (!history.messages?.[messageId]) {
			return;
		}

		history.messages[messageId] = message;
		await updateChat();
	};

	const deleteMessage = async (messageId) => {
		const messageToDelete = history.messages[messageId];
		const parentMessageId = messageToDelete.parentId;
		const childMessageIds = messageToDelete.childrenIds ?? [];

		// Collect all grandchildren
		const grandchildrenIds = childMessageIds.flatMap(
			(childId) => history.messages[childId]?.childrenIds ?? []
		);

		// Update parent's children
		if (parentMessageId && history.messages[parentMessageId]) {
			history.messages[parentMessageId].childrenIds = [
				...history.messages[parentMessageId].childrenIds.filter((id) => id !== messageId),
				...grandchildrenIds
			];
		}

		// Update grandchildren's parent
		grandchildrenIds.forEach((grandchildId) => {
			if (history.messages[grandchildId]) {
				history.messages[grandchildId].parentId = parentMessageId;
			}
		});

		// Delete the message and its children
		[messageId, ...childMessageIds].forEach((id) => {
			delete history.messages[id];
		});

		history.currentId = getDeepestChildId(history, parentMessageId);
		history = history;

		if (!$temporaryChatEnabled) {
			const res = await deleteChatMessageById(localStorage.token, chatId, messageId);
			if (res?.chat?.history) {
				history = res.chat.history;
			}

			await refreshChatList(localStorage.token);
		}
	};

	const triggerScroll = () => {
		if (autoScroll) {
			const element = getMessagesContainer();
			if (element) {
				autoScroll = element.scrollHeight - element.scrollTop <= element.clientHeight + 50;
				setTimeout(() => {
					scrollToBottom();
				}, 100);
			}
		}
	};
</script>

<div class={className}>
	{#if Object.keys(history?.messages ?? {}).length == 0}
		<ChatPlaceholder modelIds={selectedModels} {atSelectedModel} {onSelect} />
	{:else}
		<div class="w-full pt-2">
			{#key chatId}
				<section class="w-full" aria-labelledby="chat-conversation">
					<h2 class="sr-only" id="chat-conversation">{$i18n.t('Chat Conversation')}</h2>
					{#if messages.at(0)?.parentId !== null}
						<Loader
							on:visible={(e) => {
								console.log('visible');
								if (!messagesLoading) {
									loadMoreMessages();
								}
							}}
						>
							<div class="w-full flex justify-center py-1 text-xs animate-pulse items-center gap-2">
								<Spinner className=" size-4" />
								<div class=" ">{$i18n.t('Loading...')}</div>
							</div>
						</Loader>
					{/if}
					<ul role="log" aria-live="polite" aria-relevant="additions" aria-atomic="false">
						{#if compactionRunning}
							<ContextCheckpoint status="running" usage={contextUsage} disabled={true} />
						{:else if showCheckpointHeader}
							<ContextCheckpoint
								checkpoint={globalCheckpoint?.record}
								status={compactionStatus}
								usage={contextUsage}
								messageId={globalCheckpoint?.messageId}
								model={globalCheckpoint?.record?.model}
								disabled={!contextCompactionEnabled || compactionRunning}
								onUndo={onUndoCheckpoint}
								onRegenerate={onRegenerateCheckpoint}
							/>
						{/if}
						{#each messages as message, messageIdx (message.id)}
							{#if !compactionRunning && visibleCheckpoint?.idx === messageIdx}
								<ContextCheckpoint
									checkpoint={visibleCheckpoint.record}
									status={compactionStatus}
									usage={contextUsage}
									messageId={visibleCheckpoint.messageId}
									model={visibleCheckpoint.record?.model}
									disabled={!contextCompactionEnabled || compactionRunning}
									onUndo={onUndoCheckpoint}
									onRegenerate={onRegenerateCheckpoint}
								/>
							{/if}
							<Message
								{chatId}
								bind:history
								{selectedModels}
								messageId={message.id}
								idx={messageIdx}
								{user}
								{setInputText}
								{gotoMessage}
								{showPreviousMessage}
								{showNextMessage}
								{updateChat}
								{editMessage}
								{deleteMessage}
								{rateMessage}
								{actionMessage}
								{saveMessage}
								{submitMessage}
								{regenerateResponse}
								{continueResponse}
								{mergeResponses}
								{addMessages}
								{onToolCallResolved}
								{forkHandler}
								{allowDelete}
								{triggerScroll}
								{readOnly}
								{compactPreview}
								{editCodeBlock}
								{topPadding}
								{onInsertToNote}
							/>
						{/each}
					</ul>
				</section>
				<div class="pb-18" />
				{#if bottomPadding}
					<div class="  pb-6" />
				{/if}
			{/key}
		</div>
	{/if}
</div>
