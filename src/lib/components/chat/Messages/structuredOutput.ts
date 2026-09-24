export type OutputContentPart = {
	type?: string;
	text?: unknown;
	[key: string]: unknown;
};

export type OutputItem = {
	type?: string;
	id?: string;
	call_id?: string;
	name?: string;
	status?: string;
	arguments?: unknown;
	content?: OutputContentPart[];
	summary?: OutputContentPart[];
	output?: OutputContentPart[];
	files?: unknown;
	embeds?: unknown;
	code?: string;
	lang?: string;
	duration?: number | string | null;
	action?: Record<string, unknown>;
	actions?: Array<Record<string, unknown>>;
	queries?: unknown[];
	[key: string]: unknown;
};

export type OutputDetailToken = {
	summary: string;
	text: string;
	attributes: {
		type: string;
		id?: string;
		name?: string;
		done?: string;
		duration?: string;
		arguments?: string;
		files?: string;
		embeds?: string;
		output?: string;
		status?: string;
		sessionID?: string;
		subagentState?: string;
	};
};

export type OutputDisplayItem =
	| {
			type: 'message';
			id: string;
			text: string;
	  }
	| {
			type: 'detail_single';
			id: string;
			token: OutputDetailToken;
	  }
	| {
			type: 'detail_group';
			id: string;
			tokens: OutputDetailToken[];
	  }
	| {
			type: 'file';
			id: string;
			item: Record<string, unknown>;
	  };

type ResponseStreamEvent = {
	type?: string;
	item_id?: string;
	output_index?: number;
	content_index?: number;
	summary_index?: number;
	item?: OutputItem;
	part?: OutputContentPart;
	delta?: unknown;
	text?: unknown;
	arguments?: unknown;
	response?: {
		output?: OutputItem[];
		[key: string]: unknown;
	};
	[key: string]: unknown;
};

const GROUPABLE_OUTPUT_TYPES = new Set([
	'reasoning',
	'function_call',
	'open_webui:code_interpreter',
	'web_search_call',
	'file_search_call',
	'computer_call'
]);

const OPENAI_TOOL_NAMES: Record<string, string> = {
	web_search_call: 'Web Search',
	file_search_call: 'File Search',
	computer_call: 'Computer Use'
};

// Canonical subagent tool name plus the aliases persisted by older chats.
const SUBAGENT_TOOL_NAMES = new Set(['subagent', 'delegate_task', 'task']);

function getTextFromParts(parts: OutputContentPart[] = []): string {
	return parts
		.map((part) => {
			if (part?.text === undefined || part?.text === null) {
				return '';
			}
			return typeof part.text === 'string' ? part.text : String(part.text);
		})
		.join('');
}

function stringifyAttribute(value: unknown): string {
	if (value === undefined || value === null) {
		return '';
	}
	if (typeof value === 'string') {
		return value;
	}
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
}

function isDoneStatus(status?: string): boolean {
	return status === 'completed' || status === 'failed' || status === 'incomplete';
}

function getMessageText(item: OutputItem): string {
	return getTextFromParts(item.content ?? []);
}

function getReasoningText(item: OutputItem): string {
	const summary = Array.isArray(item.summary) && item.summary.length ? item.summary : null;
	return getTextFromParts(summary ?? item.content ?? []);
}

function getToolResultText(item?: OutputItem): string {
	return (item?.output ?? [])
		.filter((part) => part?.type !== 'input_image')
		.map((part) => {
			if (part?.text === undefined || part?.text === null) {
				return '';
			}
			return typeof part.text === 'string' ? part.text : String(part.text);
		})
		.join('');
}

function parseJSONStringValue(value: unknown): unknown {
	if (typeof value !== 'string') {
		return value;
	}

	let parsed: unknown = value.trim();
	while (typeof parsed === 'string') {
		try {
			parsed = JSON.parse(parsed);
		} catch {
			break;
		}
	}
	return parsed;
}

export type SubagentResult = {
	sessionID: string;
	state: string;
	background: boolean;
};

/**
 * Read the child chat id and run state out of a subagent tool result.
 *
 * A background dispatch returns a JSON handle
 * (`{"sessionID": "...", "status": "running", "output": "..."}`), while both
 * foreground and later-synthesised results carry the
 * `<subagent sessionID="..." state="...">` envelope. Missing fields come back as
 * empty strings so callers can fall back to plain rendering.
 */
export function parseSubagentResult(resultText: string, name?: string): SubagentResult {
	const text = resultText ?? '';
	const parsed = parseJSONStringValue(text) as Record<string, unknown> | null;

	let sessionID = '';
	let state = '';

	if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
		sessionID = typeof parsed.sessionID === 'string' ? parsed.sessionID : '';
		state = typeof parsed.status === 'string' ? parsed.status : '';
	}

	if (!sessionID && !state) {
		const envelope = text.match(/<subagent\b[^>]*>/i)?.[0] ?? '';
		sessionID = envelope.match(/sessionID\s*=\s*"([^"]*)"/i)?.[1] ?? '';
		state = envelope.match(/state\s*=\s*"([^"]*)"/i)?.[1] ?? '';
	}

	// `Background subagent` is what the current builder emits; the hyphenated
	// `Background sub-agent` is the legacy display name persisted in older chats.
	const backgroundName = /^background sub-?agent\b/i.test((name ?? '').trim());

	return {
		sessionID,
		state,
		background: state === 'running' || backgroundName
	};
}

function getInlineFileFromToolOutput(callItem?: OutputItem, resultItem?: OutputItem) {
	if (!callItem || !resultItem || callItem.name !== 'display_file') {
		return null;
	}

	const args = parseJSONStringValue(callItem.arguments) as Record<string, unknown>;
	if (!args || typeof args !== 'object') {
		return null;
	}

	const result = parseJSONStringValue(getToolResultText(resultItem)) as Record<string, unknown>;
	if (
		!result ||
		typeof result !== 'object' ||
		result.type !== 'file' ||
		result.source !== 'open_terminal' ||
		result.exists === false ||
		!result.path ||
		!result.terminal_selector
	) {
		return null;
	}

	return result.page === undefined && args.page !== undefined
		? { ...result, page: args.page }
		: result;
}

function buildToolCallToken(
	item: OutputItem,
	toolOutputByCallId: Record<string, OutputItem>,
	completedSubagentIds?: Set<string>,
	sessions?: Map<string, string>,
	messageId?: string
) {
	const callId = item.call_id ?? item.id ?? '';
	const resultItem = toolOutputByCallId[callId];
	const status = String(item.status ?? '');
	const isPending = status === 'pending';
	const isDone = !!resultItem || status === 'failed' || status === 'incomplete';
	const isExecuting = !isDone && status === 'completed';
	let name = item.name ?? '';
	let subagent: SubagentResult = { sessionID: '', state: '', background: false };
	let argSessionId = '';
	if (SUBAGENT_TOOL_NAMES.has(name)) {
		try {
			const args =
				typeof item.arguments === 'string'
					? JSON.parse(item.arguments || '{}')
					: (item.arguments ?? {});
			// `description` is the current argument; `task` is the persisted old shape.
			const description =
				typeof args.description === 'string' && args.description
					? args.description
					: typeof args.task === 'string' && args.task
						? args.task
						: '?';
			const label = args.background === true ? 'Background subagent' : 'Subagent';
			name = `${label}: "${description.length > 60 ? `${description.slice(0, 60)}...` : description}"`;
			argSessionId = typeof args.sessionID === 'string' ? args.sessionID.trim() : '';
		} catch {
			name = 'Subagent';
		}

		subagent = parseSubagentResult(getToolResultText(resultItem), name);
		// A continuation/follow-up call carries the child id in its arguments; the
		// result envelope is preferred, but fall back to the args so the row links
		// before the child returns.
		if (!subagent.sessionID && argSessionId) {
			subagent = { ...subagent, sessionID: argSessionId };
		}
		// A live `subagent:created` event reports the child before its result
		// exists; use it so a running foreground row is clickable.
		if (!subagent.sessionID && sessions) {
			const liveChildId = sessions.get(`${messageId ?? ''}:${callId}`);
			if (liveChildId) {
				subagent = { ...subagent, sessionID: liveChildId };
			}
		}
	}

	// A background dispatch returns a `running` handle immediately, so the tool
	// call itself looks finished while the child is still working. Keep the row
	// spinning until the child's completion shows up in history.
	const isSubagentRunning =
		!!subagent.sessionID &&
		subagent.state === 'running' &&
		!(completedSubagentIds?.has(subagent.sessionID) ?? false);
	const done = isDone && !isSubagentRunning;

	return {
		summary: isPending
			? 'Tool Approval Needed'
			: done
				? 'Tool Executed'
				: isSubagentRunning || isExecuting
					? 'Executing...'
					: 'Preparing...',
		text: getToolResultText(resultItem),
		attributes: {
			type: 'tool_calls',
			id: callId,
			name,
			done: done ? 'true' : 'false',
			status: isSubagentRunning ? 'in_progress' : status,
			sessionID: subagent.sessionID,
			subagentState: subagent.state,
			arguments: stringifyAttribute(item.arguments ?? ''),
			files: stringifyAttribute(resultItem?.files),
			embeds: stringifyAttribute(resultItem?.embeds)
		}
	};
}

function buildReasoningToken(item: OutputItem, isLastItem: boolean) {
	const duration = item.duration ?? '';
	const isDone = isDoneStatus(item.status) || item.duration !== undefined || !isLastItem;
	const text = getReasoningText(item)
		.split('\n')
		.map((line) => (line.startsWith('>') ? line : `> ${line}`))
		.join('\n');

	return {
		summary: isDone ? `Thought for ${duration || 0} seconds` : 'Thinking...',
		text,
		attributes: {
			type: 'reasoning',
			done: isDone ? 'true' : 'false',
			duration: String(duration)
		}
	};
}

function buildCodeInterpreterToken(item: OutputItem, isLastItem: boolean) {
	const duration = item.duration ?? '';
	const isDone = isDoneStatus(item.status) || item.duration !== undefined || !isLastItem;
	const code = item.code ?? '';
	const lang = item.lang ?? 'python';

	return {
		summary: isDone ? 'Ran Python code' : 'Running Python code…',
		text: code ? `\`\`\`${lang}\n${code}\n\`\`\`` : '',
		attributes: {
			type: 'code_interpreter',
			done: isDone ? 'true' : 'false',
			duration: String(duration),
			output: stringifyAttribute(item.output)
		}
	};
}

function getOpenAIToolSummary(item: OutputItem): string {
	if (item.type === 'web_search_call') {
		const action = item.action ?? {};
		const actionType = action.type;
		if (actionType === 'search') {
			const queries = Array.isArray(action.queries) ? action.queries : [];
			const query = typeof action.query === 'string' ? action.query : '';
			return queries.length ? `Search: ${queries.join(', ')}` : query ? `Search: ${query}` : '';
		}
		if (actionType === 'open_page' && typeof action.url === 'string') {
			return `Open page: ${action.url}`;
		}
		if (actionType === 'find_in_page' && typeof action.pattern === 'string') {
			return `Find in page: ${action.pattern}`;
		}
	}

	if (item.type === 'file_search_call') {
		const queries = item.queries ?? [];
		return queries.length ? `Queries: ${queries.join(', ')}` : '';
	}

	if (item.type === 'computer_call') {
		if (item.action?.type) {
			return `Action: ${item.action.type}`;
		}
		if (Array.isArray(item.actions) && item.actions.length) {
			return `Actions: ${item.actions.map((action) => action.type ?? '?').join(', ')}`;
		}
	}

	return '';
}

function buildOpenAIToolToken(item: OutputItem, isLastItem: boolean) {
	const isDone = isDoneStatus(item.status) || !isLastItem;
	return {
		summary: isDone ? 'Tool Executed' : 'Executing...',
		text: getOpenAIToolSummary(item),
		attributes: {
			type: 'tool_calls',
			id: item.id ?? '',
			name: OPENAI_TOOL_NAMES[item.type ?? ''] ?? item.type ?? '',
			done: isDone ? 'true' : 'false',
			arguments: ''
		}
	};
}

function buildDetailToken(
	item: OutputItem,
	isLastItem: boolean,
	toolOutputByCallId: Record<string, OutputItem>,
	completedSubagentIds?: Set<string>,
	sessions?: Map<string, string>,
	messageId?: string
): OutputDetailToken | null {
	if (item.type === 'function_call') {
		return buildToolCallToken(item, toolOutputByCallId, completedSubagentIds, sessions, messageId);
	}
	if (item.type === 'reasoning') {
		return buildReasoningToken(item, isLastItem);
	}
	if (item.type === 'open_webui:code_interpreter') {
		return buildCodeInterpreterToken(item, isLastItem);
	}
	if (item.type && OPENAI_TOOL_NAMES[item.type]) {
		return buildOpenAIToolToken(item, isLastItem);
	}
	return null;
}

export function buildOutputDisplayItems(
	output: OutputItem[] = [],
	completedSubagentIds?: Set<string>,
	sessions?: Map<string, string>,
	messageId?: string
): OutputDisplayItem[] {
	const displayItems: OutputDisplayItem[] = [];
	const currentDetailTokens: OutputDetailToken[] = [];
	const toolOutputByCallId: Record<string, OutputItem> = {};
	const toolCallByCallId: Record<string, OutputItem> = {};

	for (const item of output) {
		if (item?.type === 'function_call_output' && item.call_id) {
			toolOutputByCallId[item.call_id] = item;
		} else if (item?.type === 'function_call' && (item.call_id || item.id)) {
			toolCallByCallId[item.call_id ?? item.id ?? ''] = item;
		}
	}

	const flushDetails = () => {
		if (currentDetailTokens.length > 1) {
			displayItems.push({
				type: 'detail_group',
				id: `detail-group-${displayItems.length}`,
				tokens: [...currentDetailTokens]
			});
		} else if (currentDetailTokens.length === 1) {
			displayItems.push({
				type: 'detail_single',
				id: `detail-${displayItems.length}`,
				token: currentDetailTokens[0]
			});
		}
		currentDetailTokens.length = 0;
	};

	output.forEach((item, index) => {
		if (!item) {
			return;
		}

		if (item.type === 'function_call_output') {
			const inlineFile = getInlineFileFromToolOutput(toolCallByCallId[item.call_id ?? ''], item);
			if (inlineFile) {
				flushDetails();
				displayItems.push({
					type: 'file',
					id: item.id ?? `file-${index}`,
					item: inlineFile
				});
			}
			return;
		}

		if (
			item.type === 'function_call' &&
			item.name === 'ask_user' &&
			(item.status === 'pending' || item.status === 'in_progress')
		) {
			return;
		}

		if (item.type && GROUPABLE_OUTPUT_TYPES.has(item.type)) {
			const token = buildDetailToken(
				item,
				index === output.length - 1,
				toolOutputByCallId,
				completedSubagentIds,
				sessions,
				messageId
			);
			if (token) {
				currentDetailTokens.push(token);
			}
			return;
		}

		if (item.type === 'message') {
			const text = getMessageText(item);
			if (text.trim()) {
				flushDetails();
				displayItems.push({
					type: 'message',
					id: item.id ?? `message-${index}`,
					text
				});
			}
			return;
		}

		const fallbackText = getMessageText(item);
		if (fallbackText.trim()) {
			flushDetails();
			displayItems.push({
				type: 'message',
				id: item.id ?? `output-${index}`,
				text: fallbackText
			});
		}
	});

	flushDetails();
	return displayItems;
}

export function getOutputText(output?: OutputItem[] | null): string {
	return (output ?? [])
		.filter((item) => item?.type === 'message')
		.map(getMessageText)
		.filter((text) => text.trim())
		.join('\n');
}

type HistoryMessage = {
	content?: unknown;
	meta?: Record<string, unknown> | null;
};

/**
 * Collect the ids of subagent children that have finished, from the persisted
 * internal result messages (`meta.internal === true && meta.type === 'subagent'`).
 *
 * A `pending`/`running` state means the child has not finished; any other (or a
 * missing) state counts as finished. Child ids are read from the current and
 * legacy `meta` fields plus the `<subagent sessionID="...">` envelope in the
 * message content, so a background dispatch row can keep spinning until its
 * child is complete.
 */
export function collectCompletedSubagentIds(
	messages: Record<string, HistoryMessage> | null | undefined
): Set<string> {
	const completed = new Set<string>();

	for (const message of Object.values(messages ?? {})) {
		const meta = message?.meta;
		if (meta?.internal !== true || meta?.type !== 'subagent') {
			continue;
		}

		const state = typeof meta.state === 'string' ? meta.state : '';
		const status = typeof meta.status === 'string' ? meta.status : '';
		const unfinished =
			state === 'pending' ||
			state === 'running' ||
			(!state && (status === 'pending' || status === 'running'));
		if (unfinished) {
			continue;
		}

		const envelopeID = parseSubagentResult(
			typeof message.content === 'string' ? message.content : ''
		).sessionID;
		const ids = [
			meta.childID,
			meta.subagent_chat_id,
			...(Array.isArray(meta.childIDs) ? meta.childIDs : []),
			...(Array.isArray(meta.subagent_chat_ids) ? meta.subagent_chat_ids : []),
			envelopeID
		];
		for (const id of ids) {
			if (typeof id === 'string' && id) {
				completed.add(id);
			}
		}
	}

	return completed;
}

function appendDelta(current: unknown, delta: unknown): unknown {
	if (typeof current === 'string' || typeof delta === 'string') {
		return `${current ?? ''}${delta ?? ''}`;
	}
	if (
		current &&
		delta &&
		typeof current === 'object' &&
		typeof delta === 'object' &&
		!Array.isArray(current) &&
		!Array.isArray(delta)
	) {
		return { ...(current as Record<string, unknown>), ...(delta as Record<string, unknown>) };
	}
	return delta ?? current ?? '';
}

function ensureOutputItem(
	output: OutputItem[],
	outputIndex: number,
	fallback?: OutputItem
): OutputItem {
	while (output.length <= outputIndex) {
		// Only the addressed slot gets the event's item; filler slots must not reuse its id.
		const item =
			output.length === outputIndex && fallback
				? { ...fallback }
				: { type: 'message', status: 'in_progress', role: 'assistant', content: [] };
		output.push(item);
	}
	output[outputIndex] = { ...output[outputIndex] };
	return output[outputIndex];
}

function ensurePart(parts: OutputContentPart[], index: number, fallback?: OutputContentPart) {
	while (parts.length <= index) {
		parts.push(fallback ?? { type: 'output_text', text: '' });
	}
	parts[index] = { ...parts[index] };
	return parts[index];
}

function setPart(
	parts: OutputContentPart[],
	index: number,
	part: OutputContentPart,
	fallback?: OutputContentPart
): void {
	// Assigning past the end leaves a hole that later spreads turn into undefined parts.
	ensurePart(parts, index, fallback);
	parts[index] = part;
}

function findOutputItemIndex(output: OutputItem[], item: OutputItem): number {
	return output.findIndex(
		(existing) =>
			(!!item.id && existing?.id === item.id) ||
			(!!item.call_id && existing?.type === item.type && existing?.call_id === item.call_id)
	);
}

function responseEventUpdatesOutputItem(eventType: string): boolean {
	return (
		eventType === 'response.content_part.added' ||
		eventType === 'response.reasoning_summary_part.added' ||
		eventType.endsWith('.delta') ||
		eventType.endsWith('.done')
	);
}

export function applyResponseStreamEvent(
	output: OutputItem[] = [],
	event: ResponseStreamEvent
): OutputItem[] {
	const eventType = event?.type ?? '';
	if (!eventType.startsWith('response.')) {
		return output;
	}

	if (eventType === 'response.completed') {
		if (!event.response?.output?.length) return output;

		// Completion covers one provider response, not the earlier tool-call rounds.
		const nextOutput = [...output];
		for (const item of event.response.output) {
			const index = findOutputItemIndex(nextOutput, item);
			if (index >= 0) {
				nextOutput[index] = item;
			} else {
				nextOutput.push(item);
			}
		}
		return nextOutput;
	}

	const nextOutput = [...output];
	const eventItemIndex = event.item_id
		? nextOutput.findIndex((item) => item?.id === event.item_id || item?.call_id === event.item_id)
		: -1;
	const outputIndex =
		eventItemIndex >= 0 ? eventItemIndex : (event.output_index ?? Math.max(output.length - 1, 0));

	if (eventType === 'response.output_item.added') {
		if (!event.item) {
			return output;
		}
		const item = { ...event.item };
		const existingIndex = findOutputItemIndex(nextOutput, item);
		if (existingIndex >= 0) {
			nextOutput[existingIndex] = item;
		} else if (outputIndex < nextOutput.length) {
			nextOutput.splice(outputIndex, 0, item);
		} else {
			nextOutput.push(item);
		}
		return nextOutput;
	}

	if (eventType === 'response.output_item.done') {
		if (!event.item) {
			return output;
		}
		const item = { ...event.item };
		const existingIndex = findOutputItemIndex(nextOutput, item);
		if (existingIndex >= 0) {
			nextOutput[existingIndex] = item;
		} else if (outputIndex < nextOutput.length) {
			nextOutput[outputIndex] = item;
		} else {
			nextOutput.push(item);
		}
		return nextOutput;
	}

	if (!responseEventUpdatesOutputItem(eventType)) {
		return output;
	}

	const item = ensureOutputItem(nextOutput, outputIndex, {
		id: event.item_id,
		type: eventType.includes('reasoning')
			? 'reasoning'
			: eventType.includes('function_call')
				? 'function_call'
				: 'message',
		status: 'in_progress',
		role: 'assistant',
		content: []
	});

	if (eventType === 'response.content_part.added') {
		if (item.type === 'reasoning' || !event.part) {
			return nextOutput;
		}
		item.content = [...(item.content ?? [])];
		setPart(item.content, event.content_index ?? item.content.length, { ...event.part });
		return nextOutput;
	}

	if (eventType === 'response.reasoning_summary_part.added') {
		if (!event.part) {
			return nextOutput;
		}
		item.summary = [...(item.summary ?? [])];
		const summaryIndex = event.summary_index ?? item.summary.length;
		setPart(item.summary, summaryIndex, { ...event.part }, { type: 'summary_text', text: '' });
		return nextOutput;
	}

	if (eventType.endsWith('.delta')) {
		const deltaType = eventType.split('.')[1];
		if (deltaType === 'function_call_arguments') {
			item.arguments = appendDelta(item.arguments ?? '', event.delta);
			return nextOutput;
		}

		if (deltaType === 'reasoning_summary_text') {
			const summaryIndex = event.summary_index ?? 0;
			item.summary = [...(item.summary ?? [])];
			const part = ensurePart(item.summary, summaryIndex, { type: 'summary_text', text: '' });
			part.text = appendDelta(part.text ?? '', event.delta);
			return nextOutput;
		}

		if (item.type === 'open_webui:code_interpreter') {
			item.code = `${item.code ?? ''}${event.delta ?? ''}`;
			return nextOutput;
		}

		const key = deltaType === 'output_text' || deltaType === 'reasoning_text' ? 'text' : deltaType;
		item.content = [...(item.content ?? [])];
		const part = ensurePart(item.content, event.content_index ?? 0);
		part[key] = appendDelta(part[key], event.delta);
		return nextOutput;
	}

	if (eventType.endsWith('.done')) {
		const typeName = eventType.split('.')[1];
		if (typeName === 'content_part' && event.part) {
			item.content = [...(item.content ?? [])];
			const contentIndex = event.content_index ?? Math.max(item.content.length - 1, 0);
			setPart(item.content, contentIndex, { ...event.part });
		} else if (typeName === 'function_call_arguments' && event.arguments !== undefined) {
			item.arguments = event.arguments;
		} else if (
			(typeName === 'output_text' || typeName === 'text' || typeName === 'reasoning_text') &&
			event.text !== undefined
		) {
			item.content = [...(item.content ?? [])];
			const part = ensurePart(item.content, event.content_index ?? 0);
			part.text = event.text;
		}
	}

	return nextOutput;
}

export function replaceOutputMessageText(
	output: OutputItem[] = [],
	oldContent: string,
	newContent: string
): OutputItem[] {
	if (!oldContent) {
		return output;
	}

	let replaced = false;
	const nextOutput = output.map((item) => {
		if (replaced || item?.type !== 'message' || !Array.isArray(item.content)) {
			return item;
		}

		const partIndex = item.content.findIndex(
			(part) => typeof part?.text === 'string' && part.text.includes(oldContent)
		);
		if (partIndex === -1) {
			return item;
		}

		replaced = true;
		const nextContent = [...item.content];
		const part = nextContent[partIndex];
		nextContent[partIndex] = {
			...part,
			text: (part.text as string).replace(oldContent, newContent)
		};

		return {
			...item,
			content: nextContent
		};
	});

	return replaced ? nextOutput : output;
}
