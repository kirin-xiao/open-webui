/**
 * Presentation layer that turns a raw tool call into a short, readable label.
 *
 * The module is free of runtime imports (no Svelte, no i18next) so it can be unit
 * tested directly and consumed from either render path. It returns translation
 * descriptors (`{ key, values }`) for UI chrome and raw `{ text }` for content that
 * must not be translated (tool names, subagent titles). Callers render them with
 * `i18n.t(key, values)`; `formatToolLabel` is a convenience for that.
 *
 * This file is `.js` and `formatToolLabel` calls `t()` with literal
 * keys on purpose: `i18next-parser` only scans `src/**\/*.{js,svelte}` and only
 * understands literal translation calls, so dynamic descriptors alone would be
 * deleted by `npm run i18n:parse`.
 *
 * Grouping/streaming stays where it is: this only decides what the group header and
 * the per-call rows say.
 */

/**
 * @typedef {{ key: string, values?: Record<string, unknown> } | { text: string }} ToolLabelText
 */

/**
 * @typedef {object} ToolLabel
 * @property {ToolLabelText} active Shown while the call is preparing or executing.
 * @property {ToolLabelText} done Shown once the call has a result.
 * @property {'search' | 'read' | 'code' | 'question' | 'other'} category Used to aggregate calls in the group header.
 */

/**
 * @typedef {object} ToolLabelToken
 * @property {object} [attributes]
 * @property {string} [attributes.type]
 * @property {string} [attributes.name]
 * @property {string} [attributes.done]
 * @property {string} [attributes.status]
 * @property {string} [attributes.arguments]
 */

const SUMMARY_MAX_LENGTH = 60;

/** Argument keys tried, in order, when summarising an unknown tool call. */
const SUMMARY_ARG_KEYS = [
	'query',
	'url',
	'path',
	'pattern',
	'name',
	'description',
	'task',
	'command'
];

/** Argument keys whose string value is source code rather than prose. */
const CODE_ARG_KEYS = new Set(['code', 'source', 'script']);

/** Known aliases for the same underlying tool, keyed by lower-cased name. */
/** @type {Record<string, string>} */
const TOOL_NAME_ALIASES = {
	web_search: 'search_web',
	websearch: 'search_web',
	fetch: 'fetch_url',
	webfetch: 'fetch_url',
	fetch_webpage: 'fetch_url',
	get_url: 'fetch_url',
	run_python: 'execute_code',
	python: 'execute_code',
	python_execute: 'execute_code',
	code_interpreter: 'execute_code',
	request_user_input: 'ask_user',
	askuser: 'ask_user',
	delegate_task: 'subagent',
	task: 'subagent'
};

/** Raw names that identify the subagent tool itself, not a rendered call. */
const SUBAGENT_TOOL_NAMES = new Set(['subagent', 'delegate_task', 'task']);

/**
 * @param {ToolLabelText} part
 * @returns {part is { key: string, values?: Record<string, unknown> }}
 */
export function isKeyLabel(part) {
	return 'key' in part;
}

/**
 * @param {string | null | undefined} raw
 * @returns {Record<string, unknown> | null}
 */
export function parseToolArguments(raw) {
	if (!raw) return null;

	/** @type {unknown} */
	let value = raw;
	while (typeof value === 'string') {
		try {
			value = JSON.parse(value);
		} catch {
			break;
		}
	}

	if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
		return /** @type {Record<string, unknown>} */ (value);
	}
	return null;
}

/**
 * @param {string} name
 * @returns {string}
 */
function canonicalToolName(name) {
	const lower = (name ?? '').trim().toLowerCase();
	if (TOOL_NAME_ALIASES[lower]) return TOOL_NAME_ALIASES[lower];

	// MCP/custom tools are often namespaced (`server.tool`, `server:tool`). Match the
	// trailing segment so an alias or canonical name still applies.
	const tail = lower.split(/[.:/]/).pop() ?? lower;
	return TOOL_NAME_ALIASES[tail] ?? tail;
}

/**
 * `structuredOutput.ts` rewrites a `subagent` call (or a persisted `delegate_task`
 * / `task` call) into a display name such as
 * `Subagent: "..."` (or `Background subagent: "..."`). Those must stay data, not be
 * re-summarised, so detect them here as well.
 *
 * Older chats persisted `Sub-agent:` / `Background sub-agent:` titles, so the
 * hyphen is optional to keep those legacy rows rendering as data too.
 *
 * @param {string} name
 * @returns {boolean}
 */
function isDelegateDisplayName(name) {
	return /^(background )?sub-?agent\b/i.test((name ?? '').trim());
}

/**
 * @param {string} value
 * @param {number} [max]
 * @returns {string}
 */
function truncate(value, max = SUMMARY_MAX_LENGTH) {
	const collapsed = value.trim().replace(/\s+/g, ' ');
	if (collapsed.length <= max) return collapsed;
	return `${collapsed.slice(0, max - 1).trimEnd()}…`;
}

/**
 * @param {Record<string, unknown> | null} args
 * @returns {string}
 */
function summarizeArguments(args) {
	if (!args) return '';

	for (const key of SUMMARY_ARG_KEYS) {
		const value = args[key];
		if (typeof value === 'string' && value.trim()) return truncate(value);
	}

	for (const value of Object.values(args)) {
		if (typeof value === 'string' && value.trim()) return truncate(value);
		if (typeof value === 'number' || typeof value === 'boolean') return String(value);
	}

	return '';
}

/**
 * @param {string} url
 * @returns {string}
 */
function hostOf(url) {
	const raw = url.trim();
	if (!raw) return '';
	// Only prepend a scheme when there is none; otherwise `ftp://x` and `HTTP://x`
	// would be mis-parsed as a path on a fake host.
	const hasScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw);
	try {
		const parsed = new URL(hasScheme ? raw : `https://${raw}`);
		const path = parsed.pathname && parsed.pathname !== '/' ? parsed.pathname : '';
		return truncate(`${parsed.host}${path}`);
	} catch {
		return truncate(raw);
	}
}

/**
 * @param {Record<string, unknown> | null} args
 * @returns {number}
 */
function questionCount(args) {
	const questions = args?.questions;
	return Array.isArray(questions) ? questions.length : 0;
}

/**
 * @param {Record<string, unknown> | null} args
 * @returns {ToolLabel}
 */
function searchWebLabel(args) {
	const query = typeof args?.query === 'string' ? truncate(args.query) : '';
	return {
		active: query
			? { key: 'Searching the web for "{{QUERY}}"', values: { QUERY: query } }
			: { key: 'Searching the web…' },
		done: query
			? { key: 'Searched the web for "{{QUERY}}"', values: { QUERY: query } }
			: { key: 'Searched the web' },
		category: 'search'
	};
}

/**
 * @param {Record<string, unknown> | null} args
 * @returns {ToolLabel}
 */
function fetchUrlLabel(args) {
	const host = typeof args?.url === 'string' ? hostOf(args.url) : '';
	return {
		active: host
			? { key: 'Reading {{HOST}}…', values: { HOST: host } }
			: { key: 'Reading a page…' },
		done: host ? { key: 'Read {{HOST}}', values: { HOST: host } } : { key: 'Read a page' },
		category: 'read'
	};
}

/**
 * @returns {ToolLabel}
 */
function executeCodeLabel() {
	return {
		active: { key: 'Running Python code…' },
		done: { key: 'Ran Python code' },
		category: 'code'
	};
}

/**
 * @param {Record<string, unknown> | null} args
 * @returns {ToolLabel}
 */
function askUserLabel(args) {
	const count = questionCount(args);
	if (count <= 1) {
		const values = count === 1 ? { COUNT: count } : undefined;
		return {
			active:
				count === 1 ? { key: 'Asking {{COUNT}} question…', values } : { key: 'Asking a question…' },
			done: count === 1 ? { key: 'Asked {{COUNT}} question', values } : { key: 'Asked a question' },
			category: 'question'
		};
	}

	const values = { COUNT: count };
	return {
		active: { key: 'Asking {{COUNT}} questions…', values },
		done: { key: 'Asked {{COUNT}} questions', values },
		category: 'question'
	};
}

/**
 * @param {string} displayName
 * @returns {ToolLabel}
 */
function delegateTaskLabel(displayName) {
	// Subagent names carry the task text and are already meaningful; treat them as data.
	// Bare tool names (canonical or alias, including a namespaced form) are not.
	const trimmed = (displayName ?? '').trim();
	const tail = trimmed.toLowerCase().split(/[.:/]/).pop() ?? '';
	const name =
		trimmed && !SUBAGENT_TOOL_NAMES.has(trimmed.toLowerCase()) && !SUBAGENT_TOOL_NAMES.has(tail)
			? trimmed
			: 'Subagent';
	/** @type {ToolLabelText} */
	const label = { text: name };
	return { active: label, done: label, category: 'other' };
}

/**
 * @param {string} displayName
 * @param {Record<string, unknown> | null} args
 * @returns {ToolLabel}
 */
function genericLabel(displayName, args) {
	const name = displayName || 'tool';
	const summary = summarizeArguments(args);
	/** @type {ToolLabelText} */
	const label = summary
		? { key: '{{NAME}} — {{SUMMARY}}', values: { NAME: name, SUMMARY: summary } }
		: { text: name };
	return { active: label, done: label, category: 'other' };
}

/**
 * @param {string} name
 * @param {Record<string, unknown> | null} [args]
 * @returns {ToolLabel}
 */
export function getToolLabel(name, args) {
	switch (canonicalToolName(name)) {
		case 'search_web':
			return searchWebLabel(args ?? null);
		case 'fetch_url':
			return fetchUrlLabel(args ?? null);
		case 'execute_code':
			return executeCodeLabel();
		case 'ask_user':
			return askUserLabel(args ?? null);
		case 'subagent':
			return delegateTaskLabel(name ?? '');
		default:
			if (isDelegateDisplayName(name)) return delegateTaskLabel(name);
			return genericLabel(name ?? '', args ?? null);
	}
}

/**
 * @param {'search' | 'read' | 'code' | 'question'} category
 * @param {number} count
 * @param {boolean} active
 * @returns {ToolLabelText}
 */
function groupPart(category, count, active) {
	if (category === 'search') {
		return { key: active ? 'searching the web…' : 'searched the web' };
	}

	const values = { COUNT: count };

	// Group parts are lower-case phrases; the header capitalises only the first
	// phrase, so `Searched the web, read 1 page` reads naturally and mid-list
	// phrases stay consistent.
	if (category === 'read') {
		if (active) {
			return {
				key: count === 1 ? 'reading {{COUNT}} page…' : 'reading {{COUNT}} pages…',
				values
			};
		}
		return { key: count === 1 ? 'read {{COUNT}} page' : 'read {{COUNT}} pages', values };
	}

	if (category === 'code') {
		if (active) {
			return {
				key: count === 1 ? 'running {{COUNT}} code block…' : 'running {{COUNT}} code blocks…',
				values
			};
		}
		return { key: count === 1 ? 'ran {{COUNT}} code block' : 'ran {{COUNT}} code blocks', values };
	}

	// question
	if (active) {
		return {
			key: count === 1 ? 'asking {{COUNT}} question…' : 'asking {{COUNT}} questions…',
			values
		};
	}
	return { key: count === 1 ? 'asked {{COUNT}} question' : 'asked {{COUNT}} questions', values };
}

/**
 * Build the group header parts for a run of consecutive detail tokens.
 *
 * Search reads as a single statement, file reads / code runs / questions are counted,
 * and unknown tools keep their name. Parts preserve first-seen order. An empty result
 * means the caller should fall back to the generic "Exploring"/"Explored" prefix.
 *
 * @param {ToolLabelToken[]} [tokens]
 * @param {boolean} [active]
 * @returns {ToolLabelText[]}
 */
export function getToolGroupSummary(tokens = [], active = false) {
	/** @type {string[]} */
	const order = [];
	/** @type {Record<string, number>} */
	const counts = {};
	const seenOther = new Set();

	for (const token of tokens) {
		const type = token?.attributes?.type;

		if (type === 'code_interpreter') {
			if (!order.includes('code')) order.push('code');
			counts.code = (counts.code ?? 0) + 1;
			continue;
		}

		if (type !== 'tool_calls') continue;

		const name = token?.attributes?.name ?? '';
		const args = parseToolArguments(token?.attributes?.arguments);
		const label = getToolLabel(name, args);

		if (label.category === 'other') {
			const display = name || 'tool';
			if (seenOther.has(display)) continue;
			seenOther.add(display);
			order.push(`name:${display}`);
			continue;
		}

		if (!order.includes(label.category)) order.push(label.category);
		// A single ask_user call can carry several questions; count those, not calls.
		const increment = label.category === 'question' ? Math.max(1, questionCount(args)) : 1;
		counts[label.category] = (counts[label.category] ?? 0) + increment;
	}

	/** @type {ToolLabelText[]} */
	const parts = [];
	for (const entry of order) {
		if (entry.startsWith('name:')) {
			parts.push({ text: entry.slice('name:'.length) });
			continue;
		}
		const count = counts[entry] ?? 0;
		if (count <= 0) continue;
		parts.push(groupPart(/** @type {any} */ (entry), count, active));
	}

	return parts;
}

/**
 * Resolve a label descriptor to a display string.
 *
 * `t` is passed in (rather than importing i18next) so the module stays testable.
 * The switch exists so `i18next-parser`, which only understands literal keys passed
 * directly to the translate function, keeps every key in the catalogue.
 *
 * @param {{ key: string, values?: Record<string, unknown> } | { text: string }} part
 * @param {{ t: (key: string, values?: Record<string, unknown>) => string }} i18n
 * @returns {string}
 */
export function formatToolLabel(part, i18n) {
	if (!isKeyLabel(part)) return part.text;

	const { key, values = {} } = part;
	const t = i18n.t.bind(i18n);

	switch (key) {
		case 'Searching the web for "{{QUERY}}"':
			return t('Searching the web for "{{QUERY}}"', values);
		case 'Searching the web…':
			return t('Searching the web…', values);
		case 'Searched the web for "{{QUERY}}"':
			return t('Searched the web for "{{QUERY}}"', values);
		case 'Searched the web':
			return t('Searched the web', values);
		case 'Reading {{HOST}}…':
			return t('Reading {{HOST}}…', values);
		case 'Reading a page…':
			return t('Reading a page…', values);
		case 'Read {{HOST}}':
			return t('Read {{HOST}}', values);
		case 'Read a page':
			return t('Read a page', values);
		case 'Running Python code…':
			return t('Running Python code…', values);
		case 'Ran Python code':
			return t('Ran Python code', values);
		case 'Asking {{COUNT}} question…':
			return t('Asking {{COUNT}} question…', values);
		case 'Asking a question…':
			return t('Asking a question…', values);
		case 'Asked {{COUNT}} question':
			return t('Asked {{COUNT}} question', values);
		case 'Asked a question':
			return t('Asked a question', values);
		case 'Asking {{COUNT}} questions…':
			return t('Asking {{COUNT}} questions…', values);
		case 'Asked {{COUNT}} questions':
			return t('Asked {{COUNT}} questions', values);
		case '{{NAME}} — {{SUMMARY}}':
			return t('{{NAME}} — {{SUMMARY}}', values);
		case 'searched the web':
			return t('searched the web', values);
		case 'searching the web…':
			return t('searching the web…', values);
		case 'reading {{COUNT}} page…':
			return t('reading {{COUNT}} page…', values);
		case 'reading {{COUNT}} pages…':
			return t('reading {{COUNT}} pages…', values);
		case 'read {{COUNT}} page':
			return t('read {{COUNT}} page', values);
		case 'read {{COUNT}} pages':
			return t('read {{COUNT}} pages', values);
		case 'running {{COUNT}} code block…':
			return t('running {{COUNT}} code block…', values);
		case 'running {{COUNT}} code blocks…':
			return t('running {{COUNT}} code blocks…', values);
		case 'ran {{COUNT}} code block':
			return t('ran {{COUNT}} code block', values);
		case 'ran {{COUNT}} code blocks':
			return t('ran {{COUNT}} code blocks', values);
		case 'asking {{COUNT}} question…':
			return t('asking {{COUNT}} question…', values);
		case 'asking {{COUNT}} questions…':
			return t('asking {{COUNT}} questions…', values);
		case 'asked {{COUNT}} question':
			return t('asked {{COUNT}} question', values);
		case 'asked {{COUNT}} questions':
			return t('asked {{COUNT}} questions', values);
		default:
			return t(key, values);
	}
}

/**
 * @typedef {object} AskUserQuestionView
 * @property {string} header
 * @property {string} question
 * @property {Array<{ label: string, description: string }>} options
 * @property {boolean} allowOther
 */

/**
 * @typedef {object} ToolArgumentRow
 * @property {string} key Argument name.
 * @property {'scalar' | 'text' | 'code' | 'json' | 'questions'} kind How the value should be rendered.
 * @property {string} [text] Value for the scalar/text/code/json kinds.
 * @property {string} [lang] Language for the `code` kind.
 * @property {AskUserQuestionView[]} [questions] Value for the `questions` kind.
 */

/**
 * @param {unknown} value
 * @returns {string}
 */
function stringValue(value) {
	if (typeof value === 'string') return value;
	if (value === undefined || value === null) return '';
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
}

/**
 * Normalise an `ask_user` `questions` array into plain views.
 *
 * Mirrors the shape the backend normalises to (`backend/open_webui/utils/ask_user.py`):
 * `{ id, header, question, options: [{ label, description }], allow_other }`. Returns
 * `null` when the array does not match that shape, so the caller can fall back to
 * pretty-printed JSON instead of dropping data.
 *
 * @param {unknown} value
 * @param {boolean} [defaultAllowOther] Top-level `allow_other`, used when a question
 *   omits its own. Matches `AskUserCard.svelte`'s `question.allow_other ?? allowOther`.
 * @returns {AskUserQuestionView[] | null}
 */
export function formatAskUserQuestions(value, defaultAllowOther = true) {
	if (!Array.isArray(value) || value.length === 0) return null;

	/** @type {AskUserQuestionView[]} */
	const views = [];
	for (const question of value) {
		if (!question || typeof question !== 'object' || Array.isArray(question)) return null;
		const entry = /** @type {Record<string, unknown>} */ (question);

		const rawOptions = entry.options;
		if (!Array.isArray(rawOptions)) return null;
		/** @type {Array<{ label: string, description: string }>} */
		const options = [];
		for (const option of rawOptions) {
			if (!option || typeof option !== 'object' || Array.isArray(option)) return null;
			const optionEntry = /** @type {Record<string, unknown>} */ (option);
			options.push({
				label: stringValue(optionEntry.label),
				description: stringValue(optionEntry.description)
			});
		}

		views.push({
			header: stringValue(entry.header),
			question: stringValue(entry.question),
			options,
			allowOther:
				entry.allow_other === undefined || entry.allow_other === null
					? defaultAllowOther
					: Boolean(entry.allow_other)
		});
	}

	return views;
}

/**
 * @param {string | undefined} value
 * @returns {string}
 */
function normalizeLang(value) {
	const lang = (value ?? '').trim().toLowerCase();
	return /^[a-z0-9+#-]{1,20}$/.test(lang) ? lang : '';
}

/**
 * @param {string} name Canonical tool name.
 * @param {Record<string, unknown>} args
 * @returns {string}
 */
function defaultLang(name, args) {
	const declared = normalizeLang(
		typeof args.lang === 'string'
			? args.lang
			: typeof args.language === 'string'
				? args.language
				: ''
	);
	if (declared) return declared;
	return name === 'execute_code' ? 'python' : '';
}

/**
 * Decide how one argument value should be rendered.
 *
 * A plain `JSON.stringify` for every object (the previous behaviour) collapsed
 * multi-line code and printed `ask_user` questions as one unreadable line.
 *
 * @param {string} name Canonical tool name.
 * @param {string} key
 * @param {unknown} value
 * @param {Record<string, unknown>} args Sibling arguments, for `lang`.
 * @returns {ToolArgumentRow}
 */
function formatArgumentValue(name, key, value, args) {
	if (name === 'ask_user' && key === 'questions') {
		const questions = formatAskUserQuestions(value, args.allow_other !== false);
		if (questions) return { key, kind: 'questions', questions };
	}

	if (value === null) {
		// Keep the key visible; the previous template rendered `JSON.stringify(null)`.
		return { key, kind: 'scalar', text: 'null' };
	}

	if (typeof value === 'object' && value !== null) {
		return { key, kind: 'json', text: JSON.stringify(value, null, 2) };
	}

	if (typeof value === 'string') {
		if (CODE_ARG_KEYS.has(key)) {
			return { key, kind: 'code', text: value, lang: defaultLang(name, args) };
		}
		if (value.includes('\n')) {
			return { key, kind: 'text', text: value };
		}
	}

	return { key, kind: 'scalar', text: stringValue(value) };
}

/**
 * Turn a tool call's arguments into ordered, kind-tagged rows for the expanded
 * tool-call card. Only the presentation of values changes; argument names and
 * order are preserved.
 *
 * Returns `null` when the arguments are not a JSON object, so callers keep the
 * existing raw-arguments fallback.
 *
 * @param {string | null | undefined} name Raw tool name.
 * @param {Record<string, unknown> | null | undefined} args Parsed arguments.
 * @returns {ToolArgumentRow[] | null}
 */
export function formatToolArguments(name, args) {
	if (!args || typeof args !== 'object' || Array.isArray(args)) return null;

	const canonical = canonicalToolName(name ?? '');
	return Object.entries(args).map(([key, value]) =>
		formatArgumentValue(canonical, key, value, args)
	);
}

/**
 * @typedef {object} ToolResultSearchItem
 * @property {string} title
 * @property {string} link
 * @property {string} snippet
 */

/**
 * @typedef {object} ToolResultQuestionAnswer
 * @property {string} header
 * @property {string} question
 * @property {string} answer '' when unanswered.
 * @property {boolean} isOther True when the answer was free text rather than an option.
 */

/**
 * @typedef {object} ToolResultView
 * @property {'error' | 'search' | 'terminal' | 'markdown' | 'qa'} kind How the result should be rendered.
 * @property {string} [text] Message for the `error` kind.
 * @property {ToolResultSearchItem[]} [results] Value for the `search` kind.
 * @property {string} [stdout] Value for the `terminal` kind.
 * @property {string} [stderr] Value for the `terminal` kind.
 * @property {string} [result] Value for the `terminal` kind.
 * @property {'answered' | 'cancelled'} [status] Value for the `qa` kind.
 * @property {ToolResultQuestionAnswer[]} [questions] Value for the `qa` kind.
 */

/**
 * Unwrap nested JSON-encoded strings, mirroring the components' iterative parse.
 *
 * @param {unknown} value
 * @returns {unknown}
 */
function parseJSONValue(value) {
	let parsed = value;
	while (typeof parsed === 'string') {
		try {
			parsed = JSON.parse(parsed);
		} catch {
			break;
		}
	}
	return parsed;
}

/**
 * Detect the error envelope shared by the built-in tools and return its message.
 *
 * `builtin.py` reports failures as `{error}` (search/fetch/execute) or
 * `{status: 'error', error}` (`ask_user`), and some tools use
 * `{success: false, message}`. Anything else returns `null`.
 *
 * @param {unknown} parsed Already-parsed result.
 * @returns {string | null}
 */
function extractToolError(parsed) {
	if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
	const value = /** @type {Record<string, unknown>} */ (parsed);

	const error = value.error;
	const errorMessage =
		typeof error === 'string' && error.trim()
			? error
			: typeof error === 'object' && error !== null
				? stringValue(error)
				: '';

	const status = typeof value.status === 'string' ? value.status.trim().toLowerCase() : '';
	if (status === 'error' || status === 'failed') {
		return errorMessage || stringValue(value.message) || 'Error';
	}

	if (errorMessage) return errorMessage;

	const message = value.message;
	const hasMessage =
		(typeof message === 'string' && message.trim()) ||
		(typeof message === 'object' && message !== null);
	if ((value.success === false || value.ok === false) && hasMessage) {
		return typeof message === 'string' ? message : stringValue(message);
	}

	return null;
}

/**
 * @param {unknown} parsed
 * @returns {ToolResultView | null}
 */
function formatSearchResult(parsed) {
	if (!Array.isArray(parsed)) return null;

	/** @type {ToolResultSearchItem[]} */
	const results = [];
	for (const entry of parsed) {
		if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return null;
		const item = /** @type {Record<string, unknown>} */ (entry);
		const title = stringValue(item.title);
		const link = stringValue(item.link);
		if (!title && !link) return null;
		results.push({ title, link, snippet: stringValue(item.snippet) });
	}

	return { kind: 'search', results };
}

/**
 * `fetch_url` returns extracted page text (occasionally valid JSON, e.g. an API
 * response). JSON-shaped pages keep the pretty-JSON fallback; everything else is
 * rendered as Markdown.
 *
 * @param {string} raw
 * @param {unknown} parsed
 * @returns {ToolResultView | null}
 */
function formatFetchResult(raw, parsed) {
	if (parsed !== null && typeof parsed === 'object') return null;
	if (typeof parsed === 'string') {
		return parsed.trim() ? { kind: 'markdown', text: parsed } : null;
	}
	return raw.trim() ? { kind: 'markdown', text: raw } : null;
}

/**
 * @param {unknown} parsed
 * @returns {ToolResultView | null}
 */
function formatTerminalResult(parsed) {
	if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
	const value = /** @type {Record<string, unknown>} */ (parsed);
	if (!('stdout' in value) && !('stderr' in value) && !('result' in value)) return null;

	return {
		kind: 'terminal',
		stdout: stringValue(value.stdout),
		stderr: stringValue(value.stderr),
		result: stringValue(value.result)
	};
}

/**
 * `ask_user` returns `{status, answers}` keyed by question id. Rebuild the Q&A
 * from the call's normalised `questions` so the prompt text is available; answer
 * values are the `AskUserCard` drafts (`{type: 'option', label}` /
 * `{type: 'other', text}`).
 *
 * @param {unknown} parsed
 * @param {Record<string, unknown> | null} args
 * @returns {ToolResultView | null}
 */
function formatAskUserResult(parsed, args) {
	if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
	const value = /** @type {Record<string, unknown>} */ (parsed);
	if (!('answers' in value) && typeof value.status !== 'string') return null;

	const status = value.status === 'cancelled' ? 'cancelled' : 'answered';
	const answers =
		value.answers && typeof value.answers === 'object' && !Array.isArray(value.answers)
			? /** @type {Record<string, unknown>} */ (value.answers)
			: {};

	/** @type {ToolResultQuestionAnswer[]} */
	const questions = [];
	const rawQuestions = Array.isArray(args?.questions) ? args.questions : [];
	let found = false;
	for (const entry of rawQuestions) {
		if (!entry || typeof entry !== 'object' || Array.isArray(entry)) continue;
		const question = /** @type {Record<string, unknown>} */ (entry);
		const id = stringValue(question.id);
		if (!id) continue;
		found = true;
		questions.push({
			header: stringValue(question.header) || stringValue(question.question),
			question: stringValue(question.question),
			...answerView(answers[id])
		});
	}

	// Without matching arguments (e.g. an older stored message) keep the answers
	// visible keyed by id rather than dropping them.
	if (!found) {
		for (const [id, answer] of Object.entries(answers)) {
			questions.push({ header: id, question: '', ...answerView(answer) });
		}
	}

	return { kind: 'qa', status, questions };
}

/**
 * @param {unknown} answer
 * @returns {{ answer: string, isOther: boolean }}
 */
function answerView(answer) {
	if (typeof answer === 'string') return { answer, isOther: false };
	if (!answer || typeof answer !== 'object' || Array.isArray(answer))
		return { answer: '', isOther: false };
	const value = /** @type {Record<string, unknown>} */ (answer);
	if (value.type === 'other') return { answer: stringValue(value.text), isOther: true };
	return { answer: stringValue(value.label), isOther: false };
}

/**
 * Decide how a completed tool call's result should be rendered.
 *
 * The previous behaviour stringified every object result as pretty JSON, which is
 * a poor fit for the built-ins whose output is structured and human-meaningful
 * (`search_web` result lists, `execute_code` stdout/stderr, `ask_user` answers,
 * `fetch_url` page text). Returns `null` for unknown/MCP tools and for shapes
 * that do not match, so the caller keeps the pretty-JSON fallback rather than
 * dropping data.
 *
 * @param {string | null | undefined} name Raw tool name.
 * @param {string} result Raw result string as carried on the token.
 * @param {Record<string, unknown> | null} [args] Parsed arguments, for `ask_user`.
 * @returns {ToolResultView | null}
 */
export function formatToolResult(name, result, args) {
	const raw =
		typeof result === 'string'
			? result
			: result === undefined || result === null
				? ''
				: stringValue(result);
	if (!raw.trim()) return null;

	const canonical = canonicalToolName(name ?? '');
	const parsed = parseJSONValue(raw);

	const error = extractToolError(parsed);
	if (error) return { kind: 'error', text: error };

	switch (canonical) {
		case 'search_web':
			return formatSearchResult(parsed);
		case 'fetch_url':
			return formatFetchResult(raw, parsed);
		case 'execute_code':
			return formatTerminalResult(parsed);
		case 'ask_user':
			return formatAskUserResult(parsed, args ?? null);
		default:
			return null;
	}
}
