import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
	formatAskUserQuestions,
	formatToolArguments,
	formatToolLabel,
	formatToolResult,
	getToolGroupSummary,
	getToolLabel,
	isKeyLabel,
	parseToolArguments,
	type ToolLabelText,
	type ToolLabelToken
} from './toolLabels.js';

/** Minimal `t()` that interpolates `{{TOKEN}}`, mirroring i18next with en-US keys. */
const stubI18n = {
	t: (key: string, values: Record<string, unknown> = {}) =>
		key.replace(/\{\{(\w+)\}\}/g, (_, token: string) => String(values[token] ?? ''))
};

/** Render a descriptor the way the components do, without pulling in i18next. */
function render(part: ToolLabelText): string {
	return formatToolLabel(part, stubI18n);
}

function args(value: unknown): string {
	return JSON.stringify(value);
}

function localeKeys(locale: string): Set<string> {
	const path = fileURLToPath(
		new URL(`../i18n/locales/${locale}/translation.json`, import.meta.url)
	);
	return new Set(Object.keys(JSON.parse(readFileSync(path, 'utf-8'))));
}

describe('parseToolArguments', () => {
	it('unwraps nested JSON-encoded argument strings', () => {
		expect(parseToolArguments(JSON.stringify(JSON.stringify({ query: 'x' })))).toEqual({
			query: 'x'
		});
	});

	it('returns null for scalars and arrays', () => {
		expect(parseToolArguments('5')).toBeNull();
		expect(parseToolArguments('[1,2]')).toBeNull();
		expect(parseToolArguments('')).toBeNull();
		expect(parseToolArguments(undefined)).toBeNull();
	});

	it('returns null for malformed input', () => {
		expect(parseToolArguments('{not json')).toBeNull();
	});
});

describe('getToolLabel row labels', () => {
	it('describes a web search using its query', () => {
		const label = getToolLabel('search_web', { query: 'svelte 5 runes' });
		expect(render(label.done)).toBe('Searched the web for "svelte 5 runes"');
		expect(render(label.active)).toBe('Searching the web for "svelte 5 runes"');
		expect(label.category).toBe('search');
	});

	it('falls back to a bare web search when the query is missing', () => {
		expect(render(getToolLabel('search_web', {}).done)).toBe('Searched the web');
		expect(render(getToolLabel('search_web', null).active)).toBe('Searching the web…');
	});

	it('describes a fetched page by host and path', () => {
		const label = getToolLabel('fetch_url', { url: 'https://docs.example.com/guide/intro?x=1' });
		expect(render(label.done)).toBe('Read docs.example.com/guide/intro');
		expect(render(label.active)).toBe('Reading docs.example.com/guide/intro…');
		expect(label.category).toBe('read');
	});

	it('preserves the scheme of non-http urls', () => {
		expect(render(getToolLabel('fetch_url', { url: 'HTTP://Example.com/A' }).done)).toBe(
			'Read example.com/A'
		);
		expect(render(getToolLabel('fetch_url', { url: 'ftp://files.example.com/x' }).done)).toBe(
			'Read files.example.com/x'
		);
	});

	it('describes code execution without leaking the code', () => {
		const label = getToolLabel('execute_code', { code: 'print(1)' });
		expect(render(label.done)).toBe('Ran Python code');
		expect(render(label.active)).toBe('Running Python code…');
		expect(label.category).toBe('code');
	});

	it('counts questions and switches tense', () => {
		const one = getToolLabel('ask_user', { questions: [{ id: 'a' }] });
		expect(render(one.done)).toBe('Asked 1 question');
		expect(render(one.active)).toBe('Asking 1 question…');

		const many = getToolLabel('ask_user', { questions: [{ id: 'a' }, { id: 'b' }] });
		expect(render(many.done)).toBe('Asked 2 questions');
		expect(render(many.active)).toBe('Asking 2 questions…');
		expect(many.category).toBe('question');
	});

	it('handles an ask_user without parseable questions', () => {
		expect(render(getToolLabel('ask_user', null).done)).toBe('Asked a question');
		expect(render(getToolLabel('ask_user', null).active)).toBe('Asking a question…');
		expect(getToolLabel('ask_user', { questions: 'nope' }).category).toBe('question');
	});

	it('resolves aliases and namespaced tool names', () => {
		expect(getToolLabel('web_search', { query: 'a' }).category).toBe('search');
		expect(getToolLabel('webfetch', { url: 'https://a.com' }).category).toBe('read');
		expect(getToolLabel('run_python', {}).category).toBe('code');
		expect(getToolLabel('request_user_input', {}).category).toBe('question');
		// Namespaced aliases...
		expect(getToolLabel('mcp.web_search', { query: 'a' }).category).toBe('search');
		expect(getToolLabel('server.fetch', { url: 'https://a.com' }).category).toBe('read');
		// ...and namespaced canonical names.
		expect(render(getToolLabel('mcp.search_web', { query: 'q' }).done)).toBe(
			'Searched the web for "q"'
		);
		expect(getToolLabel('server.execute_code', {}).category).toBe('code');
		expect(render(getToolLabel('mcp.ask_user', { questions: [{ id: 'a' }] }).done)).toBe(
			'Asked 1 question'
		);
		expect(render(getToolLabel('mcp.fetch_url', { url: 'https://a.com/x' }).done)).toBe(
			'Read a.com/x'
		);
	});

	it('summarises an unknown tool from its first meaningful argument', () => {
		const label = getToolLabel('my_custom_tool', { path: '/etc/hosts', limit: 10 });
		expect(render(label.done)).toBe('my_custom_tool — /etc/hosts');
	});

	it('prefers query over url and falls back to primitives', () => {
		expect(render(getToolLabel('my_tool', { url: 'https://a.com', query: 'needle' }).done)).toBe(
			'my_tool — needle'
		);
		expect(render(getToolLabel('my_tool', { limit: 10 }).done)).toBe('my_tool — 10');
		expect(render(getToolLabel('my_tool', { enabled: true }).done)).toBe('my_tool — true');
	});

	it('truncates long summaries', () => {
		const long = 'x'.repeat(200);
		const rendered = render(getToolLabel('my_custom_tool', { query: long }).done);
		expect(rendered.startsWith('my_custom_tool — ')).toBe(true);
		expect(rendered.length).toBeLessThan(100);
		expect(rendered.endsWith('…')).toBe(true);
	});

	it('falls back to the bare tool name when there is nothing to summarise', () => {
		expect(render(getToolLabel('mystery', {}).done)).toBe('mystery');
		expect(render(getToolLabel('', {}).done)).toBe('tool');
	});

	it('keeps the subagent task title as data', () => {
		const label = getToolLabel('Subagent: "summarise the repo"', { task: 'summarise the repo' });
		expect(render(label.done)).toBe('Subagent: "summarise the repo"');
		expect(label.category).toBe('other');
		expect(render(getToolLabel('Background subagent: "x"', {}).done)).toBe(
			'Background subagent: "x"'
		);
	});

	it('still treats legacy hyphenated sub-agent titles as data', () => {
		expect(render(getToolLabel('Sub-agent: "old"', {}).done)).toBe('Sub-agent: "old"');
		expect(render(getToolLabel('Background sub-agent: "old"', {}).done)).toBe(
			'Background sub-agent: "old"'
		);
	});

	it('resolves the canonical subagent name and its aliases', () => {
		expect(render(getToolLabel('subagent', {}).done)).toBe('Subagent');
		expect(render(getToolLabel('delegate_task', {}).done)).toBe('Subagent');
		expect(render(getToolLabel('task', {}).done)).toBe('Subagent');
		expect(render(getToolLabel('mcp.subagent', {}).done)).toBe('Subagent');
		expect(render(getToolLabel('mcp.delegate_task', {}).done)).toBe('Subagent');
		expect(getToolLabel('subagent', {}).category).toBe('other');
	});

	it('accepts pre-parsed arguments directly', () => {
		expect(render(getToolLabel('search_web', parseToolArguments(args({ query: 'q' }))).done)).toBe(
			'Searched the web for "q"'
		);
	});
});

describe('formatToolArguments', () => {
	const question = (id: string, header: string, text: string, optionLabels: string[]) => ({
		id,
		header,
		question: text,
		options: optionLabels.map((label) => ({ label, description: `${label} description` }))
	});

	it('returns null when the arguments are not an object', () => {
		expect(formatToolArguments('execute_code', null)).toBeNull();
		expect(formatToolArguments('execute_code', undefined)).toBeNull();
		expect(
			formatToolArguments('execute_code', [1, 2] as unknown as Record<string, unknown>)
		).toBeNull();
	});

	it('keeps argument order and scalar values inline', () => {
		const rows = formatToolArguments('my_tool', { path: '/etc/hosts', limit: 10, dry: true });
		expect(rows?.map((row) => row.kind)).toEqual(['scalar', 'scalar', 'scalar']);
		expect(rows?.map((row) => row.text)).toEqual(['/etc/hosts', '10', 'true']);
	});

	it('preserves multi-line Python code in a pre block', () => {
		const code = 'import math\n\nprint(math.pi)\n';
		const rows = formatToolArguments('execute_code', { code });
		expect(rows).toHaveLength(1);
		expect(rows?.[0].kind).toBe('code');
		expect(rows?.[0].text).toBe(code);
		expect(rows?.[0].lang).toBe('python');
	});

	it('uses a declared language for code and drops invalid ones', () => {
		expect(
			formatToolArguments('execute_code', { code: 'x = 1', lang: 'javascript' })?.[0].lang
		).toBe('javascript');
		expect(formatToolArguments('execute_code', { code: 'x = 1', language: 'R' })?.[0].lang).toBe(
			'r'
		);
		expect(
			formatToolArguments('execute_code', { code: 'x = 1', lang: 'not a lang' })?.[0].lang
		).toBe('python');
	});

	it('treats a multi-line non-code string as text', () => {
		const rows = formatToolArguments('my_tool', { notes: 'line one\nline two' });
		expect(rows?.[0].kind).toBe('text');
		expect(rows?.[0].text).toBe('line one\nline two');
	});

	it('pretty-prints nested objects and arrays as json', () => {
		const rows = formatToolArguments('my_tool', { config: { a: [1, 2] } });
		expect(rows?.[0].kind).toBe('json');
		expect(rows?.[0].text).toBe(JSON.stringify({ a: [1, 2] }, null, 2));
	});

	it('renders ask_user questions as structured views', () => {
		const rows = formatToolArguments('ask_user', {
			questions: [
				question('a', 'Database', 'Which store?', ['Postgres', 'SQLite']),
				question('b', 'Deploy', 'Where?', ['Cloud'])
			]
		});
		expect(rows?.[0].kind).toBe('questions');
		const views = rows?.[0].questions ?? [];
		expect(views).toHaveLength(2);
		expect(views[0]).toEqual({
			header: 'Database',
			question: 'Which store?',
			options: [
				{ label: 'Postgres', description: 'Postgres description' },
				{ label: 'SQLite', description: 'SQLite description' }
			],
			allowOther: true
		});
	});

	it('honours allow_other on ask_user questions', () => {
		const rows = formatToolArguments('ask_user', {
			questions: [{ ...question('a', 'H', 'Q', ['x']), allow_other: false }]
		});
		expect(rows?.[0].questions?.[0].allowOther).toBe(false);
	});

	it('falls back to json when ask_user questions do not match the shape', () => {
		const rows = formatToolArguments('ask_user', { questions: [{ nonsense: true }] });
		expect(rows?.[0].kind).toBe('json');
		expect(formatAskUserQuestions('nope')).toBeNull();
		expect(formatAskUserQuestions([])).toBeNull();
		expect(formatAskUserQuestions([{ header: 'h', question: 'q' }])).toBeNull();
	});

	it('classifies aliased and namespaced execute_code calls as code', () => {
		expect(formatToolArguments('run_python', { code: 'a\nb' })?.[0].kind).toBe('code');
		expect(formatToolArguments('server.execute_code', { code: 'a\nb' })?.[0].kind).toBe('code');
	});

	it('marks a `source` argument as code even for an unknown tool', () => {
		expect(formatToolArguments('my_tool', { source: 'a\nb' })?.[0].kind).toBe('code');
	});

	it('keeps null argument values visible', () => {
		const rows = formatToolArguments('my_tool', { param: null });
		expect(rows?.[0].kind).toBe('scalar');
		expect(rows?.[0].text).toBe('null');
	});

	it('falls back to the top-level allow_other when a question omits it', () => {
		const args = {
			questions: [{ header: 'H', question: 'Q', options: [{ label: 'x', description: 'd' }] }],
			allow_other: false
		};
		expect(formatToolArguments('ask_user', args)?.[0].questions?.[0].allowOther).toBe(false);
		expect(formatAskUserQuestions(args.questions, false)?.[0]?.allowOther).toBe(false);
	});
});

describe('formatToolResult', () => {
	it('returns null for empty results, unknown tools, and mismatched shapes', () => {
		expect(formatToolResult('search_web', '')).toBeNull();
		expect(formatToolResult('search_web', undefined as unknown as string)).toBeNull();
		expect(formatToolResult('mcp.mystery_tool', JSON.stringify({ a: 1 }))).toBeNull();
		// A search payload that is not a list of result objects stays JSON.
		expect(formatToolResult('search_web', JSON.stringify({ query: 'x' }))).toBeNull();
		expect(formatToolResult('execute_code', JSON.stringify({ unrelated: true }))).toBeNull();
	});

	it('renders search_web results as title/link/snippet', () => {
		const view = formatToolResult(
			'search_web',
			JSON.stringify([
				{ title: 'A', link: 'https://a.com/x', snippet: 'snip a' },
				{ title: 'B', link: 'https://b.com', snippet: 'snip b' }
			])
		);
		expect(view?.kind).toBe('search');
		expect(view?.results).toEqual([
			{ title: 'A', link: 'https://a.com/x', snippet: 'snip a' },
			{ title: 'B', link: 'https://b.com', snippet: 'snip b' }
		]);
	});

	it('keeps an empty search result list but reports it as search', () => {
		const view = formatToolResult('search_web', '[]');
		expect(view?.kind).toBe('search');
		expect(view?.results).toEqual([]);
	});

	it('resolves aliases and namespaced names for results too', () => {
		expect(formatToolResult('web_search', '[]')?.kind).toBe('search');
		expect(formatToolResult('mcp.search_web', '[]')?.kind).toBe('search');
		expect(formatToolResult('server.execute_code', JSON.stringify({ stdout: 'x' }))?.kind).toBe(
			'terminal'
		);
	});

	it('renders execute_code stdout/stderr/result as terminal output', () => {
		const view = formatToolResult(
			'execute_code',
			JSON.stringify({ status: 'success', stdout: 'hello\n', stderr: '', result: '42' })
		);
		expect(view).toEqual({ kind: 'terminal', stdout: 'hello\n', stderr: '', result: '42' });
	});

	it('renders fetch_url text as markdown and keeps JSON pages as-is', () => {
		expect(formatToolResult('fetch_url', '# Title\n\nBody')).toEqual({
			kind: 'markdown',
			text: '# Title\n\nBody'
		});
		expect(formatToolResult('fetch_url', '  ')).toBeNull();
		// A page that parses to JSON is not prose; fall back to pretty JSON.
		expect(formatToolResult('fetch_url', JSON.stringify({ key: 'value' }))).toBeNull();
		expect(formatToolResult('fetch_url', '{"key": "value"}')).toBeNull();
	});

	it('builds ask_user Q&A from the questions and the answer drafts', () => {
		const args = {
			questions: [
				{
					id: 'db',
					header: 'Database',
					question: 'Which store?',
					options: [{ label: 'Postgres', description: 'd' }]
				},
				{
					id: 'deploy',
					header: 'Deploy',
					question: 'Where?',
					options: [{ label: 'Cloud', description: 'd' }]
				}
			]
		};
		const view = formatToolResult(
			'ask_user',
			JSON.stringify({
				status: 'answered',
				answers: {
					db: { type: 'option', option_index: 0, label: 'Postgres', description: 'd' },
					deploy: { type: 'other', text: 'On-prem' }
				}
			}),
			args
		);
		expect(view?.kind).toBe('qa');
		expect(view?.status).toBe('answered');
		expect(view?.questions).toEqual([
			{ header: 'Database', question: 'Which store?', answer: 'Postgres', isOther: false },
			{ header: 'Deploy', question: 'Where?', answer: 'On-prem', isOther: true }
		]);
	});

	it('reports a cancelled ask_user and unanswered questions', () => {
		const view = formatToolResult(
			'ask_user',
			JSON.stringify({ status: 'cancelled', answers: {} }),
			{
				questions: [
					{ id: 'a', header: 'H', question: 'Q', options: [{ label: 'x', description: 'd' }] }
				]
			}
		);
		expect(view?.status).toBe('cancelled');
		expect(view?.questions?.[0].answer).toBe('');
	});

	it('falls back to answer-keyed entries when the arguments are unavailable', () => {
		const view = formatToolResult(
			'ask_user',
			JSON.stringify({ status: 'answered', answers: { db: { type: 'option', label: 'SQLite' } } })
		);
		expect(view?.questions).toEqual([
			{ header: 'db', question: '', answer: 'SQLite', isOther: false }
		]);
	});

	it('renders the error envelope shared by the built-ins', () => {
		expect(formatToolResult('search_web', JSON.stringify({ error: 'boom' }))).toEqual({
			kind: 'error',
			text: 'boom'
		});
		expect(
			formatToolResult('ask_user', JSON.stringify({ status: 'error', error: 'nope' }))
		).toEqual({
			kind: 'error',
			text: 'nope'
		});
		expect(
			formatToolResult('execute_code', JSON.stringify({ status: 'failed', message: 'x' }))
		).toEqual({ kind: 'error', text: 'x' });
		// An `error` field on an otherwise unknown tool is still surfaced.
		expect(formatToolResult('mcp.custom', JSON.stringify({ error: 'bad' }))?.kind).toBe('error');
	});

	it('unwraps JSON-encoded string results before classifying', () => {
		const nested = JSON.stringify(JSON.stringify({ stdout: 'out', stderr: '', result: '' }));
		expect(formatToolResult('execute_code', nested)).toEqual({
			kind: 'terminal',
			stdout: 'out',
			stderr: '',
			result: ''
		});
	});
});

describe('getToolGroupSummary', () => {
	const call = (name: string, argumentsValue: unknown = {}, status = 'completed') => ({
		attributes: { type: 'tool_calls', name, arguments: args(argumentsValue), status }
	});

	it('renders a search as a single statement', () => {
		expect(getToolGroupSummary([call('search_web', { query: 'a' })]).map(render)).toEqual([
			'searched the web'
		]);
	});

	it('counts fetched pages', () => {
		const parts = getToolGroupSummary([
			call('fetch_url', { url: 'https://a.com' }),
			call('fetch_url', { url: 'https://b.com' })
		]);
		expect(parts.map(render)).toEqual(['read 2 pages']);
	});

	it('uses the active tense while running', () => {
		expect(
			getToolGroupSummary([call('fetch_url', { url: 'https://a.com' }, 'in_progress')], true).map(
				render
			)
		).toEqual(['reading 1 page…']);
		expect(
			getToolGroupSummary([call('search_web', { query: 'a' }, 'in_progress')], true).map(render)
		).toEqual(['searching the web…']);
		expect(
			getToolGroupSummary([call('execute_code', {}, 'in_progress')], true).map(render)
		).toEqual(['running 1 code block…']);
		expect(
			getToolGroupSummary(
				[call('ask_user', { questions: [{ id: 'a' }] }, 'in_progress')],
				true
			).map(render)
		).toEqual(['asking 1 question…']);
	});

	it('orders parts by first appearance and dedupes groups', () => {
		const parts = getToolGroupSummary([
			call('fetch_url', { url: 'https://a.com' }),
			call('search_web', { query: 'a' }),
			call('fetch_url', { url: 'https://b.com' })
		]);
		expect(parts.map(render)).toEqual(['read 2 pages', 'searched the web']);
	});

	it('includes code interpreter tokens with the code-block phrasing', () => {
		const parts = getToolGroupSummary([
			{ attributes: { type: 'code_interpreter' } },
			{ attributes: { type: 'code_interpreter' } }
		]);
		expect(parts.map(render)).toEqual(['ran 2 code blocks']);
	});

	it('counts questions across calls', () => {
		const parts = getToolGroupSummary([
			call('ask_user', { questions: [{ id: 'a' }, { id: 'b' }] })
		]);
		expect(parts.map(render)).toEqual(['asked 2 questions']);
		expect(
			getToolGroupSummary([call('ask_user', { questions: [{ id: 'a' }] })]).map(render)
		).toEqual(['asked 1 question']);
	});

	it('keeps unknown tools by name and dedupes them', () => {
		const parts = getToolGroupSummary([call('mystery'), call('mystery'), call('other_tool')]);
		expect(parts.map(render)).toEqual(['mystery', 'other_tool']);
	});

	it('ignores non-tool tokens and returns empty when nothing is groupable', () => {
		expect(getToolGroupSummary([{ attributes: { type: 'reasoning' } }])).toEqual([]);
		expect(getToolGroupSummary([])).toEqual([]);
	});

	it('mixes categories in first-seen order', () => {
		const parts = getToolGroupSummary([
			call('execute_code', { code: 'x' }),
			call('search_web', { query: 'a' }),
			{ attributes: { type: 'code_interpreter' } }
		]);
		expect(parts.map(render)).toEqual(['ran 2 code blocks', 'searched the web']);
	});
});

describe('i18n catalogue coverage', () => {
	// The module's dynamic descriptors are invisible to i18next-parser, so every key
	// it can emit is guarded here instead. Descriptors are collected by *calling* the
	// exported functions (not by regex over source), so ternary-built keys are
	// covered too — a regex-based check previously missed every group key.
	function moduleSource(): string {
		return readFileSync(fileURLToPath(new URL('./toolLabels.js', import.meta.url)), 'utf-8');
	}

	function emittedKeys(): Set<string> {
		const keys = new Set<string>();
		const collect = (part: ToolLabelText) => {
			if (isKeyLabel(part)) keys.add(part.key);
		};

		const rows: Array<[string, Record<string, unknown> | null]> = [
			['search_web', { query: 'q' }],
			['search_web', {}],
			['fetch_url', { url: 'https://a.com/x' }],
			['fetch_url', {}],
			['execute_code', {}],
			['ask_user', { questions: [{ id: 'a' }] }],
			['ask_user', { questions: [{ id: 'a' }, { id: 'b' }] }],
			['ask_user', null],
			['my_tool', { query: 'q' }],
			['my_tool', {}],
			['delegate_task', {}],
			['Subagent: "x"', {}],
			['Sub-agent: "legacy"', {}]
		];
		for (const [name, args] of rows) {
			const label = getToolLabel(name, args);
			collect(label.active);
			collect(label.done);
		}

		const call = (name: string, argumentsValue: unknown, status: string): ToolLabelToken => ({
			attributes: {
				type: 'tool_calls',
				name,
				arguments: JSON.stringify(argumentsValue),
				status
			}
		});
		for (const active of [false, true]) {
			const status = active ? 'in_progress' : 'completed';
			for (const count of [1, 2]) {
				const repeat = <T>(factory: () => T) => Array.from({ length: count }, factory);
				const groups = [
					getToolGroupSummary(
						repeat(() => call('search_web', { query: 'a' }, status)),
						active
					),
					getToolGroupSummary(
						repeat(() => call('fetch_url', { url: 'https://a.com' }, status)),
						active
					),
					getToolGroupSummary(
						repeat(() => call('execute_code', {}, status)),
						active
					),
					getToolGroupSummary(
						repeat(() => call('ask_user', { questions: [{ id: 'a' }] }, status)),
						active
					),
					getToolGroupSummary(
						repeat(() => ({ attributes: { type: 'code_interpreter' } })),
						active
					)
				];
				for (const group of groups) for (const part of group) collect(part);
			}
		}

		return keys;
	}

	it('has every emitted key present in en-US and zh-CN', () => {
		const emitted = emittedKeys();

		expect(emitted.size).toBeGreaterThan(20);
		for (const locale of ['en-US', 'zh-CN']) {
			const available = localeKeys(locale);
			const missing = [...emitted].filter((key) => !available.has(key));
			expect(missing, `${locale} is missing ${missing.join(', ')}`).toEqual([]);
		}
	});

	it('routes every emitted key through the parser-visible translate switch', () => {
		const translationLiterals = new Set(
			[...moduleSource().matchAll(/\bt\('([^']+)'/g)].map((match) => match[1])
		);

		const unrouted = [...emittedKeys()].filter((key) => !translationLiterals.has(key));
		expect(unrouted, `unrouted keys: ${unrouted.join(', ')}`).toEqual([]);
	});

	it('keeps the translate switch free of dead keys', () => {
		const emitted = emittedKeys();
		const translationLiterals = new Set(
			[...moduleSource().matchAll(/\bt\('([^']+)'/g)].map((match) => match[1])
		);

		const dead = [...translationLiterals].filter((key) => !emitted.has(key));
		expect(dead, `dead keys: ${dead.join(', ')}`).toEqual([]);
	});

	it('distinguishes key descriptors from raw text descriptors', () => {
		expect(isKeyLabel({ key: 'x' })).toBe(true);
		expect(isKeyLabel({ text: 'x' })).toBe(false);
	});
});
