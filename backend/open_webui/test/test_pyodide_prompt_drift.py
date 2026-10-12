"""Drift test: the Pyodide prompt text vs. the runtime that implements it (P3).

The ``CODE_INTERPRETER_PYODIDE_*`` prompt constants in ``config.py`` describe
behaviour that lives in ``src/lib/pyodide/*`` (probe/install/retry and file
persistence). The two evolve independently today: the prompt is Python, the
runtime is TypeScript, and nothing fails when one describes an operation the
other no longer performs.

These tests are deliberately *structural*, not prose-exact. They assert that the
operations/behaviours the prompt names are actually implemented in the runtime
(and vice versa for the load-bearing facts), so a rename or removal on either
side breaks a test instead of silently shipping an inaccurate prompt.

They read both files as text and AST-extract the prompt constants without
importing ``open_webui.config`` — importing config runs module-level code and
wipes ``backend/open_webui/static/*`` (see the backend-testing skill), which is
exactly what we must avoid in a test run.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PY = REPO_ROOT / 'backend' / 'open_webui' / 'config.py'
MIDDLEWARE_PY = REPO_ROOT / 'backend' / 'open_webui' / 'utils' / 'middleware.py'
PYODIDE_DIR = REPO_ROOT / 'src' / 'lib' / 'pyodide'

PROMPT_CONSTANTS = (
    'CODE_INTERPRETER_PYODIDE_PROMPT',
    'CODE_INTERPRETER_PYODIDE_PERSISTENCE_PROMPT',
)
FILE_SYSTEM_HEADER = '##### Persistent File System'
PYTHON_STATE_HEADER = '##### Persistent Python State'
ENVIRONMENT_HEADER = '##### Pyodide Environment'


# ---------------------------------------------------------------------------
# extraction helpers
# ---------------------------------------------------------------------------


def _extract_prompt_constants() -> dict[str, str]:
    """Pull the string constants straight out of config.py without importing it."""
    tree = ast.parse(CONFIG_PY.read_text())
    found: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in PROMPT_CONSTANTS
        ):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), (
                f'{node.targets[0].id} must stay a plain string literal for this drift test'
            )
            found[node.targets[0].id] = node.value.value
    missing = set(PROMPT_CONSTANTS) - set(found)
    assert not missing, f'missing prompt constants: {missing}'
    return found


def _runtime_source(*names: str) -> str:
    """Concatenate the named TS runtime modules into one searchable blob."""
    return '\n'.join((PYODIDE_DIR / name).read_text() for name in names)


RUNTIME = _runtime_source(
    'pyodideSandboxHost.ts',
    'pyodideFileSync.ts',
    'pyodideFileStore.ts',
    'pyodidePackages.ts',
    'pyodideRuntimePool.ts',
)


def _base_and_full() -> tuple[str, str]:
    constants = _extract_prompt_constants()
    base = constants['CODE_INTERPRETER_PYODIDE_PROMPT']
    persistence = constants['CODE_INTERPRETER_PYODIDE_PERSISTENCE_PROMPT']
    # Mirror middleware.py: base, then persistence appended when the env flag is on.
    return base, base + persistence


# ---------------------------------------------------------------------------
# 1. Merged file-system section (P3 bullet 1)
# ---------------------------------------------------------------------------


def test_exactly_one_file_system_section():
    base, full = _base_and_full()
    assert base.count(FILE_SYSTEM_HEADER) == 1
    # The persistence text continues the same section; it must not open a new one.
    assert '##### Durable File System' not in full
    constants = _extract_prompt_constants()
    # No header line of its own (an inline textual reference to the section name
    # is fine); the continuation must not open a second section.
    assert not any(
        line.startswith('##### ')
        for line in constants['CODE_INTERPRETER_PYODIDE_PERSISTENCE_PROMPT'].splitlines()
    )


def test_prompt_section_headers_are_exactly_the_expected_set():
    base, full = _base_and_full()
    headers = [line for line in full.splitlines() if line.startswith('##### ')]
    assert headers == [ENVIRONMENT_HEADER, PYTHON_STATE_HEADER, FILE_SYSTEM_HEADER]


def test_file_system_section_distinguishes_session_from_reload():
    base, full = _base_and_full()
    base_lower = base.lower()
    full_lower = full.lower()
    # Session tier: stated in the base section, present even persistence-off.
    assert 'within a session' in base_lower
    assert '/mnt/uploads' in base
    # Reload tier: only in the appended continuation, never implied when off.
    assert 'across page reloads' in full_lower
    assert 'browser restarts' in full_lower
    assert 'across page reloads' not in base_lower
    assert 'browser restarts' not in base_lower


def test_persistence_text_is_only_appended_under_the_env_flag():
    """The merge keeps the reload tier conditional; pin the middleware guard."""
    tree = ast.parse(MIDDLEWARE_PY.read_text())
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    usages = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == 'CODE_INTERPRETER_PYODIDE_PERSISTENCE_PROMPT'
    ]
    assert len(usages) == 2, f'expected legacy + native-FC appends, found {len(usages)}'

    def guarded(node: ast.AST) -> bool:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.If):
                test = current.test
                names = {
                    child.id
                    for child in ast.walk(test)
                    if isinstance(child, ast.Name)
                }
                if 'ENABLE_PYODIDE_FILE_PERSISTENCE' in names:
                    return True
        return False

    assert all(guarded(node) for node in usages)


# ---------------------------------------------------------------------------
# 2. Uploads dir matches the runtime (P3 bullet 4)
# ---------------------------------------------------------------------------


def test_prompt_upload_dir_matches_runtime_constant():
    base, _ = _base_and_full()
    import re

    match = re.search(r"UPLOADS_DIR\s*=\s*'([^']+)'", RUNTIME)
    assert match, 'pyodideFileSync.UPLOADS_DIR must stay a literal for this drift test'
    runtime_dir = match.group(1)
    assert base.count(f'`{runtime_dir}/`') >= 1, f'prompt must reference {runtime_dir}/'
    # The sandbox actually creates that tree.
    assert f"mkdirTree('{runtime_dir}')" in RUNTIME


# ---------------------------------------------------------------------------
# 3. Probe / install / retry operations the prompt names (P3 bullet 4)
# ---------------------------------------------------------------------------


def test_probe_and_install_operations_are_grounded_in_runtime():
    base, _ = _base_and_full()
    sandbox = _runtime_source('pyodideSandboxHost.ts')
    packages = _runtime_source('pyodidePackages.ts')

    # The retry is driven by parsing ModuleNotFoundError, exactly as the prompt says.
    assert 'ModuleNotFoundError' in base
    assert 'No module named' in packages
    assert 'parseMissingModule' in packages
    assert 'parseMissingModule(' in sandbox

    # micropip is the install mechanism the prompt points at, and it is awaited.
    assert 'micropip' in base
    assert 'micropip' in sandbox
    assert '.install(' in sandbox
    assert 'await pyodide.pyimport(' in sandbox

    # Literal dynamic-import probes are recognised by the runtime's extractor.
    assert 'importlib.import_module' in base
    assert '__import__' in base
    assert 'find_spec' in base
    for token in ('import_module', '__import__', 'find_spec'):
        assert token in packages, f'findLiteralDynamicImports must keep matching {token}'
    assert 'findLiteralDynamicImports' in sandbox
    assert 'loadPackagesFromImports' in sandbox


# ---------------------------------------------------------------------------
# 4. Persistence behaviour the durable tier describes (P3 bullet 4)
# ---------------------------------------------------------------------------


def test_durable_tier_is_backed_by_hydrate_flush_and_indexeddb():
    _, full = _base_and_full()
    assert 'indexedDB' in RUNTIME, 'file persistence must be IndexedDB-backed'
    assert 'hydratePyodideFiles' in RUNTIME
    assert 'flushPyodideFiles' in RUNTIME
    # Persistence is user-namespaced and scoped to the uploads tree only.
    assert 'UPLOADS_DIR' in RUNTIME
    assert 'namespace' in RUNTIME
    # The prompt says "stored in the browser"; the store lives in the parent page
    # precisely because the sandbox iframe has no usable IndexedDB.
    assert 'stored in the browser' in full.lower()


def test_python_state_not_persisted_claim_matches_runtime():
    _, full = _base_and_full()
    # Prompt: a reload starts a fresh interpreter; only /mnt/uploads survives.
    assert 'a reload starts a fresh interpreter' in full
    assert 'Python variables and imported modules are not part of it' in full
    # Runtime: hydration restores only the uploads tree, and a runtime is built
    # around a freshly created worker, so variables/modules cannot come back.
    assert 'hydratePyodideFiles' in RUNTIME
    assert 'UPLOADS_DIR' in RUNTIME
    assert 'workerFactory' in RUNTIME
