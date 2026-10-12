import { PyodideSandboxHost } from '$lib/pyodide/pyodideSandboxHost';

/**
 * Create a Pyodide runtime. Code always executes in a null-origin sandboxed
 * iframe, which isolates it from the session, cookies, local storage, and the
 * app's own endpoints. File persistence is handled parent-side (see
 * `pyodideFileSync.ts`), so there is a single execution backend regardless of
 * the `enable_pyodide_file_persistence` setting.
 */
export const createPyodideWorker = (): Worker => new PyodideSandboxHost() as unknown as Worker;
