# Memory Tools TODO

## `add_memory` default type mismatch

`add_memory`'s builtin tool signature defaults `type='user'`
(`backend/open_webui/tools/builtin.py`), while `AddMemoryForm` defaults
`'context'` and `normalize_memory_type` coerces anything not exactly `'user'` to
`'context'` (`backend/open_webui/models/memories.py`). A write that omits `type`
therefore files as `user` — the inverse of the read-side misfire the type-filter
guidance change addressed.

Decide: set the tool default to `'context'`, or keep `'user'` and state that
default in the docstring.
