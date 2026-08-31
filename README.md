# MiniCoder

MiniCoder is a command-line coding agent implemented from first principles for a software engineering assessment.

The project is currently being developed in small, test-backed increments. Do not place a real API key in this repository.

## Current capabilities

MiniCoder provides a synchronous multi-turn model/tool loop, workspace-scoped
file tools, cross-platform command execution, bounded diagnostic output,
context compaction, an explicit plan-before-execution phase,
verification-before-completion, project memory, explicit states, and
sanitized Observer events for the console and optional JSONL traces. Each
process also keeps an exact private session archive, restores the latest
same-workspace context on restart, and uses the configured model for semantic
context maintenance.

Configure any OpenAI-compatible Chat Completions endpoint with tool calling:

```bash
export MINICODER_API_KEY="your-key"
export MINICODER_BASE_URL="https://api.deepseek.com"
export MINICODER_MODEL="deepseek-v4-pro"
python -m minicoder --check-config --workspace .
```

Start an interactive session and keep the same conversation and tool artifacts
until `/exit`, `/quit`, end-of-input, or Ctrl-C:

```bash
minicoder --workspace .
```

The original one-shot form remains available:

```bash
python -m minicoder --workspace . "inspect this project and run its tests"
```

Append a sanitized audit trace whose parent directory already exists:

```bash
python -m minicoder \
  --workspace . \
  --trace ./minicoder-trace.jsonl \
  "inspect this project and run its tests"
```

The trace records event order, model steps, tool names, call IDs, status, error
codes, and character counts. It intentionally excludes API keys, user task text,
displayed plan text, displayed paths and commands, tool argument/output bodies,
final response bodies, and model reasoning content.

Each new interactive user turn receives a fresh model-step allowance while prior
messages and completion evidence remain available to the session.

The default allowance is 40 task-model requests per user turn. It includes the
planning request and every model continuation after a tool result. The one
post-turn maintenance request and any model-assisted context-compaction request
are separate housekeeping calls and do not consume this task loop allowance.
Override the task allowance for a smaller or larger task budget:

```bash
export MINICODER_MAX_STEPS=60
```

## Context budget

MiniCoder uses an approximate character budget to keep each execution request
below the configured model's context window. The default is 180,000 characters.
The estimate now includes messages, the current Tool Registry JSON Schemas, and
space reserved for the next model response. This value remains provider-neutral:
it is not an exact token count or a claim about every compatible model's maximum.

Override it when the selected endpoint has a smaller or larger context window:

```bash
export MINICODER_CONTEXT_BUDGET_CHARS=300000
```

The default response reserve is 8,000 characters (or 10% for an explicitly
small context budget). It can be overridden, but must remain below the total:

```bash
export MINICODER_CONTEXT_RESPONSE_RESERVE_CHARS=12000
```

Every completed or failed external user turn is finalized by one no-tool
maintenance pass that updates a structured rolling session summary. On later task-model
requests, MiniCoder pins that model-written summary, the latest same-workspace
session recovery data, and bounded durable project memory alongside the System
message as quoted data. The current user request and current workspace evidence
always take priority.

When older raw protocol groups still need to be omitted, production composition
uses another no-tool model request to summarize those groups semantically. The
summary preserves requirements, files, tool outcomes, errors, verification and
pending work while excluding private reasoning. A model failure falls back to
the deterministic summary strategy. The complete source conversation remains in
the private session archive either way. If fixed fields alone exceed the budget,
MiniCoder does not send an oversized request and reports actionable configuration
guidance. Maintenance and context-compaction calls use the same input-budget
preflight; if even their fixed prompt cannot fit an unusually small custom
budget, MiniCoder records a deterministic recovery checkpoint without sending
that housekeeping request.

## Planning before execution

Planning is enabled by default. At the start of every user turn, the first model
request cannot call tools and may only return a concise numbered action plan. It
receives a bounded capability catalog containing current Registry names and
descriptions, while the execution requests receive the complete current JSON
Schemas through the API's separate `tools` field. Plan size follows task
complexity: a direct answer should use one item, read-only inspection two or
three, and code changes three to seven.
MiniCoder then adds an execution instruction and makes the tools available.
The system prompt tells the model to follow that plan unless file contents, tool
results, errors, or safety rules provide a concrete reason to adapt it. The plan
remains in normal conversation history and its model request counts toward
`MINICODER_MAX_STEPS`.

The console prints the bounded plan items and emits ordered “started” and
“completed” transitions while tools move through the plan. Progress is associated
with deterministic tool-activity facts: operation type, target path or query, and
verification command. Source edits, test edits, documentation updates, and
verification can therefore advance different plan items even when the provider
omits tool-call content.

Model-supplied step numbers are treated as hints and are accepted only when their
plan item is compatible with the actual tool action. Tool-using plan items must
run in order: if a call targets a later item before the current or next required
item has observable work, MiniCoder does not execute it and returns a correlated
`PLAN_STEP_OUT_OF_ORDER` result to the model. The model can then perform the
missing item and retry. If a reliable association is unavailable at finalization,
MiniCoder reports the item as not individually tracked; it never fabricates
instantaneous start/completion events.

Compatible models may attach `[plan_step=N]` to a tool-calling response for an
exact association. This is reserved host metadata: the application decodes it
immediately after the Model Port returns and removes it before conversation
history or final user-visible text is created. When a provider omits tool-call
content, the deterministic activity mapping remains available.

`replace_text` supports either one `old_text`/`new_text` pair or an atomic
`replacements` batch of up to 20 related exact edits to the same file. A batch is
validated completely in memory and written once; one missing, repeated, or
no-op match rejects the whole batch without partially changing the file. This
reduces model round trips for files with several independent edit locations.

Planning can be disabled for providers or small models that do not handle this
two-phase interaction well:

```bash
export MINICODER_PLANNING_ENABLED=false
```

## Exact sessions and cross-process short-term context

Exact session archiving is enabled by default. Disable it for a workspace whose
raw requests and responses must never be persisted:

```bash
export MINICODER_SESSION_ARCHIVE_ENABLED=false
```

Each MiniCoder process owns one append-only file at
`~/.minicoder/sessions/<workspace-hash>/<timestamp>-<session-id>.jsonl`. Records include the
exact external user task, every normalized model request, the current tool
schemas advertised with that request, every normalized model response, complete
model-visible ToolResults and host metadata, terminal success or failure, and
the post-turn maintenance decision. Records are flushed as work proceeds and
the directory/file modes are restricted to the current operating-system user
where supported.

The next process started with the same canonical workspace path always loads the
latest usable session, regardless of whether the new prompt says “continue”. It
injects the previous task, complete/failed/in-progress state, stop reason,
model-maintained rolling summary, and a bounded tail of visible messages and
tool exchanges. This recovered text is explicitly labeled as historical data,
not as a new instruction. A normal `/exit`, `/quit`, EOF or context-manager close
adds a session-close record; if the process was interrupted earlier, the next
startup builds a deterministic recovery checkpoint from the records already
flushed.

The archive is intentionally exact and may therefore contain source text,
commands, outputs, model reasoning exposed by a provider, or secrets that were
present in the conversation. It is separate from the sanitized `--trace` file.
There is currently no rotation because this design prioritizes complete local
recoverability; remove or disable archives explicitly for sensitive workspaces.

## Selective project long-term memory

Project memory is enabled by default. Disable it for a sensitive workspace or
when the extra summary-model request is not wanted:

```bash
export MINICODER_MEMORY_ENABLED=false
```

At the end of every successful or failed external turn, the same no-tool
maintenance request that updates rolling context must return either
`memory_action=none` or `memory_action=append`. It appends only a stable,
important, non-duplicate project fact that should survive beyond the latest
session; transient conversation and ordinary answers remain only in the exact
archive and rolling context. Model errors produce a deterministic recovery
summary but never create a guessed durable memory.

Selected records are redacted for the configured API key, stored outside the
project at `~/.minicoder/memory/<workspace-hash>.jsonl`, and supplied on every
task-model request within a bounded section. A memory read or write failure is a
warning and never changes the task result. Disabling `MINICODER_MEMORY_ENABLED`
prevents durable appends but does not disable exact session recovery or the
rolling-context maintenance needed by that feature.

The workspace path remains the project identity: moving the project starts new
session and memory locations. The current stores intentionally have no automatic
rotation, cross-process file locking, or `/forget` command.

## Terminal experience

On a real terminal, interactive input uses Python's editable `input()` path and
loads `readline` where the platform provides it. This gives committed Unicode
text character-aware Backspace handling and normal line history instead of
editing raw UTF-8 bytes. Injected streams keep a deterministic plain line reader
for tests and redirected input.

The default console shows user-facing Chinese progress and the exact tool name,
target path, search query, or bounded command needed to understand each action.
It never displays create/replace text bodies or internal call IDs, and common
API-key, token, password, authorization, credential, and secret command arguments
are redacted. Successful completion noise, message counts, and provider exception
class names remain omitted. Tool failures and terminal CLI failures use concise
Chinese explanations with actionable guidance. The optional JSONL trace retains
only the sanitized technical event fields for diagnosis.

Final model responses are rendered as Markdown with Rich, including headings,
lists, inline emphasis, and syntax-highlighted fenced code blocks. Valid fence
markers such as three backticks are interpreted rather than printed literally.

## Verification before completion

After a successful `create_file` or `replace_text`, MiniCoder accepts the final
response only after a relevant command succeeds. Built-in recognition covers
common Python checks, npm/pnpm/yarn scripts, Go, Cargo, .NET, Maven, Gradle,
Make, C and C++ compilers, CMake builds, CTest, and Ninja. For example, one
recognition rule covers any `g++` compilation containing a C or C++ source file;
individual source filenames do not need to be registered.

The model marks commands intended as evidence with
`purpose="verification"`. That declaration makes the intent explicit but does
not turn an arbitrary successful command such as `echo done` into proof. If a
toolchain is not built in, configure one stable whole-project verifier rather
than one command per source file:

```toml
# .minicoder.toml
[verification]
commands = [
  ["zig", "build", "test"],
  ["python", "scripts/verify_project.py"],
]
```

Each entry is an exact argv alternative, not shell text: pipes, redirection,
globs, and `&&` are not expanded. MiniCoder reads and validates this file once
at startup, then keeps an immutable snapshot for the session. This prevents an
in-session file edit from authorizing a new verifier. Review configuration
changes and restart MiniCoder to load them. `--check-config` reports only the
number of configured commands, not their contents.

When a successful command is marked for verification but is neither built in
nor present in the startup snapshot, MiniCoder gives the model one correction
opportunity and suggests recognized checks such as `python -m py_compile` or
`python -m pytest`. A direct application run such as `python app.py` remains a
general command because an application can hide a failed check and still exit
successfully. If the model proposes completion again without obtaining
recognized evidence for the same edit, the task ends with
`verification_unsupported` and configuration guidance instead of consuming the
remaining model-step budget.
