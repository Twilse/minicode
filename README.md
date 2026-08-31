# MiniCoder

MiniCoder is a command-line coding agent implemented from first principles for a software engineering assessment.

The project is currently being developed in small, test-backed increments. Do not place a real API key in this repository.

## Current capabilities

MiniCoder provides a synchronous multi-turn model/tool loop, workspace-scoped
file tools, cross-platform command execution, bounded diagnostic output,
context compaction, an explicit plan-before-execution phase,
verification-before-completion, project memory, explicit states, and
sanitized Observer events for the console and optional JSONL traces.

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

## Context budget

MiniCoder uses an approximate character budget to keep conversation history
below the configured model's context window. The default is 180,000 characters,
three times the previous default, and older tool-heavy history is compacted only
after that threshold is reached. This value is deliberately provider-neutral:
it is not a token count or a claim about every compatible model's maximum.

Override it when the selected endpoint has a smaller or larger context window:

```bash
export MINICODER_CONTEXT_BUDGET_CHARS=300000
```

## Planning before execution

Planning is enabled by default. At the start of every user turn, the first model
request receives no tool definitions and may only return a concise numbered
action plan. Plan size follows task complexity: a direct answer should use one
item, read-only inspection two or three, and code changes three to seven.
MiniCoder then adds an execution instruction and makes the tools available.
The system prompt tells the model to follow that plan unless file contents, tool
results, errors, or safety rules provide a concrete reason to adapt it. The plan
remains in normal conversation history and its model request counts toward
`MINICODER_MAX_STEPS`.

The console prints the bounded plan items and emits ordered “started” and
“completed” transitions while tools move through the plan. Even if a model
directly associates a tool with a later item, MiniCoder closes and displays each
intermediate item in sequence instead of jumping over its number. The terminal
whole-plan event closes any remaining items without inventing individual tool
work for them.

Compatible models may attach `[plan_step=N]` to a tool-calling response for an
exact association. This is reserved host metadata: the application decodes it
immediately after the Model Port returns and removes it before conversation
history or final user-visible text is created. When a provider omits tool-call
content, MiniCoder falls back to a deterministic mapping from read/search,
create/replace, and command tools to inspection, implementation, and
verification plan items.

Planning can be disabled for providers or small models that do not handle this
two-phase interaction well:

```bash
export MINICODER_PLANNING_ENABLED=false
```

## Project memory

Project memory is enabled by default. Disable it for a sensitive workspace or
when the extra summary-model request is not wanted:

```bash
export MINICODER_MEMORY_ENABLED=false
```

After each successfully completed user turn, MiniCoder makes one additional
no-tool model request to create a bounded semantic summary. A later process
started with the same canonical workspace path loads up to eight recent
summaries and supplies them as historical data on the first turn. This memory is
separate from the optional audit trace and is stored outside the project at
`~/.minicoder/memory/<workspace-hash>.jsonl`.

Only the original user request and final answer are sent to the configured model
for summarization; tool output, model reasoning, and the full conversation are
not included. The configured API key is redacted if it appears in persisted
text. A summary-model failure uses a deterministic bounded fallback, while a
memory read or write failure is reported as a warning and never changes a
successful task into a failed one. That fallback contains bounded excerpts from
the request and answer rather than a model-written abstraction. Failed tasks are
not remembered.

Memory therefore adds one model request per successful turn and stores
derived project information locally. Other secrets contained in a request or
answer cannot be identified with certainty, so review this tradeoff before
using the default. The workspace path is the project identity: moving the project
starts a different memory file. The current lightweight store intentionally has
no automatic rotation, cross-process file locking, or `/forget` command.

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
