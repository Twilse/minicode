# MiniCoder

MiniCoder is a command-line coding agent implemented from first principles for a software engineering assessment.

The project is currently being developed in small, test-backed increments. Do not place a real API key in this repository.

## Current capabilities

MiniCoder provides a synchronous multi-turn model/tool loop, workspace-scoped
file tools, cross-platform command execution, bounded diagnostic output,
context compaction, verification-before-completion, explicit states, and
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
tool argument/output bodies, final response bodies, and model reasoning content.

Each new interactive user turn receives a fresh model-step allowance while prior
messages and completion evidence remain available to the session.

## Terminal experience

On a real terminal, interactive input uses Python's editable `input()` path and
loads `readline` where the platform provides it. This gives committed Unicode
text character-aware Backspace handling and normal line history instead of
editing raw UTF-8 bytes. Injected streams keep a deterministic plain line reader
for tests and redirected input.

The default console shows user-facing Chinese progress such as “正在读取文件” and
“C/C++ 编译检查已通过”. Successful tool-completion noise, internal tool names,
call IDs, message counts, and provider error classes are omitted. The optional
JSONL trace retains the sanitized technical event fields for diagnosis.

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

When a command explicitly marked for verification succeeds but is neither
built in nor present in the startup snapshot, the task ends once with
`verification_unsupported` and configuration guidance. It does not repeatedly
reject the same final answer until the model-step limit is exhausted.
