# MiniCoder

MiniCoder is a command-line coding agent implemented from first principles for a software engineering assessment.

The project is currently being developed in small, test-backed increments. Do not place a real API key in this repository.

## Current increment

I07 provides a synchronous model/tool Agent loop, workspace-scoped file tools,
cross-platform command execution, bounded diagnostic output, explicit states, and
sanitized Observer events for the console and optional JSONL traces.

Configure any OpenAI-compatible Chat Completions endpoint with tool calling:

```bash
export MINICODER_API_KEY="your-key"
export MINICODER_BASE_URL="https://api.deepseek.com"
export MINICODER_MODEL="deepseek-v4-pro"
python -m minicoder --check-config --workspace .
```

Run one task:

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

The project remains under incremental development. Context compaction,
verification-before-completion, and final demonstration assets are added in later
increments.
