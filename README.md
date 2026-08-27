# MiniCoder

MiniCoder is a command-line coding agent implemented from first principles for a software engineering assessment.

The project is currently being developed in small, test-backed increments. Do not place a real API key in this repository.

## Current increment

I01 provides the installable Python package, validated configuration, cross-platform detection, immutable domain values, and the application bootstrap factory.

```bash
export DEEPSEEK_API_KEY="your-key"
python -m minicoder --check-config --workspace .
```

The actual model adapter and agent loop are intentionally added in later increments so that every change remains reviewable and explainable.
