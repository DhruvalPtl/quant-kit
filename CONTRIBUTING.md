# Contributing to quant-kit

Thank you for your interest in contributing to quant-kit! This tool aims to be the standard way the community builds and evaluates GGUF models.

## Development Setup

1. Fork and clone the repository.
2. Install the package in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Ensure you have the `llama.cpp` binaries in a `llama.cpp/` folder at the root of the project, as the scripts depend on them.
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

This project uses `ruff` for fast linting and formatting.
Before submitting a PR, run:
```bash
ruff check .
```

## Adding Features

If you want to add a new quantization type, quality benchmark, or major feature:
1. Open an Issue first to discuss the feature.
2. Ensure you add any new dependencies to `pyproject.toml`.
3. Update `README.md` if necessary.

## Pull Request Process

1. Create a branch (`feature/your-feature-name` or `fix/issue-number`).
2. Make your changes.
3. Test thoroughly (especially on Windows and Linux, if possible).
4. Submit the PR with a clear description of the problem solved.
