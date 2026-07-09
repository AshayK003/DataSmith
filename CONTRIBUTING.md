# Contributing to DataSmith

Thanks for your interest in contributing. This document covers the practical details.

## Getting Started

```bash
git clone https://github.com/AshayK003/DataSmith.git
cd DataSmith
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -e ".[dev]"
```

Verify everything works:

```bash
pytest tests/ -v             # all tests pass
streamlit run app.py         # dashboard launches
```

## Development Workflow

1. Create a branch from `master`: `git checkout -b feature/my-change`
2. Make your changes
3. Add or update tests
4. Run `pytest tests/ -v` — all tests must pass
5. Commit and push
6. Open a pull request

## What to Work On

Check [open issues](https://github.com/AshayK003/DataSmith/issues) for planned work. Good first contributions:

- Add test coverage for `generation/`, `imperfections/`, `quality/` modules
- Fix test configuration (`testpaths` in `pyproject.toml`)
- Improve error messages in `core/database.py`
- Add new text profiles to `generation/text_profiles.py`

## Code Conventions

### Python

- Follow existing style in the file you're editing
- Use `logging` module for debug/info output
- Keep imports sorted: stdlib, third-party, local
- Type hints are encouraged but not required

### Architecture

- Core logic lives in `datasmith/`
- API layer is in `api.py`
- Streamlit UI is in `app.py`
- Use dependency injection for LLM providers

### Testing

- Test files go in `tests/`
- Name tests `test_<module>.py`
- Use `pytest` fixtures from `conftest.py`
- Write behavior-focused tests, not implementation tests

## Commit Messages

Use short imperative descriptions:

```
add test coverage for correlator module
fix text profile regex patterns
update API rate limiting
improve schema enrichment logic
```

## Pull Requests

- Keep PRs focused — one change per PR
- Include a description of what changed and why
- Reference related issues

## Questions?

Open an issue or start a discussion on GitHub.
