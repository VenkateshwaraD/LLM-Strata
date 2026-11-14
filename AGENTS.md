# Coding Standards

These rules apply to all Python code in this repository.

## Secrets and configuration

- All secrets come from environment variables or `.env`. Never hardcode them. Never commit `.env`.
- Validate configuration at startup with `os.environ["KEY"]` (fails loud) — not deep in call stacks.
- Never log secrets, tokens, or credentials. Sanitize before logging.

## Validation

- Do not use `assert` for runtime validation — it is stripped with `-O`. Use `if` / `raise ValueError`.

## Testing

- Aim for ≥80% meaningful coverage. Do not write tests just to hit a number.
- Test error paths, edge cases, and empty inputs — not just the happy path.
- Mock at boundaries only, never internal implementation.

## Function shape and naming

- Functions: ≤20 lines, return early, keyword-only arguments for 3+ parameters.
- Naming: no single-letter variable names — use descriptive names that convey
  intent (e.g., `result` not `r`, `directory` not `d`).

## Error handling

- Catch specific exceptions only. No bare `except:`. No `except Exception: pass`.
- Chain with `raise ... from e` when wrapping exceptions.
- Handle errors at one layer, not multiple.

## Logging

- `logging` module only for diagnostics — lazy formatting:
  `logger.info("user %s", uid)` — never f-strings in log calls.
- **CLI carve-out:** `print()` is permitted for user-facing output in command-line
  runners (headers, progress, results, interactive prompts). All diagnostics,
  warnings, and errors go through `logger`, never `print()`.

## Typing

- No legacy imports from `typing` for builtins (`List`, `Dict`, `Optional`,
  `Tuple`). Use builtin generics (`list`, `dict`, `X | None`).
- Use `TypeIs` over `TypeGuard`. Use `TypeVar` defaults where appropriate.
- `# type: ignore` must include an error code: `# type: ignore[attr-defined]`.

## Docstrings

- Google-style docstrings on public functions. Do not restate type hints in
  docstrings.

## Project layout

```text
core/                 Shared package: env_loader, profile_loader, client_factory
layers/scan/          Pre-deployment assessment runners (garak, deepeval, ragas, pyrit, llm_guard)
layers/attack/        Adversarial simulation runners (encoded_attacks, in_context_attacks,
                      multilingual, tool_inject, backdoor, rag_security,
                      agent_security, human_redteam)
layers/monitor/       Production monitoring (langfuse)
tests/                Top-level unittest suite (run: python -m unittest discover -s tests -v)
docs/                 Threat model, architecture, CI/CD, triage, integration guides
results/              Generated reports (never commit secrets in reports)
```

## Commands

```bash
python run_security.py --dry-run                        # orchestration check (no network)
python -m unittest discover -s tests -v                 # unit tests
coverage run -m unittest discover -s tests && coverage report   # coverage
python -m compileall -q .                               # syntax check
```
