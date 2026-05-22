# Backend Repository Instructions

## Core Principles

- Write production-ready, readable, maintainable, and fully typed code.
- Follow existing repository structure, naming conventions, and patterns.
- Prefer explicit code over unnecessary abstractions.
- Keep functions small and single-responsibility.
- Use async-first patterns for all DB/network/IO operations.

---

## Project Structure

```text
api/routes   -> Controllers / request handling
api/services -> Business logic
api/utils    -> Shared utilities/helpers
api/schemas  -> Pydantic request/response models
```

### Rules

- Routes should only handle validation, request parsing, service calls, and responses.
- Business logic must stay inside services.
- Shared/reusable logic belongs in utils.
- Always use `api.utils.response` for outer API response structure.

---

## API Standards

- Always use Pydantic models for request and response validation.
- Use proper HTTP status codes.
- Add meaningful error messages.
- Never return raw dictionaries directly from routes.

Example:

```python
return success_response(
    message="Success",
    data=response_data
)
```

---

## Error Handling

- Use proper try/except blocks in critical flows.
- Never silently swallow exceptions.
- Log unexpected exceptions with stack traces.
- Raise proper `HTTPException` where appropriate.

---

## Database Standards

- Use async/await for all DB operations.
- Use typed SQLAlchemy models (`Mapped[]`).
- Use proper constraints, indexes, and relationships.
- Prefer UUID primary keys unless repository conventions differ.

---

## Configuration

- Never hardcode secrets, URLs, or environment-specific values.
- Always load configs from `config.settings`.

Example:

```python
from config.settings import settings
```

---

## Logging

- Use structured logging consistently.
- Log start/end of important operations.
- Log execution time for scripts/jobs.
- Log failures with stack traces.

Example:

```python
start_time = time.time()

logger.info("Job started")

logger.info(
    "Job completed in %.2f seconds",
    time.time() - start_time
)
```

---

## Script Standards

- Use constants instead of CLI args unless explicitly requested.
- Add dynamic parent `sys.path` imports.
- Add proper logging and execution timing.
- Scripts should support direct execution.
- Always run scripts inside `.venv`.

Example:

```python
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
```

Run scripts using:

```bash
source .venv/bin/activate
python path/to/script.py
```

---

## Code Quality

- Add explicit typing everywhere.
- Avoid `Any` unless absolutely necessary.
- Use descriptive variable/function names.
- Avoid duplicate logic and unnecessary abstractions.
- Prefer reusable utility functions.
- Add concise comments/docstrings where useful.

---

## Import Order

1. Standard library
2. Third-party libraries
3. Internal imports

---

## Performance & Reliability

- Avoid N+1 queries.
- Use pagination for list endpoints.
- Prefer bulk operations where appropriate.
- Avoid blocking operations inside async code.

---

## Validation Before Completion

After every file modification:

- Check syntax errors
- Check imports and typing
- Run linting
- Verify Pylance diagnostics (`PylanceFileSyntaxErrors`)
- Ensure code compiles cleanly before completion

---

## Security

- Never log secrets/tokens.
- Validate all external inputs.
- Use parameterized queries/ORM methods.
- Avoid exposing internal exceptions to clients.
