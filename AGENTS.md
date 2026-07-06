# AGENTS.md

## Project Context

- This repository contains Secretary Agent, a personal secretary agent for CLI, Telegram, Feishu/Lark, and HTTP webhooks.
- Read `README.md` first for setup, run commands, and test commands.
- Use `README.zh-CN.md` when Chinese-language project wording is needed.
- Treat `plan/*.md` as planning and design context when the task touches roadmap, architecture, or behavior decisions.

## Development

- Keep changes scoped to the requested behavior and consistent with the existing Python code style.
- Prefer existing project patterns and helpers over introducing new abstractions.
- Do not commit secrets, tokens, local config files, runtime logs, virtual environments, or generated caches.
- Do not run `manage.sh` directly from Codex sandboxed commands because it can start the service incorrectly. Ask the user to start, stop, or restart the service manually when service control is needed.
- For Python changes, run the relevant tests when feasible:

  ```bash
  cd secretary_v2
  ./venv/bin/python -m pytest tests -q
  ```

## Commits

- Use Conventional Commits for commit messages, such as `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, or `chore:`.
