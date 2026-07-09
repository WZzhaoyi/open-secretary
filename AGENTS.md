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
- For Python changes, run the relevant tests when feasible: `python -m pytest tests -q` from `secretary_v2`.
- Do not assume where the Python environment lives. The runtime venv is decoupled from the source tree and varies by host (a repo-local `secretary_v2/venv` on some dev machines, an external uv-managed venv such as `~/.venvs/secretary` on deployed hosts, or something else entirely). If the session has not already established which interpreter to use, ask the user instead of guessing a path.

## Operations (service control, logs)

- Before touching the service, identify how this host runs it. Check `systemctl status secretary` first: if the unit exists, the host is a systemd deployment (VPS, set up by `deploy/deploy.sh`); if not, the service is managed by `secretary_v2/manage.sh` (development machines).
- Never blindly restart. First read the tail of `secretary_v2/logs/secretary_v2.log` and understand why the service is down or unhealthy. A failed startup self-test, an invalid `config.yaml`, or an OOM kill will fail again on restart and only consume systemd's restart budget.
- On systemd hosts, use `systemctl` only and do not mix in `manage.sh`: `manage.sh stop` kills the process but systemd immediately restarts it, and `manage.sh start` can race the unit. The unit allows at most 5 failed starts per 10 minutes, then enters the `failed` state; recover with `sudo systemctl reset-failed secretary && sudo systemctl start secretary` only after fixing the root cause.
- Logs: the live file is `secretary_v2/logs/secretary_v2.log`. On systemd hosts it is rotated in place by logrotate (copytruncate) and old rotations are swept by the `secretary-logclean.timer`; do not rotate or delete it manually. With `manage.sh` the file rotates only at start and grows unbounded while the process runs — treat a huge log file as expected there, not as a bug to fix by restarting.
- The HTTP webhook binds `127.0.0.1:11269` by design; external access goes through Cloudflare Tunnel. Do not "fix" unreachability by exposing the port.

## Commits

- Use Conventional Commits for commit messages, such as `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, or `chore:`.
