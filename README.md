# Secretary Agent

[简体中文](README.zh-CN.md)

A personal secretary agent for CLI, Telegram, Feishu/Lark, and HTTP webhooks. It helps you keep long-term memory, manage reminders, follow up on open loops, and launch background research when needed.

## Key Features

- Multi-channel conversation: CLI, Telegram, Feishu/Lark, and HTTP webhook.
- Long-term memory: stores stable preferences, collaboration agreements, and tracked items.
- Reminders and follow-ups: supports scheduled reminders, review reminders, pending-response checks, and stale-item checks.
- Background research: can launch multi-stage deep research with local Codex or Claude, with an isolated internal agent fallback for allowlisted search commands, and notify you when it finishes.
- DeepSeek API: 99.9% hit rate.
- Skill extension: supports built-in skills and optional global skills.
- Multilingual behavior: `language` controls agent replies, and `ui_language` controls command/status text. Both support `auto`, `zh`, and `en`.

## Quick Start

```bash
python -m venv secretary_v2/venv
source secretary_v2/venv/bin/activate
pip install -r requirements.txt

cd secretary_v2
cp config.yaml.example config.yaml
# Edit config.yaml with your model, Telegram/Feishu, and HTTP token settings.

python main.py --channel cli
```

Run one CLI message:

```bash
cd secretary_v2
python main.py --channel cli --send "Hello"
```

Run in the background:

```bash
cd secretary_v2
./manage.sh start
./manage.sh status
./manage.sh logs
```

`manage.sh` starts `python main.py --channel all`, which runs every configured non-CLI channel: Telegram, Feishu/Lark, and HTTP.

### Archive and rebuild the database (systemd deployment)

To archive the complete SQLite database, recreate the current schema, and keep
only `events` rows whose status is `open` plus every `scheduled_tasks` row:

```bash
bash deploy/rebuild-database.sh --yes
```

The script stops `secretary.service`, writes a timestamped database under
`secretary_v2/archive/`, validates the archive and rebuilt database, then starts
the service and waits for its startup self-test. Use `--no-start` to leave a
previously running service stopped. Set `SECRETARY_VENV` or `SECRETARY_PYTHON`
when the runtime is not in `~/.venvs/secretary`.

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Show system status |
| `/skills` | List available skills |
| `/compact` | Compact conversation history |
| `/help` | Show help |

Feishu/Lark supports the same text commands when using `python main.py --channel feishu`. Use `python main.py --channel all` to run Telegram, Feishu/Lark, and HTTP together.

## Tests

```bash
cd secretary_v2
./venv/bin/python -m pytest tests -q
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
