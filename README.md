# Secretary Agent

[简体中文](README.zh-CN.md)

A personal secretary agent for CLI, Telegram, and HTTP webhooks. It helps you keep long-term memory, manage reminders, follow up on open loops, and launch background research when needed.

## Key Features

- Multi-channel conversation: CLI, Telegram, and HTTP webhook.
- Long-term memory: stores stable preferences, collaboration agreements, and tracked items.
- Reminders and follow-ups: supports scheduled reminders, review reminders, pending-response checks, and stale-item checks.
- Background research: can launch multi-stage deep research with local Codex or Claude and notify you when it finishes.
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
# Edit config.yaml with your model, Telegram, and HTTP token settings.

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

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Show system status |
| `/skills` | List available skills |
| `/compact` | Compact conversation history |
| `/help` | Show help |

## Tests

```bash
cd secretary_v2
./venv/bin/python -m pytest tests -q
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
