# Secretary Agent

[English](README.md)

个人秘书 agent，支持 CLI、Telegram、飞书/Lark 和 HTTP webhook。它可以帮你记录长期记忆、管理提醒、跟进未闭环事项，并按需启动后台研究。

## 主要功能

- 多渠道对话：CLI、Telegram、飞书/Lark、HTTP webhook。
- 长期记忆：记录稳定偏好、协作约定和需要持续跟踪的事项。
- 提醒与跟进：支持定时提醒、复盘提醒、未回复提醒检查和长期停滞事项检查。
- 后台研究：可用本地 Codex 或 Claude 启动多阶段深度研究；本机 CLI 不可用时可兜底到隔离的内部 agent，并按白名单使用搜索命令，在完成后通知你。
- DeepSeek API：命中率 99.9%。
- 技能扩展：支持内置技能和可选全局 skills。
- 多语言：`language` 控制 agent 回复，`ui_language` 控制命令和状态文案；二者都支持 `auto`、`zh`、`en`。

## 快速开始

```bash
python -m venv secretary_v2/venv
source secretary_v2/venv/bin/activate
pip install -r requirements.txt

cd secretary_v2
cp config.yaml.example config.yaml
# 编辑 config.yaml，填写模型、Telegram/飞书、HTTP token 等配置

python main.py --channel cli
```

单次 CLI 执行：

```bash
cd secretary_v2
python main.py --channel cli --send "你好"
```

后台运行：

```bash
cd secretary_v2
./manage.sh start
./manage.sh status
./manage.sh logs
```

`manage.sh` 会启动 `python main.py --channel all`，运行所有已配置的非 CLI 渠道：Telegram、飞书/Lark 和 HTTP。

### 存档并重建数据库（systemd 部署）

如需完整存档现有 SQLite 数据库、按当前结构从头建库，并且只保留
`status='open'` 的 `events` 与全部 `scheduled_tasks`，运行：

```bash
bash deploy/rebuild-database.sh --yes
```

脚本会停止 `secretary.service`，将带时间戳的完整旧库保存到
`secretary_v2/archive/`，校验存档和新库，然后启动服务并等待启动自检通过。
使用 `--no-start` 可让原本运行中的服务在完成后保持停止；运行环境不在
`~/.venvs/secretary` 时可设置 `SECRETARY_VENV` 或 `SECRETARY_PYTHON`。

## Telegram 命令

| 命令 | 说明 |
|------|------|
| `/status` | 查看系统状态 |
| `/skills` | 查看可用技能 |
| `/compact` | 手动压缩历史 |
| `/help` | 查看帮助 |

飞书/Lark 使用 `python main.py --channel feishu` 时支持同样的文本命令。使用 `python main.py --channel all` 可同时运行 Telegram、飞书/Lark 和 HTTP。

## 测试

```bash
cd secretary_v2
./venv/bin/python -m pytest tests -q
```

## 许可证

本项目采用 GNU General Public License v3.0 许可证。详见 [LICENSE](LICENSE)。
