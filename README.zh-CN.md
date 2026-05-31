# Secretary Agent

[English](README.md)

个人秘书 agent，支持 CLI、Telegram 和 HTTP webhook。它可以帮你记录长期记忆、管理提醒、跟进未闭环事项，并按需启动后台研究。

## 主要功能

- 多渠道对话：CLI、Telegram、HTTP webhook。
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
# 编辑 config.yaml，填写模型、Telegram、HTTP token 等配置

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

## Telegram 命令

| 命令 | 说明 |
|------|------|
| `/status` | 查看系统状态 |
| `/skills` | 查看可用技能 |
| `/compact` | 手动压缩历史 |
| `/help` | 查看帮助 |

## 测试

```bash
cd secretary_v2
./venv/bin/python -m pytest tests -q
```

## 许可证

本项目采用 GNU General Public License v3.0 许可证。详见 [LICENSE](LICENSE)。
