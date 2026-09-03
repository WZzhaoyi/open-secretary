"""Small UI string helper for user-facing channel text.

This intentionally does not touch agent prompts, schedule prompts, tool names,
schema names, or logs. Those stay as implementation/runtime text.
"""

from __future__ import annotations

from typing import Any, Optional

SUPPORTED_LANGS = {"en", "zh"}


MESSAGES = {
    "en": {
        "telegram.command.start": "Start",
        "telegram.command.help": "Show help",
        "telegram.command.status": "Show system status",
        "telegram.command.skills": "List available skills",
        "telegram.command.compact": "Compact conversation history",
        "telegram.start": (
            "👋 *Hi! I'm your personal secretary.*\n\n"
            "Tell me what you want to do in natural language.\n\n"
            "Use /help to see all commands."
        ),
        "telegram.help": (
            "🤖 *Personal Secretary Guide*\n\n"
            "*Features:*\n"
            "• 📝 Remember things - tell me your plans, ideas, and decisions\n"
            "• ⏰ Set reminders - tell me when and what to remind you about\n"
            "• 📊 Summarize information - ask me to organize and summarize information\n"
            "• ❓ Follow up - I can proactively ask about progress\n\n"
            "*Commands:*\n"
            "/start - Start\n"
            "/help - Show help\n"
            "/status - Show system status\n"
            "/skills - List available skills\n"
            "/compact - Compact this conversation's history\n\n"
            "Tell me what you want to do in natural language."
        ),
        "command.status.memory_missing": "not created",
        "command.status.read_failed": "read failed",
        "command.status.usage": "{input_tokens:,} / {window:,} tokens ({pct}), {at} from `{origin}`",
        "command.status.run_usage": (
            "{input_tokens:,} cumulative input tokens across {requests} requests, "
            "{at} from `{origin}`"
        ),
        "command.status.no_run": "no completed run yet",
        "command.status.no_cache_metrics": "last run did not return cache metrics",
        "command.status.cache_last_run": (
            "`{ratio}` (hit `{cache_hit:,}`, miss `{cache_miss:,}`, "
            "write `{cache_write:,}`)"
        ),
        "command.status.token_window_metrics": (
            "input `{input_tokens:,}`, output `{output_tokens:,}`, "
            "total `{total_tokens:,}` across `{runs}` runs"
        ),
        "command.status.no_token_window_metrics": "no samples",
        "command.status.cache_window_metrics": "`{ratio}` over `{runs}` runs",
        "command.status.no_cache_window_metrics": "no samples",
        "command.status.enabled": "enabled",
        "command.status.disabled": "disabled",
        "command.status.health_unknown": "unavailable",
        "command.status": (
            "📊 *System Status*\n\n"
            "🟢 Service is running\n"
            "🤖 Model: `{model}`\n"
            "⏰ Timezone: `{timezone}`\n"
            "📡 Channels: {channels}\n"
            "🩺 Channel health: {channel_health}\n"
            "💬 Messages (DB total / replayable in this conversation): {total_messages} / {history_count}\n"
            "🧮 Latest request context: {usage_status}\n"
            "🔥 Tokens (24h, all channels): {token_24h_status}\n"
            "🧊 Cache: last run {last_cache_status}; 24h all channels {cache_24h_status}\n"
            "🗜️ Compaction: threshold `{compact_threshold:,}` tokens, tail `{tail_budget:,}` tokens\n"
            "🧷 Auto compaction: `{auto_compact_status}`, minimum `{min_messages}` messages, "
            "cooldown `{cooldown_minutes}` minutes, tool output `{tool_output_chars}` chars\n"
            "🧠 memory.md: `{memory_status}`\n"
        ),
        "telegram.skills.empty": "📦 *No available skills*",
        "telegram.skills.title": "📦 *Available Skills* ({count})\n",
        "telegram.skills.no_triggers": "no triggers",
        "command.compact.running": "⏳ Compacting history...",
        "command.compact.failed": "❌ Compaction failed: {error}",
        "command.compact.completed": "✅ Compaction complete: {before} → {after} messages",
        "command.compact.not_needed": "ℹ️ Recent history is already within budget; no compaction needed",
        "command.compact.not_enough_history": "ℹ️ Not enough conversation history to compact",
        "command.compact.partial_failure": "❌ Compaction persistence failed: {error}",
        "telegram.message.error": "Sorry, an error occurred while processing your message. Please try again later.",
    },
    "zh": {
        "telegram.command.start": "开始使用",
        "telegram.command.help": "显示帮助",
        "telegram.command.status": "查看系统状态",
        "telegram.command.skills": "查看可用技能",
        "telegram.command.compact": "压缩对话历史",
        "telegram.start": (
            "👋 *你好！我是你的个人秘书。*\n\n"
            "直接用自然语言告诉我你想做什么就行！\n\n"
            "输入 /help 查看所有命令。"
        ),
        "telegram.help": (
            "🤖 *个人秘书使用指南*\n\n"
            "*功能：*\n"
            "• 📝 记住事情 - 告诉我你的计划、想法、决定\n"
            "• ⏰ 设置提醒 - 告诉我什么时候需要提醒你什么\n"
            "• 📊 汇总信息 - 让我帮你整理和总结信息\n"
            "• ❓ 追问跟进 - 我会主动问你事情的进展\n\n"
            "*命令：*\n"
            "/start - 开始使用\n"
            "/help - 显示帮助\n"
            "/status - 查看系统状态\n"
            "/skills - 查看可用技能\n"
            "/compact - 压缩当前会话历史\n\n"
            "直接用自然语言告诉我你想做什么就行！"
        ),
        "command.status.memory_missing": "未创建",
        "command.status.read_failed": "读取失败",
        "command.status.usage": "{input_tokens:,} / {window:,} tokens ({pct})，{at} 来自 `{origin}`",
        "command.status.run_usage": (
            "{input_tokens:,} 个累计输入 tokens（{requests} 次请求），"
            "{at} 来自 `{origin}`"
        ),
        "command.status.no_run": "尚无完成的 run",
        "command.status.no_cache_metrics": "上轮未返回缓存指标",
        "command.status.cache_last_run": (
            "`{ratio}`（命中 `{cache_hit:,}`，未命中 `{cache_miss:,}`，"
            "写入 `{cache_write:,}`）"
        ),
        "command.status.token_window_metrics": (
            "输入 `{input_tokens:,}`，输出 `{output_tokens:,}`，"
            "合计 `{total_tokens:,}`（`{runs}` 次运行）"
        ),
        "command.status.no_token_window_metrics": "暂无样本",
        "command.status.cache_window_metrics": "`{ratio}`（`{runs}` 次运行）",
        "command.status.no_cache_window_metrics": "暂无样本",
        "command.status.enabled": "开启",
        "command.status.disabled": "关闭",
        "command.status.health_unknown": "暂不可用",
        "command.status": (
            "📊 *系统状态*\n\n"
            "🟢 服务运行中\n"
            "🤖 模型：`{model}`\n"
            "⏰ 时区：`{timezone}`\n"
            "📡 Channels: {channels}\n"
            "🩺 Channel 健康：{channel_health}\n"
            "💬 历史消息（DB 总数 / 当前会话可重放）：{total_messages} / {history_count}\n"
            "🧮 最近一次请求上下文：{usage_status}\n"
            "🔥 Token 消耗（24h，全渠道）：{token_24h_status}\n"
            "🧊 缓存：上轮 {last_cache_status}；24h 全渠道 {cache_24h_status}\n"
            "🗜️ 压缩策略：阈值 `{compact_threshold:,}` tokens，tail `{tail_budget:,}` tokens\n"
            "🧷 自动压缩：`{auto_compact_status}`，最少 `{min_messages}` 条，"
            "冷却 `{cooldown_minutes}` 分钟，工具输出 `{tool_output_chars}` chars\n"
            "🧠 memory.md：`{memory_status}`\n"
        ),
        "telegram.skills.empty": "📦 *暂无可用技能*",
        "telegram.skills.title": "📦 *可用技能*（{count} 个）\n",
        "telegram.skills.no_triggers": "无触发词",
        "command.compact.running": "⏳ 正在压缩历史…",
        "command.compact.failed": "❌ 压缩失败：{error}",
        "command.compact.completed": "✅ 压缩完成：{before} → {after} 条消息",
        "command.compact.not_needed": "ℹ️ 最近历史已在预算内，无需压缩",
        "command.compact.not_enough_history": "ℹ️ 当前会话历史不足，无法压缩",
        "command.compact.partial_failure": "❌ 压缩快照保存失败：{error}",
        "telegram.message.error": "抱歉，处理消息时出错。请稍后再试。",
    },
}


def normalize_lang(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("en"):
        return "en"
    return None


def resolve_ui_language(
    ui_language: str = "auto",
    channel_language: Optional[str] = None,
    agent_language: str = "auto",
) -> str:
    explicit = normalize_lang(ui_language)
    if explicit:
        return explicit
    channel = normalize_lang(channel_language)
    if channel:
        return channel
    agent = normalize_lang(agent_language)
    if agent:
        return agent
    return "en"


def t(key: str, lang: str = "en", **kwargs: Any) -> str:
    resolved = normalize_lang(lang) or "en"
    template = MESSAGES.get(resolved, MESSAGES["en"]).get(key, MESSAGES["en"].get(key, key))
    return template.format(**kwargs)
