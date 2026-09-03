"""Agent execution profiles and structured outputs for bounded v2 workflows.

Profiles are selected deterministically by the host application.  They are not
chosen by an LLM, so narrowing a profile never requires an extra classifier
request and cannot silently broaden tool access.
"""

from typing import FrozenSet, Literal, Optional

from pydantic import BaseModel, Field, model_validator


INTERACTIVE_PROFILE = "interactive"
WEBHOOK_PROFILE = "webhook_ingest"
SCHEDULED_NOTIFICATION_PROFILE = "scheduled_notification"
SCHEDULED_MAINTENANCE_PROFILE = "scheduled_maintenance"


# ``None`` means the existing full toolset.  An empty set means output tools
# only: Pydantic AI may expose its structured final-result tool, but none of
# Secretary's side-effecting/read tools are sent to the model.
PROFILE_TOOL_ALLOWLISTS: dict[str, Optional[FrozenSet[str]]] = {
    INTERACTIVE_PROFILE: None,
    WEBHOOK_PROFILE: frozenset(),
    SCHEDULED_NOTIFICATION_PROFILE: frozenset(),
    SCHEDULED_MAINTENANCE_PROFILE: frozenset(
        {
            "db_query",
            "db_execute",
            "memory_view",
            "memory_str_replace",
            "memory_insert",
        }
    ),
}


LIGHTWEIGHT_SCHEDULE_TASKS = frozenset(
    {
        "morning_briefing",
        "morning_trend_scan",
        "pending_response_check",
        "review_reminder",
        "stale_check",
    }
)

ALWAYS_NOTIFY_SCHEDULE_TASKS = frozenset(
    {"morning_trend_scan", "review_reminder"}
)


WEBHOOK_SYSTEM_PROMPT = """你是 Secretary 的 webhook 分析器。

只处理当前 webhook，并结合应用提供的长期记忆、事件索引和时间上下文给出简洁、可核查的回复。
不要把关注列表推断成持仓，不要编造实时价格、市场走势或未提供的事实。
本 profile 没有函数工具；需要保存的结果只能通过结构化输出字段交给应用执行。
`reply` 是发给用户的正文；`event_summary` 只概括原始 webhook；`events` 只包含原始记录之外、确实需要持续跟进的独立事项。
遵循应用提供的语言策略。"""


SCHEDULED_NOTIFICATION_SYSTEM_PROMPT = """你是 Secretary 的单次定时通知决策器。

应用已经预取完成判断所需的数据；本 profile 没有函数工具，也不需要调用 send_message。
严格依据任务说明和预取快照决定是否通知。不要编造实时数据，也不要发送“检查完成”“暂无事项”等无价值状态播报。
通知必须简洁、按优先级排列并使用纯文本。只有 pending_response_check 可以返回待解决的事件 ID，且只能选择快照中的候选 ID。
最终只返回规定的结构化结果，应用负责消息投递和经过校验的数据库更新。"""


def tool_allowlist_for_profile(profile: str) -> Optional[FrozenSet[str]]:
    """Return the profile allowlist; unknown profiles fail closed."""
    return PROFILE_TOOL_ALLOWLISTS.get(profile, frozenset())


def system_prompt_for_profile(profile: str, default: str) -> str:
    if profile == WEBHOOK_PROFILE:
        return WEBHOOK_SYSTEM_PROMPT
    if profile == SCHEDULED_NOTIFICATION_PROFILE:
        return SCHEDULED_NOTIFICATION_SYSTEM_PROMPT
    return default


class WebhookEventAction(BaseModel):
    """One distinct follow-up event proposed by a webhook analysis."""

    event_type: Literal["remind", "check", "response", "note", "triggered"]
    content: str = Field(min_length=1, max_length=4000)
    status: Literal["logged", "open"] = "open"
    summary: Optional[str] = Field(default=None, max_length=240)


class WebhookAgentOutput(BaseModel):
    """Single-pass webhook result consumed by the host application."""

    reply: str = Field(
        min_length=1,
        max_length=8000,
        description="User-visible reply. Use NO_ACTION only when silence was explicitly requested.",
    )
    event_summary: Optional[str] = Field(
        default=None,
        max_length=240,
        description="Concise factual index summary of the original webhook, not of the reply.",
    )
    events: list[WebhookEventAction] = Field(
        default_factory=list,
        max_length=3,
        description="Distinct actionable follow-ups not already represented by the webhook record.",
    )


class ScheduledNotificationOutput(BaseModel):
    """One-pass scheduled decision; delivery and mutations happen host-side."""

    should_notify: bool
    message: str = Field(
        default="",
        max_length=8000,
        description="Plain-text notification; empty exactly when should_notify is false.",
    )
    resolve_event_ids: list[int] = Field(
        default_factory=list,
        max_length=50,
        description="Only pending_response_check candidate IDs proven resolved or superseded.",
    )

    @model_validator(mode="after")
    def validate_notification_shape(self):
        self.message = self.message.strip()
        if self.should_notify and not self.message:
            raise ValueError("message is required when should_notify is true")
        if not self.should_notify and self.message:
            raise ValueError("message must be empty when should_notify is false")
        return self
