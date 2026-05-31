"""Secretary v2 configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import yaml


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    effort: str = ""
    thinking: str = ""
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class HTTPConfig:
    enabled: bool = True
    token: str = ""


@dataclass
class ChannelConfig:
    enabled: bool = True
    default_outgoing: str = "cli"
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)


@dataclass
class DatabaseConfig:
    path: str = "secretary_v2.db"


@dataclass
class SkillsConfig:
    max_size: int = 50000
    max_loaded: int = 5
    auto_load: List[str] = field(default_factory=list)
    include_global: bool = True
    paths: List[str] = field(default_factory=list)
    triggers: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SearchConfig:
    backend: str = "tavily"
    tavily_api_key: str = ""
    searxng_url: str = "http://localhost:8080"


@dataclass
class HistoryConfig:
    context_tokens: int = 100000
    compress_threshold: float = 0.75
    tail_token_budget: int = 20000
    max_events: int = 10
    auto_persist_compact: bool = True
    persist_compact_threshold: float = 0.85
    persist_compact_min_active_messages: int = 80
    persist_compact_cooldown_minutes: int = 120
    compact_tool_output_max_chars: int = 4000


@dataclass
class CodexSubagentConfig:
    model: str = ""
    enable_search: bool = True
    sandbox: str = "read-only"
    approval_policy: str = "never"
    config_overrides: List[str] = field(default_factory=list)


@dataclass
class ClaudeSubagentConfig:
    model: str = "sonnet"
    effort: str = "high"
    allowed_tools: List[str] = field(
        default_factory=lambda: ["WebSearch", "WebFetch", "Read"]
    )
    allowed_bash: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)


@dataclass
class AgentSubagentConfig:
    enabled: bool = True
    model: str = ""
    allowed_tools: List[str] = field(
        default_factory=lambda: [
            "Bash(opencli list*)",
            "Bash(opencli * -h)",
            "Bash(opencli * --help)",
            "Bash(opencli * * -h)",
            "Bash(opencli * * --help)",
            "Bash(opencli grok *)",
            "Bash(opencli doubao *)",
            "Bash(opencli gemini *)",
        ]
    )
    disallowed_tools: List[str] = field(default_factory=list)
    shell_timeout: int = 60
    system_prompt: str = (
        "You are a compact, isolated research subagent. Complete the assigned "
        "stage using only the prompt content and explicitly allowed tools. Do "
        "not claim access to local files, memory, reminders, message channels, "
        "or the main secretary agent state. Return only the stage output."
    )


@dataclass
class SubagentConfig:
    default_engine: str = "claude"
    fallback_engine: str = "agent"
    codex: CodexSubagentConfig = field(default_factory=CodexSubagentConfig)
    claude: ClaudeSubagentConfig = field(default_factory=ClaudeSubagentConfig)
    agent: AgentSubagentConfig = field(default_factory=AgentSubagentConfig)


@dataclass
class ScheduleConfig:
    cron: str
    enabled: bool = True
    prompt: str = ""
    protected: bool = False


@dataclass
class Config:
    llm: LLMConfig
    channels: ChannelConfig
    database: DatabaseConfig
    skills: SkillsConfig
    search: SearchConfig
    history: HistoryConfig
    subagent: SubagentConfig
    schedules: Dict[str, ScheduleConfig]
    timezone: str = "Asia/Shanghai"
    language: str = "auto"
    ui_language: str = "auto"
    system_prompt: str = ""

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                "SECRETARY_CONFIG",
                str(Path(__file__).parent / "config.yaml")
            )

        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        # Parse LLM config
        llm_data = config_data.get("llm", {})
        llm = LLMConfig(**llm_data)

        # Parse channel config
        channels_data = config_data.get("channels", {})
        telegram_data = channels_data.get("telegram", {})
        http_data = channels_data.get("http", {})
        channels = ChannelConfig(
            enabled=channels_data.get("enabled", True),
            default_outgoing=channels_data.get("default_outgoing", "cli"),
            telegram=TelegramConfig(**telegram_data),
            http=HTTPConfig(**http_data),
        )

        # Parse database config
        database_data = config_data.get("database", {})
        database = DatabaseConfig(**database_data)

        # Parse skills config
        skills_data = config_data.get("skills", {})
        skills = SkillsConfig(**skills_data)

        # Parse search config
        search_data = config_data.get("search", {})
        search = SearchConfig(**search_data)

        # Parse history config
        history_data = config_data.get("history", {})
        history = HistoryConfig(**history_data)

        # Parse subagent sidecar config
        subagent_data = config_data.get("subagent", {})
        subagent = SubagentConfig(
            default_engine=subagent_data.get("default_engine", "claude"),
            fallback_engine=subagent_data.get("fallback_engine", "agent"),
            codex=CodexSubagentConfig(**subagent_data.get("codex", {})),
            claude=ClaudeSubagentConfig(**subagent_data.get("claude", {})),
            agent=AgentSubagentConfig(**subagent_data.get("agent", {})),
        )

        # Parse schedules
        schedules_data = config_data.get("schedules", {})
        schedules = {}
        for task_id, task_data in schedules_data.items():
            schedules[task_id] = ScheduleConfig(**task_data)

        return cls(
            llm=llm,
            channels=channels,
            database=database,
            skills=skills,
            search=search,
            history=history,
            subagent=subagent,
            schedules=schedules,
            timezone=config_data.get("timezone", "Asia/Shanghai"),
            language=config_data.get("language", "auto"),
            ui_language=config_data.get("ui_language", "auto"),
            system_prompt=config_data.get("system_prompt", SECRETARY_PERSONA),
        )


# Secretary persona - static system prompt
SECRETARY_PERSONA = """你是一个个人秘书 agent，通过对话帮助用户管理信息、任务和时间。

## 行为准则

- 持有用户的注意力，不替用户做决定
- 在用户情绪波动时，提醒用户自己的原始判断
- 主动关联相关信息，发现模式
- 保持简洁，避免冗余解释
- 遵循 dynamic_context 中的语言策略回复
- 时间约定见 dynamic_context 中的"时间约定"段（配置驱动）
- 核心秘书工作流、长期记忆和 events 规则由自动加载的 `secretary-core` skill 提供。

## 定时任务输出契约（强制）

**前提**：定时任务触发时，用户**看不到**你的 final output。要把内容送达用户，唯一方式是调 `send_message` 工具。final output 只用于内部 NO_ACTION 检测和日志，对用户不可见。

因此你只有两种合法路径：

**路径 A — 有内容要发给用户**
1. 调 `send_message(text="...")` 把内容送出去
2. final output 写 `NO_ACTION`（工具已经送达，final output 没用了，写 NO_ACTION 避免污染历史）

**路径 B — 没内容要发**
1. 不调 `send_message`
2. final output 写 `NO_ACTION`

**正例 1（pending_response_check 命中未回复）**：
```
[调 send_message(text="今早提醒的 NBIS 财报你还没回，要继续持有吗？")]
final output: NO_ACTION
```

**正例 2（pending_response_check 没未回复）**：
```
[不调任何工具]
final output: NO_ACTION
```

**正例 3（review_reminder 总是要发）**：
```
[调 send_message(text="📋 今日复盘：...")]
final output: NO_ACTION
```

**反例（错误，不要这样做）**：final output 写"已发送。"、"所有提醒已回复"、"今日无重要事项"等任何自然语言 —— 都不会送达用户，只会被当作合规失败记录。

任务类别：
- pending_response_check / stale_check / morning_briefing / system_review：命中条件才走路径 A，否则路径 B
- review_reminder / morning_trend_scan：始终走路径 A
- memory_consolidation：始终走路径 B（静默维护 memory.md 与 events，不通知用户）

重要：不要把"无需跟进"、"检查完成"等元记录写入数据库。只记录有价值的交互和决策。

## 回复记录规则
当用户回复了某个提醒或追问时，记录用户的回复：
- 只在用户明确回复了某个事项时才记录，不要把所有消息都记录为回复
"""

# Database schema hint for LLM
DB_SCHEMA_HINT = """## 数据库表结构

### events 表（事件记录 / 操作台账）
- id: INTEGER PRIMARY KEY AUTOINCREMENT
- type: TEXT NOT NULL (remind | check | response | note | triggered)
- status: TEXT NOT NULL (logged | open | resolved | promoted)，默认 logged
- content: TEXT
- created_at: DATETIME

注：长期记忆（偏好、计划、在追踪的事项）不在数据库里，而在 memory.md。
events 表只记有时点的事件流水，供定时任务做未回复检查等。
`status='open'` 是主动注意力清单；最近事件注入只是截断视图，不代表事件全集。

### messages 表（对话历史）
- id: INTEGER PRIMARY KEY AUTOINCREMENT
- source: TEXT NOT NULL (user/assistant/system/summary)
- content: TEXT
- tokens_in: INTEGER DEFAULT 0
- tokens_out: INTEGER DEFAULT 0
- created_at: DATETIME DEFAULT CURRENT_TIMESTAMP
- pydantic_ai_msg: BLOB (序列化的Pydantic AI消息)

### agent_events 表（agent 行为审计 / 系统复盘）
- id: INTEGER PRIMARY KEY AUTOINCREMENT
- run_id: TEXT (一次 agent run 或 subagent job 的关联 ID)
- origin: TEXT NOT NULL (user | scheduled | research | approval | 具体 channel)
- type: TEXT NOT NULL (run_started | run_finished | run_failed | scheduled_no_action | send_message | memory_update | subagent_step_started | subagent_step_finished 等)
- subject: TEXT
- payload_json: TEXT (JSON 字符串)
- created_at: DATETIME

注：agent_events 不是长期记忆，不要注入 memory.md。它只用于系统行为审计、
排查静默失败、统计提醒送达/响应、复盘工具调用和 subagent 阶段耗时。

### subagent_runs 表（后台 subagent 任务）
- id: TEXT PRIMARY KEY (例如 research_xxx)
- agent_name: TEXT NOT NULL (例如 deep_research)
- agent_kind: TEXT NOT NULL (例如 research)
- engine: TEXT NOT NULL (claude | codex)
- input_json: TEXT NOT NULL (JSON 字符串，例如 {"topic": "..."})
- subject: TEXT
- status: TEXT NOT NULL (pending | running | succeeded | failed | cancelled)
- origin_channel: TEXT NOT NULL
- user_id: TEXT
- stages_json: TEXT (阶段状态 JSON)
- artifact_path: TEXT
- result: TEXT
- error: TEXT
- created_at / updated_at / completed_at: DATETIME

注：deep research 使用 subagent_runs；旧 research_jobs 不再属于运行时 schema。

### scheduled_tasks 表（定时任务）
- id: TEXT PRIMARY KEY
- cron: TEXT (cron表达式)
- prompt: TEXT (触发提示词)
- enabled: INTEGER (是否启用)
- protected: INTEGER (是否受保护)
- last_run: TEXT (上次运行时间)
- created_at: DATETIME

## 工具使用指南

1. **load_skill** - 按名称读取可用技能索引中的完整 skill 说明
2. **db_query** - 读数据库（SELECT/PRAGMA）
3. **db_execute** - 写 events 等普通业务流水；不要修改系统表、调度表、消息表或子任务表
4. **memory_read** - 查看完整 memory.md
5. **memory_update** - 更新长期记忆（优先用于 memory.md）
6. **file_read** - 读普通项目文件（日志、技能、权限策略、研究产物等）；不能读 config.yaml、数据库或凭证文件
7. **file_write** - 写普通 data 文件；agent 不能用它写 memory.md、logs、permissions、research/subagent_runs、代码、配置或数据库文件
8. **http_request** - 调外部 API（知道具体 URL 时用）
9. **web_search** - 搜索互联网（不知道去哪里找信息时用）
10. **send_message** - 主动发消息给用户（通知、提醒）
11. **schedule_task** - 管理定时任务
12. **start_research / get_research_status / cancel_research** - 启动、查询、取消后台深度研究

### 工具选择
- 看到「可用技能索引」里有相关技能 → 先用 load_skill(name) 读取完整说明
- 知道具体 URL/API → http_request
- 不知道去哪里找信息 → web_search
- 交易机会、行业分析、需要多轮搜索/反证/报告的主题 → start_research，后台完成后再通知用户
- 用户询问 `research_xxx` 的状态/进度或最近研究任务 → get_research_status，绝不要因此启动新研究

### 定时任务
- 定时任务触发时，prompt 会送入 agent 循环
- 受保护任务（如 pending_response_check）不可删除
- 创建任务示例：schedule_task(action="create", task_id="xxx", cron="0 9 * * *", prompt="提醒用户...")"""


def get_config() -> Config:
    """Get configuration singleton."""
    if not hasattr(get_config, "_config"):
        get_config._config = Config.load()
    return get_config._config


def reset_config():
    """Reset configuration (for testing)."""
    if hasattr(get_config, "_config"):
        del get_config._config
