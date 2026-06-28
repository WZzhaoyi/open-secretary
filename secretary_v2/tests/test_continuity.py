"""Regression tests for the P0 fixes:

1. message_history continuity (load + persist pydantic-ai messages)
2. shell tool registered with proper guardrails
3. pre-run compaction happens before synthetic context assembly
4. Scheduler resyncs runtime-created tasks from DB on restart

These tests use Pydantic AI's TestModel so they don't burn API tokens.
"""

import asyncio

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)

import runtime
from config import (
    ChannelConfig,
    Config,
    DatabaseConfig,
    HistoryConfig,
    LLMConfig,
    SubagentConfig,
    SearchConfig,
    SkillsConfig,
)
from memory import Database
from runtime import run_agent
from scheduler import Scheduler
from pydantic_ai_summarization import SummarizationProcessor
from compaction import estimate_tokens, force_compact


class FakeUsage:
    input_tokens = 11
    output_tokens = 7
    requests = 1
    cache_read_tokens = 0
    cache_write_tokens = 0
    details = {}


class FakeRunResult:
    def __init__(self, user_text: str, output: str = "ok", usage_error: Exception | None = None):
        self.output = output
        self._user_text = user_text
        self._usage_error = usage_error

    def new_messages(self):
        return [
            ModelRequest(parts=[UserPromptPart(content=self._user_text)]),
            ModelResponse(parts=[TextPart(content=self.output)]),
        ]

    def usage(self):
        if self._usage_error:
            raise self._usage_error
        return FakeUsage()


class FakeThinkingOnlyRunResult(FakeRunResult):
    def __init__(self, user_text: str, thinking: str, output: str = "stale output"):
        super().__init__(user_text, output)
        self._thinking = thinking

    def new_messages(self):
        return [
            ModelRequest(parts=[UserPromptPart(content=self._user_text)]),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=self._thinking,
                        id="reasoning_content",
                        provider_name="deepseek",
                    )
                ],
                provider_name="deepseek",
            ),
        ]


def _config_for_model(provider: str, model: str, api_key: str = "test-key") -> Config:
    return Config(
        llm=LLMConfig(provider=provider, model=model, api_key=api_key),
        channels=ChannelConfig(),
        database=DatabaseConfig(),
        skills=SkillsConfig(),
        search=SearchConfig(),
        history=HistoryConfig(),
        subagent=SubagentConfig(),
        schedules={},
        timezone="Asia/Shanghai",
        language="auto",
    )


def test_config_loads_language_policy(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("language: en\nui_language: zh\ntimezone: UTC\n", encoding="utf-8")

    cfg = Config.load(str(config_file))

    assert cfg.language == "en"
    assert cfg.ui_language == "zh"
    assert cfg.timezone == "UTC"


def test_config_defaults_language_to_auto(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("timezone: UTC\n", encoding="utf-8")

    cfg = Config.load(str(config_file))

    assert cfg.language == "auto"
    assert cfg.ui_language == "auto"


def test_ui_i18n_resolves_auto_from_channel_language():
    from channels.telegram_channel import bot_commands
    from i18n import resolve_ui_language, t

    assert resolve_ui_language("auto", channel_language="zh-CN", agent_language="en") == "zh"
    assert resolve_ui_language("auto", channel_language=None, agent_language="zh") == "zh"
    assert resolve_ui_language("auto", channel_language=None, agent_language="auto") == "en"
    assert t("telegram.command.status", "en") == "Show system status"
    assert t("telegram.command.status", "zh") == "查看系统状态"
    assert bot_commands("en")[0].description == "Start"


def test_log_redaction_masks_telegram_bot_token():
    from logging_utils import redact_secrets

    text = (
        "HTTP Request: POST "
        "https://api.telegram.org/bot123456:ABCdef_1234567890/sendMessage"
    )

    redacted = redact_secrets(text)

    assert "ABCdef_1234567890" not in redacted
    assert "bot123456:***67890/sendMessage" in redacted


def test_no_action_marker_is_internal_channel_response():
    from channels.base import is_no_action_response
    from config import SECRETARY_PERSONA

    assert is_no_action_response("NO_ACTION")
    assert is_no_action_response("Final output: NO_ACTION")
    assert not is_no_action_response("已记录。")
    assert not is_no_action_response("NO_ACTION 是内部定时任务标记。")
    assert "origin_channel` 为 `scheduled`" in SECRETARY_PERSONA
    assert "不要用 `send_message` 回答当前用户消息" in SECRETARY_PERSONA


@pytest.fixture
def fresh_db(tmp_path):
    """Database with a unique file per test, no shared state."""
    db = Database(db_path=str(tmp_path / "test.db"))
    return db


@pytest.fixture
def fake_agent_run(monkeypatch):
    """Patch agent.run so run_agent tests don't let TestModel choose real tools."""
    async def _fake_run(user_text, *args, **kwargs):
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)
    yield


def test_build_model_uses_config_api_key(monkeypatch):
    """LLM keys in config.yaml must be honored, not require env variables."""
    for env_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)

    anthropic = runtime.build_model(
        _config_for_model("anthropic", "claude-sonnet-4-20250514")
    )
    openai = runtime.build_model(_config_for_model("openai", "gpt-4o-mini"))
    deepseek = runtime.build_model(_config_for_model("deepseek", "deepseek-v4-flash"))
    gemini = runtime.build_model(_config_for_model("gemini", "gemini-1.5-pro"))

    assert anthropic.client.api_key == "test-key"
    assert openai.client.api_key == "test-key"
    assert deepseek.client.api_key == "test-key"
    assert gemini.client._api_client.api_key == "test-key"


def test_build_model_settings_support_deepseek_v4():
    from llm_models import build_model_settings

    cfg = _config_for_model("deepseek", "deepseek-v4-pro")
    cfg.llm.effort = "xhigh"
    cfg.llm.thinking = "enabled"

    settings = build_model_settings(cfg)

    assert settings["max_tokens"] == cfg.llm.max_tokens
    assert settings["openai_reasoning_effort"] == "max"
    assert settings["extra_body"] == {"thinking": {"type": "enabled"}}


def test_build_model_settings_support_openai_compatible_effort():
    from llm_models import build_model_settings

    cfg = _config_for_model("openai", "custom-reasoner")
    cfg.llm.base_url = "https://example.com/v1"
    cfg.llm.effort = "high"

    settings = build_model_settings(cfg)

    assert settings["openai_reasoning_effort"] == "high"
    assert "extra_body" not in settings


def test_build_model_settings_support_anthropic_effort():
    from llm_models import build_model_settings

    cfg = _config_for_model("anthropic", "claude-sonnet-4-20250514")
    cfg.llm.effort = "high"

    settings = build_model_settings(cfg)

    assert settings["thinking"] == "high"
    assert "openai_reasoning_effort" not in settings


def test_build_model_settings_support_deepseek_v4_anthropic_format():
    from llm_models import build_model_settings

    cfg = _config_for_model("anthropic", "deepseek-v4-pro")
    cfg.llm.base_url = "https://api.deepseek.com/anthropic"
    cfg.llm.effort = "max"
    cfg.llm.thinking = "enabled"

    settings = build_model_settings(cfg)

    assert "openai_reasoning_effort" not in settings
    assert settings["extra_body"] == {
        "output_config": {"effort": "max"},
        "thinking": {"type": "enabled"},
    }


# ---- Fix #1: message_history continuity ----


@pytest.mark.asyncio
async def test_two_turns_share_history(fresh_db, fake_agent_run):
    """The second turn must see the first turn's messages in DB."""
    await run_agent("第一条消息：我喜欢周末复盘", db=fresh_db)
    after_one = fresh_db.load_pydantic_messages()
    assert after_one, "first turn should have persisted some messages"

    await run_agent("第二条消息", db=fresh_db)
    after_two = fresh_db.load_pydantic_messages()
    assert len(after_two) > len(after_one), "second turn must append, not overwrite"

    # The first turn's user prompt must be reloadable
    found_first_prompt = False
    for msg in after_two:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", "")
            if isinstance(content, str) and "第一条消息" in content:
                found_first_prompt = True
                break
    assert found_first_prompt, "first turn user prompt should survive in reloaded history"


def test_cache_optimized_history_puts_runtime_after_replayed_history():
    summary = ModelRequest(
        parts=[
            SystemPromptPart(content="## 当前时间\n2026-05-03T10:17:14"),
            SystemPromptPart(
                content="Summary of previous conversation:\n\nkeep this summary"
            ),
        ]
    )
    user = ModelRequest(parts=[UserPromptPart(content="historical user turn")])
    assistant = ModelResponse(parts=[TextPart(content="historical reply")])

    first = runtime._cache_optimized_history(
        [summary, user, assistant],
        stable_context="stable-context",
        runtime_context="now=2026-06-21T17:30:00+08:00",
    )
    second = runtime._cache_optimized_history(
        [summary, user, assistant],
        stable_context="stable-context",
        runtime_context="now=2026-06-21T17:31:00+08:00",
    )

    def provider_content(messages):
        return [
            [(part.part_kind, getattr(part, "content", None)) for part in message.parts]
            for message in messages
        ]

    assert provider_content(first[:-1]) == provider_content(second[:-1]), (
        "the full replay prefix must remain cache-identical"
    )
    assert "2026-05-03" not in str(first)
    assert "keep this summary" in str(first[0])
    assert "historical user turn" in str(first[-3])
    assert "17:30" in str(first[-1])
    assert "17:31" in str(second[-1])


@pytest.mark.asyncio
async def test_run_agent_runtime_tail_is_temporary(fresh_db, monkeypatch):
    captured = {}

    async def _fake_run(user_text, *args, **kwargs):
        captured["history"] = kwargs["message_history"]
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)
    await run_agent("hello", db=fresh_db)

    sent_history = captured["history"]
    assert "Trusted Runtime Context" in str(sent_history[-1])
    persisted = fresh_db.load_pydantic_messages()
    assert persisted
    assert "Trusted Runtime Context" not in str(persisted)
    assert "hello" in str(persisted)


@pytest.mark.asyncio
async def test_no_action_skips_persistence(fresh_db, monkeypatch):
    """When the agent returns NO_ACTION, neither user prompt nor reply is persisted."""
    async def _fake_run(user_text, *args, **kwargs):
        return FakeRunResult(user_text, "NO_ACTION")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)
    before = len(fresh_db.load_pydantic_messages())
    reply = await run_agent("scheduled trigger", db=fresh_db, origin_channel="scheduled")
    assert "NO_ACTION" in reply.upper()
    after = len(fresh_db.load_pydantic_messages())
    assert after == before, "NO_ACTION must not pollute the persisted conversation"
    event_types = {event.type for event in fresh_db.get_agent_events()}
    assert {"run_started", "run_finished", "scheduled_no_action"} <= event_types


def test_save_pydantic_messages_skips_thinking_only_response(fresh_db):
    """Thinking-only assistant responses must never be persisted as history."""
    fresh_db.save_pydantic_messages(
        [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="Final output: NO_ACTION",
                        id="reasoning_content",
                        provider_name="deepseek",
                    )
                ],
                provider_name="deepseek",
            )
        ]
    )

    assert fresh_db.load_pydantic_messages() == []
    assert fresh_db.get_messages(limit=10) == []


def test_save_pydantic_messages_strips_thinking_from_valid_response(fresh_db):
    """Visible text/tool-call history is kept, but reasoning content is dropped."""
    fresh_db.save_pydantic_messages(
        [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="internal reasoning",
                        id="reasoning_content",
                        provider_name="deepseek",
                    ),
                    TextPart(content="visible"),
                ],
                provider_name="deepseek",
            )
        ]
    )

    loaded = fresh_db.load_pydantic_messages()
    assert len(loaded) == 1
    assert [part.part_kind for part in loaded[0].parts] == ["text"]
    assert loaded[0].parts[0].content == "visible"


def test_load_pydantic_messages_skips_legacy_thinking_only_blob(fresh_db):
    """Legacy bad blobs should not be replayed into future model requests."""
    bad = ModelResponse(
        parts=[
            ThinkingPart(
                content="Done. Final output: NO_ACTION.",
                id="reasoning_content",
                provider_name="deepseek",
            )
        ],
        provider_name="deepseek",
    )
    blob = bytes(ModelMessagesTypeAdapter.dump_json([bad]))
    fresh_db.save_message(
        "response",
        "Done. Final output: NO_ACTION.",
        pydantic_ai_msg=blob,
    )

    assert fresh_db.load_pydantic_messages() == []


def test_archive_invalid_pydantic_messages_nulls_bad_blob(fresh_db):
    """The DB maintenance path archives invalid active history blobs."""
    bad = ModelResponse(
        parts=[
            ThinkingPart(
                content="Done. Final output: NO_ACTION.",
                id="reasoning_content",
                provider_name="deepseek",
            )
        ],
        provider_name="deepseek",
    )
    blob = bytes(ModelMessagesTypeAdapter.dump_json([bad]))
    row = fresh_db.save_message(
        "response",
        "Done. Final output: NO_ACTION.",
        pydantic_ai_msg=blob,
    )

    assert fresh_db.archive_invalid_pydantic_messages() == 1
    stored = fresh_db.get_messages(limit=1)[0]
    assert stored.id == row.id
    assert stored.content == "Done. Final output: NO_ACTION."
    assert stored.pydantic_ai_msg is None


@pytest.mark.asyncio
async def test_run_agent_treats_thinking_only_no_action_as_no_action(fresh_db, monkeypatch):
    """A NO_ACTION hidden in reasoning must not be replaced by recovered stale text."""
    async def fake_run(user_text, *args, **kwargs):
        return FakeThinkingOnlyRunResult(
            user_text,
            "Done. Final output: NO_ACTION.",
            output="已记录。周末持仓不变。",
        )

    monkeypatch.setattr(runtime.agent, "run", fake_run)

    reply = await run_agent("scheduled trigger", db=fresh_db, origin_channel="scheduled")

    assert reply == "NO_ACTION"
    assert fresh_db.load_pydantic_messages() == []
    event_types = {event.type for event in fresh_db.get_agent_events()}
    assert "scheduled_no_action" in event_types


@pytest.mark.asyncio
async def test_run_agent_retries_non_no_action_thinking_only_response(fresh_db, monkeypatch):
    """Non-NO_ACTION thinking-only output gets one visible-content retry."""
    calls = {"n": 0}

    async def fake_run(user_text, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeThinkingOnlyRunResult(
                user_text,
                "Work completed, but this is only reasoning.",
                output="stale recovered text",
            )
        return FakeRunResult(user_text, "visible result")

    monkeypatch.setattr(runtime.agent, "run", fake_run)

    reply = await run_agent("do work", db=fresh_db)

    assert calls["n"] == 2
    assert reply == "visible result"
    loaded = fresh_db.load_pydantic_messages()
    assert loaded
    for msg in loaded:
        if isinstance(msg, ModelResponse):
            assert all(part.part_kind != "thinking" for part in msg.parts)


# ---- Fix #2: shell tool ----


def test_shell_tool_registered():
    """The shell tool must be registered on the agent (futu/longbridge skills depend on it)."""
    tool_names = list(runtime.agent._function_toolset.tools.keys())
    assert "shell" in tool_names, f"shell missing from {tool_names}"


def test_memory_tools_registered():
    """Long-term memory should have dedicated tools, not rely on generic file_write."""
    tool_names = list(runtime.agent._function_toolset.tools.keys())
    assert "load_skill" in tool_names
    assert "memory_read" in tool_names
    assert "memory_update" in tool_names


@pytest.mark.asyncio
async def test_load_skill_tool_loads_discovered_skill():
    """The model can load full skill instructions after seeing the skill index."""
    loaded = await runtime.load_skill(None, "review")
    assert loaded.startswith("# Skill: review")
    assert "复盘" in loaded


# ---- Fix #3: pre-run compaction ordering ----


def test_pre_run_compaction_runs_before_synthetic_context_assembly():
    """Compaction must process persisted history, never synthetic clock layers."""
    import inspect

    source = inspect.getsource(runtime.run_agent)
    assert source.index("maybe_auto_persist_compact") < source.index(
        "_build_context_layers"
    )
    assert not hasattr(runtime, "_history_processor")
    assert not runtime.agent.history_processors


def test_estimate_tokens_uses_real_tokenizer():
    """tiktoken-based estimate, not the v1 len/2 heuristic."""
    # 'hello world' → 2 tokens with cl100k_base
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("") == 0


def test_summary_preview_prefers_compacted_summary(fresh_db):
    """DB previews should make compacted summary rows obvious to humans."""
    fresh_db.save_pydantic_messages(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="system prompt should not lead preview"),
                    SystemPromptPart(
                        content="Summary of previous conversation:\n\n用户偏好：简洁"
                    ),
                ]
            )
        ]
    )

    row = fresh_db.get_messages(limit=1)[0]
    assert row.content.startswith("Summary of previous conversation:")


@pytest.mark.asyncio
async def test_compaction_prunes_tool_output_before_summary(monkeypatch):
    """Only the summarizer input gets oversized tool outputs truncated."""
    import compaction

    cfg = compaction.get_config()
    monkeypatch.setattr(cfg.history, "compact_tool_output_max_chars", 20)

    seen = {}

    class CapturingProcessor(SummarizationProcessor):
        async def _create_summary(self, messages_to_summarize):
            seen["formatted"] = compaction.format_messages_for_summary(
                messages_to_summarize
            )
            return "[摘要]"

    processor = CapturingProcessor(
        model=TestModel(custom_output_text="[摘要]"),
        trigger=("messages", 2),
        keep=("messages", 1),
        token_counter=compaction._count_tokens,
        max_input_tokens=100000,
    )
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="shell",
                    tool_call_id="call-1",
                    content="A" * 80,
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="tail")]),
    ]

    outcome = await compaction._run_processor_compaction(
        processor, messages, reason="test"
    )

    assert outcome.changed
    assert "TEMPORAL ANCHORING FOR SUMMARY ONLY" in seen["formatted"]
    assert "Tool output truncated for compaction" in seen["formatted"]
    assert "original 80 chars" in seen["formatted"]


def test_load_pydantic_messages_uses_token_budget_and_id_order(fresh_db, monkeypatch):
    """History replay keeps the newest affordable tail in stable chronological order."""
    import memory

    monkeypatch.setattr(memory, "_estimate_msg_tokens", lambda _msgs: 1)
    for text in ("oldest", "middle", "newest"):
        fresh_db.save_pydantic_messages(
            [ModelRequest(parts=[UserPromptPart(content=text)])]
        )

    loaded = fresh_db.load_pydantic_messages(token_budget=2, batch_size=2)
    contents = [
        part.content
        for msg in loaded
        for part in getattr(msg, "parts", [])
        if hasattr(part, "content")
    ]
    assert contents == ["middle", "newest"]


def test_load_pydantic_messages_loads_in_batches(fresh_db, monkeypatch):
    """A tiny budget should not deserialize the whole active message table."""
    for i in range(30):
        fresh_db.save_pydantic_messages(
            [ModelRequest(parts=[UserPromptPart(content=f"message {i}")])]
        )

    import memory

    real_validate = memory.ModelMessagesTypeAdapter.validate_json
    calls = {"n": 0}

    def counted_validate(*args, **kwargs):
        calls["n"] += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(memory.ModelMessagesTypeAdapter, "validate_json", counted_validate)
    loaded = fresh_db.load_pydantic_messages(token_budget=2, batch_size=5)

    assert loaded
    assert calls["n"] < 30


def test_telegram_status_has_no_items_or_legacy_limit_dependency():
    """D3 removed items and D2 removed the limit= API; /status must not regress."""
    import inspect
    from channels.telegram_channel import TelegramChannel

    source = inspect.getsource(TelegramChannel._status_command)
    assert "get_items" not in source
    assert "limit=200" not in source
    assert "memory.md" in source


def test_default_schedule_prompts_follow_memory_events_design():
    """Scheduled prompts should point at memory.md/events, not the removed items table."""
    from config import get_config

    schedules = get_config().schedules
    stale_prompt = schedules["stale_check"].prompt
    consolidation_prompt = schedules["memory_consolidation"].prompt
    pending_prompt = schedules["pending_response_check"].prompt
    system_review_prompt = schedules["system_review"].prompt

    assert "memory.md" in stale_prompt
    assert "from items" not in stale_prompt.lower()
    assert "get_items" not in stale_prompt
    assert "status='open'" in stale_prompt
    # NB: cron schedule times are user preferences living in the gitignored
    # config.yaml, not part of this design contract — asserting exact values
    # here breaks whenever a user retimes a task. This test only checks that
    # scheduled prompts point at memory.md/events instead of the removed
    # items table.
    assert "status='open'" in pending_prompt
    assert "memory_read" in consolidation_prompt
    assert "secretary-core" in consolidation_prompt
    assert "memory.md rules" in consolidation_prompt
    assert "50KB" in consolidation_prompt or "50 KB" in consolidation_prompt
    assert "40KB" in consolidation_prompt or "40 KB" in consolidation_prompt
    assert (
        "短 bullet" in consolidation_prompt
        or "short bullets" in consolidation_prompt
    )
    assert "status != 'open'" in consolidation_prompt
    assert "DELETE FROM events WHERE created_at <" in consolidation_prompt
    assert "send_message" in consolidation_prompt
    assert (
        "不调 send_message" in consolidation_prompt
        or "do not call send_message" in consolidation_prompt
    )
    assert "NO_ACTION" in consolidation_prompt
    assert (
        "不要写维护报告" in consolidation_prompt
        or "Do not write a maintenance report" in consolidation_prompt
    )
    assert schedules["system_review"].cron == "30 17 * * 0"
    assert "agent_events" in system_review_prompt
    assert "origin='scheduled'" in system_review_prompt
    assert "subject NOT LIKE" in system_review_prompt
    assert "date('now','+8 hours')" in system_review_prompt
    assert "date later than local_today" in system_review_prompt
    assert "WITH RECURSIVE days(day)" in system_review_prompt
    assert "Never invent dates outside those returned rows" in system_review_prompt
    assert "type='run_failed'" in system_review_prompt
    assert "type='send_message'" in system_review_prompt
    assert "status='open'" in system_review_prompt
    assert "subagent_step_finished" in system_review_prompt
    assert (
        "不要写入 memory.md" in system_review_prompt
        or "Do not write to memory.md" in system_review_prompt
    )


@pytest.mark.asyncio
async def test_force_compact_archives_and_replaces(fresh_db, fake_agent_run, monkeypatch):
    """force_compact must run the summarizer, archive old rows, and leave only [summary, *tail]."""
    import compaction

    # Build up enough history to compact.
    for i in range(8):
        await run_agent(f"消息 {i}", db=fresh_db)

    before_count = len(fresh_db.load_pydantic_messages())
    assert before_count >= 8, f"expected at least 8 messages, got {before_count}"

    # Stub the processor: TestModel summarizer (no API call) + a 1-token tail
    # budget so the head is non-empty and gets folded into a summary.
    def fake_processor(force=False):
        return SummarizationProcessor(
            model=TestModel(custom_output_text="[摘要] 历史已压缩"),
            trigger=("messages", 4),
            keep=("tokens", 1),
            token_counter=compaction._count_tokens,
            max_input_tokens=100000,
        )
    monkeypatch.setattr(compaction, "build_summarization_processor", fake_processor)

    result = await force_compact(fresh_db)
    assert "Compaction complete" in result, f"unexpected force_compact result: {result}"

    after = fresh_db.load_pydantic_messages()
    assert len(after) < before_count, "force_compact must shrink the active history"
    # First message should now be the summary (SystemPromptPart)
    first = after[0]
    parts = getattr(first, "parts", [])
    assert any(
        "summary" in getattr(p, "content", "").lower() or "摘要" in getattr(p, "content", "")
        for p in parts
    ), f"compacted history should start with a summary, got: {first}"
    assert all(
        "## 当前时间" not in getattr(p, "content", "")
        and "## Trusted Runtime Context" not in getattr(p, "content", "")
        for p in parts
    ), "compaction must not persist app-managed clocks"


@pytest.mark.asyncio
async def test_auto_compaction_uses_single_threshold_and_persists(fresh_db, monkeypatch):
    """Automatic compaction uses compress_threshold and rewrites active history."""
    import compaction

    cfg = compaction.get_config()
    monkeypatch.setattr(cfg.history, "context_tokens", 10)
    monkeypatch.setattr(cfg.history, "compress_threshold", 0.1)
    monkeypatch.setattr(cfg.history, "auto_compact", True)
    monkeypatch.setattr(cfg.history, "compact_min_active_messages", 4)
    monkeypatch.setattr(cfg.history, "compact_cooldown_minutes", 0)
    monkeypatch.setattr(compaction, "_last_auto_persist_compact_at", None)

    def fake_processor(force=False):
        return SummarizationProcessor(
            model=TestModel(custom_output_text="[摘要] 自动压缩"),
            trigger=("messages", 4),
            keep=("messages", 1),
            token_counter=compaction._count_tokens,
            max_input_tokens=100000,
        )

    monkeypatch.setattr(compaction, "build_summarization_processor", fake_processor)

    history = [
        ModelRequest(parts=[UserPromptPart(content=f"历史消息 {i}")])
        for i in range(4)
    ]
    fresh_db.save_pydantic_messages(history)

    outcome = await compaction.maybe_auto_persist_compact(fresh_db, history=history)

    assert outcome is not None
    assert outcome.changed
    after = fresh_db.load_pydantic_messages(token_budget=100000)
    assert len(after) < len(history)
    assert any(
        "摘要" in getattr(part, "content", "")
        for part in getattr(after[0], "parts", [])
    )


# ---- Fix #4: scheduler restart-survival ----


@pytest.mark.asyncio
async def test_runtime_created_task_survives_restart(fresh_db):
    """A task created at runtime (not in config.yaml) must reload on the next start."""
    # Simulate a runtime-created task being persisted before "restart"
    fresh_db.create_scheduled_task("runtime_task", "0 9 * * *", "test prompt")

    async def noop(_msg):
        return None

    sched = Scheduler(db=fresh_db, task_handler=noop)
    await sched.start()
    try:
        job_ids = [j.id for j in sched.get_jobs()]
        assert "runtime_task" in job_ids, (
            f"runtime-created task not loaded into scheduler: {job_ids}"
        )
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_yaml_orphan_protected_task_gets_cleaned(fresh_db, monkeypatch):
    """Removing a task from config.yaml must drop it from the DB on next start.

    Pins the orphan-cleanup behavior — LLM-created tasks (protected=0) must
    survive the same sweep, otherwise we'd be wiping the agent's own state.
    """
    from config import get_config

    cfg = get_config()
    # Pre-existing rows: one used to be in yaml (protected=1), one is the LLM's
    # runtime-created task (protected=0).
    fresh_db.create_scheduled_task("yaml_only", "0 7 * * *", "from yaml", protected=True)
    fresh_db.create_scheduled_task("llm_only", "0 8 * * *", "from llm", protected=False)
    # Snapshot then replace: yaml now declares NEITHER task — yaml_only is the
    # orphan we expect to be deleted.
    monkeypatch.setattr(cfg, "schedules", {})

    async def noop(_msg):
        return None

    sched = Scheduler(db=fresh_db, task_handler=noop)
    await sched.start()
    try:
        remaining_ids = {t.id for t in fresh_db.get_scheduled_tasks(enabled_only=False)}
        assert "yaml_only" not in remaining_ids, "orphaned yaml task must be deleted"
        assert "llm_only" in remaining_ids, "runtime-created (protected=0) task must survive"
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_uses_config_timezone(fresh_db, monkeypatch):
    """Scheduler must use config.timezone, not a hardcoded 'Asia/Shanghai'."""
    from config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg, "timezone", "America/New_York")

    async def noop(_msg):
        return None

    sched = Scheduler(db=fresh_db, task_handler=noop)
    assert str(sched._scheduler.timezone) == "America/New_York"


# ---- memory.md mechanism ----


@pytest.mark.asyncio
async def test_memory_md_injected_into_system_prompt(tmp_path, monkeypatch):
    """When memory.md exists, its contents must show up in dynamic_context output."""
    from runtime import dynamic_context, SecretaryDeps
    import runtime

    sentinel = "周末复盘风格偏向口语化"
    fake_memory = tmp_path / "memory.md"
    fake_memory.write_text(f"# 用户偏好\n{sentinel}\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", fake_memory)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps
    text = await dynamic_context(_Ctx(deps))

    assert sentinel in text, "memory.md content should be injected into system prompt"
    assert "## 长期记忆 (memory.md)" in text


@pytest.mark.asyncio
async def test_memory_md_missing_is_silent(tmp_path, monkeypatch):
    """A missing memory.md must not break dynamic_context."""
    from runtime import dynamic_context, SecretaryDeps
    import runtime

    monkeypatch.setattr(runtime, "MEMORY_FILE", tmp_path / "does_not_exist.md")

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps
    text = await dynamic_context(_Ctx(deps))

    assert "## 长期记忆 (memory.md)" not in text  # section absent when file missing
    assert text  # but the rest still rendered


@pytest.mark.asyncio
async def test_dynamic_context_keeps_stable_prefix_before_runtime_tail(tmp_path, monkeypatch):
    """Diagnostic context keeps policy before mutable runtime material."""
    from config import get_config
    from runtime import dynamic_context, SecretaryDeps
    from skills_loader import reset_skills_loader
    import runtime

    cfg = get_config()
    monkeypatch.setattr(cfg.history, "max_events", 1)
    monkeypatch.setattr(cfg, "language", "en")
    fake_memory = tmp_path / "memory.md"
    fake_memory.write_text("# 长期记忆\n- 稳定偏好 sentinel\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", fake_memory)
    reset_skills_loader()

    db = Database(db_path=":memory:")
    db.create_event("remind", "运行时事件 sentinel", status="open")
    db.create_event("note", "最近流水 sentinel", status="logged")
    deps = SecretaryDeps(
        db=db,
        current_time="2026-05-26T12:34:56+08:00",
        skill_content="# Skill: runtime-skill\n\n本轮技能 sentinel",
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    text = await dynamic_context(_Ctx(deps))

    schema_at = text.index("## 数据库表结构")
    language_at = text.index("## Language Policy")
    memory_at = text.index("## 长期记忆 (memory.md)")
    skill_index_at = text.index("## 可用技能索引")
    auto_skill_at = text.index("## 自动加载技能")
    event_context_at = text.index("## 事件上下文")
    loaded_skill_at = text.index("## 已加载技能", event_context_at)
    run_context_at = text.index("## Trusted Runtime Context", loaded_skill_at)
    event_at = text.index("运行时事件 sentinel", event_context_at)
    recent_at = text.index("最近流水 sentinel", event_context_at)

    assert schema_at < language_at < skill_index_at < auto_skill_at < memory_at < event_context_at
    assert event_context_at < event_at < recent_at < loaded_skill_at < run_context_at
    assert "shown 1 / total 1" in text
    assert "Configured language: `en`" in text
    assert "default user-facing language: English" in text
    assert "shown 1 / configured 1" in text
    assert text.rfind("## Trusted Runtime Context") > text.rfind("## 已加载技能")

    reset_skills_loader()


@pytest.mark.asyncio
async def test_dynamic_context_omits_resolved_events_from_recent_window(monkeypatch):
    from config import get_config
    from runtime import dynamic_context, SecretaryDeps

    cfg = get_config()
    monkeypatch.setattr(cfg.history, "max_events", 1)

    db = Database(db_path=":memory:")
    db.create_event("note", "logged continuity sentinel", status="logged")
    db.create_event("note", "resolved closed-loop sentinel", status="resolved")
    deps = SecretaryDeps(db=db, current_time="2026-05-26T12:34:56+08:00")

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    text = await dynamic_context(_Ctx(deps))

    assert "logged continuity sentinel" in text
    assert "resolved closed-loop sentinel" not in text
    assert "shown 1 / configured 1" in text


@pytest.mark.asyncio
async def test_dynamic_context_marks_event_temporal_metadata(monkeypatch):
    from config import get_config
    from runtime import dynamic_context, SecretaryDeps

    cfg = get_config()
    monkeypatch.setattr(cfg.history, "max_events", 1)

    db = Database(db_path=":memory:")
    db.create_event("remind", "6/25周三复盘：休市静默", status="open")
    deps = SecretaryDeps(
        db=db,
        current_time="2026-06-23T08:00:00+08:00",
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    text = await dynamic_context(_Ctx(deps))

    assert "event_date=2026-06-25" in text
    assert "relative_to_local_today=future" in text
    assert "weekday=周四" in text
    assert "weekday_mismatch=周三->周四" in text


def test_language_policy_supports_auto_and_falls_back_to_auto():
    from runtime import _language_policy

    auto_policy = _language_policy("auto")
    fallback_policy = _language_policy("klingon")

    assert "Configured language: `auto`" in auto_policy
    assert "the user's current language" in auto_policy
    assert "Configured language: `auto`" in fallback_policy
    assert "the user's current language" in fallback_policy


@pytest.mark.asyncio
async def test_auto_loaded_core_skill_injected_into_system_prompt(tmp_path, monkeypatch):
    """Auto-loaded project skills must appear even outside the main chat path."""
    from runtime import dynamic_context, SecretaryDeps
    from skills_loader import reset_skills_loader
    import runtime

    monkeypatch.setattr(runtime, "MEMORY_FILE", tmp_path / "does_not_exist.md")
    reset_skills_loader()

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    text = await dynamic_context(_Ctx(deps))

    assert "## 自动加载技能" in text
    assert "# Skill: secretary-core" in text
    assert "四种基本动作" in text

    reset_skills_loader()


@pytest.mark.asyncio
async def test_memory_update_appends_to_exact_section_title(tmp_path, monkeypatch):
    """memory_update should use the exact Markdown H2 title already in memory.md."""
    import runtime
    from runtime import memory_update, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# 长期记忆\n\n"
        "## 用户偏好\n\n"
        "## 协作约定\n\n"
        "## 在追踪的事项\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_update(
        _Ctx(deps),
        section="在追踪的事项",
        content="中东局势（伊朗、霍尔木兹海峡）",
    )

    text = memory_file.read_text(encoding="utf-8")
    assert "memory.md updated: 在追踪的事项" in result
    assert "- 中东局势（伊朗、霍尔木兹海峡）" in text
    assert "## 在追踪的事项" in text
    assert "## Tracked Items" not in text
    events = deps.db.get_agent_events()
    assert events[0].type == "memory_update"
    assert events[0].subject == "在追踪的事项"


@pytest.mark.asyncio
async def test_memory_update_accepts_english_section_names(tmp_path, monkeypatch):
    import runtime
    from runtime import memory_update, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)
    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_update(
        _Ctx(deps),
        section="User Preferences",
        content="Prefers concise updates",
    )

    text = memory_file.read_text(encoding="utf-8")
    assert "memory.md updated: User Preferences" in result
    assert "## User Preferences" in text
    assert "- Prefers concise updates" in text


@pytest.mark.asyncio
async def test_memory_update_rejects_missing_section_title(tmp_path, monkeypatch):
    import runtime
    from runtime import memory_update, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    memory_file.write_text("# 长期记忆\n\n## 用户偏好\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)
    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_update(
        _Ctx(deps),
        section="Tracked Items",
        content="Should not create a new section",
    )

    text = memory_file.read_text(encoding="utf-8")
    assert "Error: memory section not found: ## Tracked Items" in result
    assert "Existing sections: ## 用户偏好" in result
    assert "## Tracked Items" not in text
    assert deps.db.get_agent_events() == []


@pytest.mark.asyncio
async def test_db_execute_blocks_protected_runtime_tables(fresh_db):
    """Generic SQL writes must not bypass dedicated tool protections."""
    from runtime import db_execute, SecretaryDeps

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(SecretaryDeps(db=fresh_db))

    result = await db_execute(
        ctx,
        "UPDATE scheduled_tasks SET enabled = 0",
    )
    assert result.startswith("PERMISSION_DENIED")
    assert "scheduled_tasks" in result
    assert "protected" in result
    assert "tool: db_execute" in result

    result = await db_execute(
        ctx,
        "UPDATE events SET status = 'promoted' WHERE 1 = 0",
    )
    assert "Statement executed successfully" in result

    events = [
        event
        for event in fresh_db.get_agent_events()
        if event.type == "permission_denied"
    ]
    assert events
    assert events[0].subject == "db_execute:protected_table"


@pytest.mark.asyncio
async def test_db_execute_rejects_event_weekday_mismatch(fresh_db):
    """Event writes with explicit date+weekday mismatches fail before persistence."""
    from runtime import db_execute, SecretaryDeps

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(
        SecretaryDeps(
            db=fresh_db,
            current_time="2026-06-23T08:00:00+08:00",
        )
    )

    result = await db_execute(
        ctx,
        (
            "INSERT INTO events (type, content, status) "
            "VALUES ('remind', '6/25周三复盘：休市静默', 'open')"
        ),
    )

    assert result.startswith("Error: temporal validation failed")
    assert "2026-06-25 is 周四" in result
    assert fresh_db.get_events() == []

    events = [
        event
        for event in fresh_db.get_agent_events()
        if event.type == "temporal_validation_failed"
    ]
    assert events
    assert events[0].subject == "db_execute:events"


@pytest.mark.asyncio
async def test_schedule_task_rejects_prompt_weekday_mismatch(fresh_db):
    """Runtime-created schedule prompts get the same deterministic date guard."""
    from runtime import schedule_task, SecretaryDeps

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(
        SecretaryDeps(
            db=fresh_db,
            current_time="2026-06-23T08:00:00+08:00",
        )
    )

    result = await schedule_task(
        ctx,
        "create",
        task_id="bad_date",
        cron="0 8 * * *",
        prompt="6/25周三复盘：休市静默",
    )

    assert result.startswith("Error: temporal validation failed")
    assert "2026-06-25 is 周四" in result
    assert {task.id for task in fresh_db.get_scheduled_tasks(False)} == set()

    events = [
        event
        for event in fresh_db.get_agent_events()
        if event.type == "temporal_validation_failed"
    ]
    assert events
    assert events[0].subject == "schedule_task:create"


@pytest.mark.asyncio
async def test_shell_cwd_must_stay_inside_base_dir(fresh_db):
    """shell(cwd=...) must not escape secretary_v2."""
    from runtime import shell, SecretaryDeps, _shell_calls

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    _shell_calls.clear()
    result = await shell(
        _Ctx(SecretaryDeps(db=fresh_db)),
        "pwd",
        cwd="../",
    )
    assert result.startswith("PERMISSION_DENIED")
    assert "reason: cwd_escape" in result


@pytest.mark.asyncio
async def test_file_permission_denial_is_structured_and_recorded(fresh_db):
    """file_read/file_write permission failures should be stable for the LLM."""
    from runtime import file_read, file_write, SecretaryDeps

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(SecretaryDeps(db=fresh_db))

    result = await file_read(ctx, "config.yaml")
    assert result.startswith("PERMISSION_DENIED")
    assert "tool: file_read" in result
    assert "reason: protected_read_file" in result

    result = await file_write(ctx, "memory.md", "bad")
    assert result.startswith("PERMISSION_DENIED")
    assert "tool: file_write" in result
    assert "allowed_alternative: memory_update" in result

    events = [
        event
        for event in fresh_db.get_agent_events()
        if event.type == "permission_denied"
    ]
    subjects = [event.subject for event in events]
    assert "file_read:protected_read_file" in subjects
    assert "file_write:protected_file" in subjects


def test_memory_template_uses_three_sections():
    """The memory template should stay simple for user and LLM maintenance."""
    template = (runtime.BASE_DIR / "memory.md.example").read_text(encoding="utf-8")
    assert "## User Preferences" in template
    assert "## Collaboration Agreements" in template
    assert "## Tracked Items" in template
    assert "## Long-Term Facts" not in template
    assert "## Active Topics" not in template


# ---- Telegram outbox during restart window ----


@pytest.mark.asyncio
async def test_telegram_send_buffers_when_not_ready():
    """When the app isn't running (watchdog window), send() must queue, not drop."""
    from channels.telegram_channel import TelegramChannel

    async def _noop(_msg):
        return ""

    chan = TelegramChannel(
        bot_token="x",
        chat_id="123",
        message_handler=_noop,
    )
    assert not chan._is_ready()

    await chan.send("hello", user_id="123")
    await chan.send("world", user_id="456")
    assert len(chan._outbox) == 2
    assert chan._outbox[0] == ("hello", "123")
    assert chan._outbox[1] == ("world", "456")


@pytest.mark.asyncio
async def test_telegram_outbox_capacity_drops_oldest():
    """Bounded outbox: oldest message gets evicted, not the newest."""
    from channels.telegram_channel import TelegramChannel

    async def _noop(_msg):
        return ""

    chan = TelegramChannel(
        bot_token="x",
        chat_id="123",
        message_handler=_noop,
        outbox_capacity=3,
    )
    for i in range(5):
        await chan.send(f"msg-{i}")
    assert len(chan._outbox) == 3
    # oldest two ("msg-0", "msg-1") should have been dropped
    assert chan._outbox[0][0] == "msg-2"
    assert chan._outbox[-1][0] == "msg-4"


def test_telegram_plain_text_cleanup_removes_common_markdown():
    from channels.telegram_channel import _plain_text_for_telegram

    text = "### **午间提醒**\n- `APPLE` 需要复盘\n__重点__：不要追高"

    cleaned = _plain_text_for_telegram(text)

    assert cleaned == "午间提醒\n- APPLE 需要复盘\n重点：不要追高"


@pytest.mark.asyncio
async def test_telegram_send_chunks_uses_plain_text_without_parse_mode():
    from channels.telegram_channel import TelegramChannel

    async def _noop(_msg):
        return ""

    sent = []

    class FakeBot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    class FakeApp:
        bot = FakeBot()

    chan = TelegramChannel(
        bot_token="x",
        chat_id="123",
        message_handler=_noop,
    )
    chan.app = FakeApp()

    await chan._send_chunks_now(["**Title**\n- `item`"], "123")

    assert sent == [{"chat_id": "123", "text": "Title\n- item"}]
    assert "parse_mode" not in sent[0]


# ---- Feishu channel parity ----


@pytest.mark.asyncio
async def test_feishu_send_buffers_when_not_ready():
    from channels.feishu_channel import FeishuChannel

    async def _noop(_msg):
        return ""

    chan = FeishuChannel(
        app_id="cli_x",
        app_secret="secret",
        default_chat_id="oc_default",
        message_handler=_noop,
    )

    await chan.send("hello", user_id="oc_1")
    await chan.send("world", user_id="oc_2")

    assert len(chan._outbox) == 2
    assert chan._outbox[0] == ("hello", "oc_1")
    assert chan._outbox[1] == ("world", "oc_2")


@pytest.mark.asyncio
async def test_feishu_outbox_capacity_drops_oldest():
    from channels.feishu_channel import FeishuChannel

    async def _noop(_msg):
        return ""

    chan = FeishuChannel(
        app_id="cli_x",
        app_secret="secret",
        default_chat_id="oc_default",
        message_handler=_noop,
        outbox_capacity=3,
    )
    for i in range(5):
        await chan.send(f"msg-{i}")

    assert len(chan._outbox) == 3
    assert chan._outbox[0][0] == "msg-2"
    assert chan._outbox[-1][0] == "msg-4"


def test_feishu_plain_text_cleanup_removes_common_markdown():
    from channels.feishu_channel import _plain_text_for_feishu

    text = "### **午间提醒**\n- `APPLE` 需要复盘\n__重点__：不要追高"

    cleaned = _plain_text_for_feishu(text)

    assert cleaned == "午间提醒\n- APPLE 需要复盘\n重点：不要追高"


@pytest.mark.asyncio
async def test_feishu_resets_lark_ws_module_loop_when_it_is_running():
    import lark_oapi.ws.client as ws_client
    from channels.feishu_channel import _ensure_lark_ws_loop_not_running

    original_loop = ws_client.loop
    running_loop = asyncio.get_running_loop()
    ws_client.loop = running_loop

    try:
        _ensure_lark_ws_loop_not_running()
        assert ws_client.loop is not running_loop
        assert not ws_client.loop.is_running()
    finally:
        ws_client.loop.close()
        ws_client.loop = original_loop


@pytest.mark.asyncio
async def test_feishu_send_chunks_uses_lark_channel_plain_text():
    from channels.feishu_channel import FeishuChannel

    async def _noop(_msg):
        return ""

    sent = []

    class FakeSdkChannel:
        async def send(self, *args):
            sent.append(args)

    chan = FeishuChannel(
        app_id="cli_x",
        app_secret="secret",
        default_chat_id="oc_default",
        message_handler=_noop,
    )
    chan._running = True
    chan._ready = True
    chan._sdk_channel = FakeSdkChannel()

    await chan._send_chunks_now(["**Title**\n- `item`"], "oc_123")

    assert sent == [
        (
            "oc_123",
            {"text": "Title\n- item"},
            {"receive_id_type": "chat_id"},
        )
    ]


@pytest.mark.asyncio
async def test_feishu_inbound_message_uses_chat_id_as_routable_user_id():
    from channels.feishu_channel import FeishuChannel

    seen = []

    async def _handler(msg):
        seen.append(msg)
        return "ok"

    class FakeSdkChannel:
        async def send(self, *args):
            pass

    class FakeInbound:
        content_text = "hello"
        chat_id = "oc_chat"
        chat_type = "p2p"
        sender_id = "ou_user"
        sender_name = "User"
        message_id = "om_msg"
        mentioned_bot = False

    chan = FeishuChannel(
        app_id="cli_x",
        app_secret="secret",
        default_chat_id="oc_default",
        message_handler=_handler,
    )
    chan._running = True
    chan._ready = True
    chan._sdk_channel = FakeSdkChannel()

    await chan._handle_message(FakeInbound())

    assert seen[0].channel == "feishu"
    assert seen[0].user_id == "ou_user"
    assert seen[0].conversation_id == "oc_chat"
    assert seen[0].metadata["sender_id"] == "ou_user"


def test_webhook_response_channel_prefers_configured_default_outgoing():
    from main import SecretaryApp

    class FakeChannel:
        pass

    app = SecretaryApp.__new__(SecretaryApp)
    app._configured_default_outgoing = "feishu"
    app._last_chat_channel_name = "telegram"
    app.channels = {
        "telegram": FakeChannel(),
        "feishu": FakeChannel(),
    }

    assert app._resolve_webhook_response_channel() is app.channels["feishu"]


def test_webhook_response_channel_respects_http_default_without_chat_fallback():
    from main import SecretaryApp

    class FakeChannel:
        pass

    app = SecretaryApp.__new__(SecretaryApp)
    app._configured_default_outgoing = "http"
    app._last_chat_channel_name = "telegram"
    app.channels = {
        "http": FakeChannel(),
        "telegram": FakeChannel(),
        "feishu": FakeChannel(),
    }

    assert app._resolve_webhook_response_channel() is app.channels["http"]


def test_webhook_response_channel_falls_back_to_recent_chat_channel():
    from main import SecretaryApp

    class FakeChannel:
        pass

    app = SecretaryApp.__new__(SecretaryApp)
    app._configured_default_outgoing = "missing"
    app._last_chat_channel_name = None
    app.channels = {
        "telegram": FakeChannel(),
        "feishu": FakeChannel(),
    }

    assert app._resolve_webhook_response_channel() is None

    app._remember_chat_channel("telegram")
    assert app._resolve_webhook_response_channel() is app.channels["telegram"]

    app._remember_chat_channel("feishu")
    assert app._resolve_webhook_response_channel() is app.channels["feishu"]


def test_channel_all_mode_starts_every_configured_non_cli_channel():
    from main import SecretaryApp

    class FakeChannel:
        def __init__(self, name):
            self.name = name

    app = SecretaryApp.__new__(SecretaryApp)
    app.channel_type = "all"
    app.channels = {
        "cli": FakeChannel("cli"),
        "telegram": FakeChannel("telegram"),
        "feishu": FakeChannel("feishu"),
        "http": FakeChannel("http"),
    }

    assert [channel.name for channel in app._channels_to_start()] == [
        "telegram",
        "feishu",
        "http",
    ]


def test_parse_args_accepts_all_channel(monkeypatch):
    from main import parse_args

    monkeypatch.setattr("sys.argv", ["main.py", "--channel", "all"])

    assert parse_args().channel == "all"


# ---- getUpdates stall watchdog ----


@pytest.mark.asyncio
async def test_stall_tracked_request_updates_timestamp_on_success(monkeypatch):
    """Successful do_request must advance last_seen_at.

    This is the core liveness signal: if it stops advancing, the watchdog
    triggers a rebuild. Verified without standing up a real Application —
    we patch the parent do_request so we don't touch the network.
    """
    import time as _time
    from channels.telegram_channel import _StallTrackedRequest
    from telegram.request import HTTPXRequest

    req = _StallTrackedRequest(connection_pool_size=1, http_version="1.1")
    initial = req.last_seen_at

    async def _fake(self, *args, **kwargs):
        return (200, b"{}")

    monkeypatch.setattr(HTTPXRequest, "do_request", _fake)
    # Sleep enough that monotonic clearly advances
    await asyncio.sleep(0.05)
    await req.do_request("u", "POST")

    assert req.last_seen_at > initial


@pytest.mark.asyncio
async def test_stall_tracked_request_does_not_advance_on_failure(monkeypatch):
    """If do_request raises (e.g. socket wedge surfaced as exception), the
    timestamp stays put — that's exactly what lets the watchdog see a stall.
    """
    from channels.telegram_channel import _StallTrackedRequest
    from telegram.request import HTTPXRequest
    from telegram.error import TimedOut

    req = _StallTrackedRequest(connection_pool_size=1, http_version="1.1")
    initial = req.last_seen_at

    async def _fake_raise(self, *args, **kwargs):
        raise TimedOut("simulated wedge")

    monkeypatch.setattr(HTTPXRequest, "do_request", _fake_raise)
    await asyncio.sleep(0.05)
    with pytest.raises(TimedOut):
        await req.do_request("u", "POST")

    assert req.last_seen_at == initial


# ---- LLM transient error retry ----


@pytest.mark.asyncio
async def test_run_agent_retries_on_transient_error(fresh_db, monkeypatch):
    """Transient httpx errors during agent.run must be retried up to 3 times."""
    import runtime
    import httpx

    calls = {"n": 0}

    async def flaky_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("simulated network blip")
        return FakeRunResult(args[0], "recovered")

    monkeypatch.setattr(runtime.agent, "run", flaky_run)
    monkeypatch.setattr(runtime, "_LLM_BACKOFF_BASE_SEC", 0.01)  # speed up test

    reply = await runtime.run_agent("hi", db=fresh_db)
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
    assert reply == "recovered"


@pytest.mark.asyncio
async def test_run_agent_logs_usage(fresh_db, monkeypatch, caplog):
    """Provider usage is observable without changing the response path."""
    async def fake_run(user_text, *args, **kwargs):
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", fake_run)
    caplog.set_level("INFO", logger="runtime")

    reply = await runtime.run_agent("hi", db=fresh_db)

    assert reply == "ok"
    assert "[run_agent] usage input=11 output=7 total=18 requests=1" in caplog.text


@pytest.mark.asyncio
async def test_run_agent_usage_failure_is_nonfatal(fresh_db, monkeypatch):
    """A provider/result without usage support must not break the turn."""
    async def fake_run(user_text, *args, **kwargs):
        return FakeRunResult(user_text, "ok", usage_error=RuntimeError("no usage"))

    monkeypatch.setattr(runtime.agent, "run", fake_run)

    reply = await runtime.run_agent("hi", db=fresh_db)

    assert reply == "ok"


@pytest.mark.asyncio
async def test_get_last_usage_populated_after_run(fresh_db, fake_agent_run):
    """get_last_usage exposes the last run's provider usage for /status."""
    await runtime.run_agent("hi", db=fresh_db)
    usage = runtime.get_last_usage()
    assert usage is not None
    assert usage["input_tokens"] == FakeUsage.input_tokens
    assert usage["total_tokens"] == FakeUsage.input_tokens + FakeUsage.output_tokens
    assert usage["cache_read_tokens"] == 0
    assert usage["origin"] == "cli"
    assert "at" in usage


def test_usage_payload_uses_pydantic_cache_fields():
    """OpenAI/Anthropic-style cache fields should flow through Pydantic AI usage."""
    class Usage:
        input_tokens = 100
        output_tokens = 9
        requests = 1
        cache_read_tokens = 70
        cache_write_tokens = 20
        details = {}

    payload = runtime._build_usage_payload(Usage(), origin_channel="test", at="now")

    assert payload["cache_hit_tokens"] == 70
    assert payload["cache_miss_tokens"] == 10
    assert payload["cache_hit_ratio"] == 0.875


def test_usage_payload_adapts_deepseek_details():
    """DeepSeek's OpenAI-compatible extras should become normalized cache metrics."""
    class Usage:
        input_tokens = 100
        output_tokens = 9
        requests = 1
        cache_read_tokens = 0
        cache_write_tokens = 0
        details = {"prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20}

    payload = runtime._build_usage_payload(Usage(), origin_channel="test", at="now")

    assert payload["cache_hit_tokens"] == 80
    assert payload["cache_miss_tokens"] == 20
    assert payload["cache_hit_ratio"] == 0.8
    assert payload["details"]["prompt_cache_hit_tokens"] == 80


def test_telegram_status_surfaces_cache_metrics():
    """Telegram /status should show recent cache usage when available."""
    import inspect
    from channels.telegram_channel import TelegramChannel

    source = inspect.getsource(TelegramChannel._status_command)
    assert "telegram.status" in source
    assert "telegram.status.cache_metrics" in source
    assert "cache_hit_tokens" in source
    assert "cache_write_tokens" in source


@pytest.mark.asyncio
async def test_run_agent_does_not_retry_on_permanent_error(fresh_db, monkeypatch):
    """Non-transient errors (e.g. ValueError from bad config) must surface immediately."""
    import runtime

    calls = {"n": 0}

    async def always_fails(*args, **kwargs):
        calls["n"] += 1
        raise ValueError("bad request")

    monkeypatch.setattr(runtime.agent, "run", always_fails)

    with pytest.raises(ValueError):
        await runtime.run_agent("hi", db=fresh_db)
    assert calls["n"] == 1, "permanent errors must not be retried"


# ---- send_message tool routing for scheduled tasks (Bug A) ----


@pytest.mark.asyncio
async def test_send_message_for_scheduled_origin_skips_logical_user_id():
    """For scheduled-task runs, ctx.deps.user_id is the literal 'scheduler' —
    a logical identity, not a routable chat_id. The send_message tool must
    NOT forward it to the underlying channel (otherwise Telegram returns
    'Chat not found')."""
    from runtime import send_message, SecretaryDeps
    from memory import Database

    captured = {"user_id": "<unset>"}

    class FakeChannel:
        async def send(self, text, user_id=None):
            captured["user_id"] = user_id

    fake = FakeChannel()
    deps = SecretaryDeps(
        db=Database(db_path=":memory:"),
        origin_channel="scheduled",
        user_id="scheduler",
        channels={"telegram": fake},
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    # Manually call the tool function (the @agent.tool decorator returns the function itself)
    result = await send_message(_Ctx(deps), "hello", channel="telegram")
    assert "Message sent via telegram" in result
    assert captured["user_id"] is None, (
        f"scheduled origin must not forward 'scheduler' as user_id, got {captured['user_id']!r}"
    )


@pytest.mark.asyncio
async def test_send_message_for_private_origin_forwards_user_id():
    """Private chats still use sender id as the routable target."""
    from runtime import send_message, SecretaryDeps
    from memory import Database

    captured = {"user_id": "<unset>"}

    class FakeChannel:
        async def send(self, text, user_id=None):
            captured["user_id"] = user_id

    fake = FakeChannel()
    deps = SecretaryDeps(
        db=Database(db_path=":memory:"),
        origin_channel="telegram",
        user_id="804416037",
        channels={"telegram": fake},
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    await send_message(_Ctx(deps), "hello")
    assert captured["user_id"] == "804416037", (
        "user-originated runs must forward user_id so the reply lands in the right chat"
    )


@pytest.mark.asyncio
async def test_send_message_for_group_origin_prefers_conversation_id():
    """Group chats must route to chat/group id, not the sender id."""
    from runtime import send_message, SecretaryDeps
    from memory import Database

    captured = {"user_id": "<unset>"}

    class FakeChannel:
        async def send(self, text, user_id=None):
            captured["user_id"] = user_id

    fake = FakeChannel()
    deps = SecretaryDeps(
        db=Database(db_path=":memory:"),
        origin_channel="telegram",
        user_id="sender-123",
        conversation_id="-100987654321",
        channels={"telegram": fake},
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    await send_message(_Ctx(deps), "hello")
    assert captured["user_id"] == "-100987654321"
