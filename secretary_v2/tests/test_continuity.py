"""Regression tests for the P0 fixes:

1. message_history continuity (load + persist pydantic-ai messages)
2. shell tool registered with proper guardrails
3. pre-run compaction happens before synthetic context assembly
4. Scheduler resyncs runtime-created tasks from DB on restart

These tests use Pydantic AI's TestModel so they don't burn API tokens.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
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
from memory import Database, Message
from runtime import run_agent
from scheduler import Scheduler
from pydantic_ai_summarization import SummarizationProcessor
from pydantic_ai.usage import RequestUsage
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

    def all_messages(self):
        return [
            ModelRequest(parts=[UserPromptPart(content=self._user_text)]),
            ModelResponse(
                parts=[TextPart(content=self.output)],
                usage=RequestUsage(input_tokens=13, output_tokens=7),
            ),
        ]


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_tokens": 0},
        {"compress_threshold": 1.0},
        {"context_tokens": 100, "compress_threshold": 0.5, "tail_token_budget": 50},
        {"compact_cooldown_minutes": -1},
        {"webhook_retention_days": -1},
    ],
)
def test_history_config_rejects_invalid_compaction_budgets(kwargs):
    with pytest.raises(ValueError):
        HistoryConfig(**kwargs)


def test_config_rejects_context_window_smaller_than_output_reserve(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  max_tokens: 4096\nhistory:\n"
        "  context_tokens: 6000\n  compress_threshold: 0.75\n"
        "  tail_token_budget: 1000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="request safety reserve"):
        Config.load(str(config_file))


def test_startup_warns_when_anthropic_cache_is_not_adapted(caplog):
    import main

    config = SimpleNamespace(llm=SimpleNamespace(provider="anthropic"))
    caplog.set_level("WARNING", logger="main")

    main._warn_if_anthropic_cache_unconfigured(config)

    assert "does not currently add Anthropic prompt-cache cache_control" in caplog.text


def test_startup_does_not_emit_anthropic_cache_warning_for_other_providers(caplog):
    import main

    config = SimpleNamespace(llm=SimpleNamespace(provider="deepseek"))
    caplog.set_level("WARNING", logger="main")

    main._warn_if_anthropic_cache_unconfigured(config)

    assert "prompt-cache cache_control" not in caplog.text


def test_config_loads_memory_path_and_backup_switch(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "timezone: UTC\nmemory:\n  path: data/custom-memory.md\n  backup_enabled: false\n",
        encoding="utf-8",
    )

    cfg = Config.load(str(config_file))

    assert cfg.memory.path == "data/custom-memory.md"
    assert cfg.memory.backup_enabled is False


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


@pytest.mark.asyncio
async def test_run_agent_isolates_history_by_session_key(fresh_db, monkeypatch):
    captured = []

    async def _fake_run(user_text, *args, **kwargs):
        captured.append(kwargs["message_history"])
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)

    await run_agent(
        "alpha session only",
        db=fresh_db,
        origin_channel="telegram",
        user_id="sender-a",
        conversation_id="chat-a",
    )
    await run_agent(
        "beta session",
        db=fresh_db,
        origin_channel="telegram",
        user_id="sender-b",
        conversation_id="chat-b",
    )

    assert "alpha session only" not in str(captured[1])
    alpha_key = runtime.build_session_key(
        channel="telegram", user_id="sender-a", conversation_id="chat-a"
    )
    beta_key = runtime.build_session_key(
        channel="telegram", user_id="sender-b", conversation_id="chat-b"
    )
    assert "alpha session only" in str(
        fresh_db.load_pydantic_messages(session_key=alpha_key, include_legacy=False)
    )
    assert "beta session" in str(
        fresh_db.load_pydantic_messages(session_key=beta_key, include_legacy=False)
    )


def test_http_history_cutoff_uses_local_calendar_days(monkeypatch):
    from config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.history, "webhook_retention_days", 5)
    monkeypatch.setattr(cfg, "timezone", "Asia/Shanghai")
    now = datetime(
        2026,
        7,
        20,
        16,
        37,
        tzinfo=timezone(timedelta(hours=8)),
    )

    assert runtime.history_created_after_for_channel("http", now=now) == datetime(
        2026, 7, 15, 16
    )
    assert runtime.history_created_after_for_channel("telegram", now=now) is None


def test_context_visibility_hides_messages_without_deleting_them(fresh_db):
    fresh_db.save_pydantic_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="hidden historical request")]),
            ModelResponse(parts=[TextPart(content="hidden historical response")]),
        ]
    )

    affected = fresh_db.execute_statement(
        "UPDATE messages SET context_visible = 0 WHERE context_visible = 1"
    )

    assert affected == 2
    assert fresh_db.load_pydantic_messages() == []
    rows = fresh_db.get_messages(limit=10)
    assert len(rows) == 2
    assert {row.context_visible for row in rows} == {0}


def test_context_visibility_never_replays_a_partial_turn(fresh_db):
    fresh_db.save_pydantic_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="hidden request boundary")]),
            ModelResponse(parts=[TextPart(content="orphan response boundary")]),
        ]
    )
    request_id = min(row.id for row in fresh_db.get_messages(limit=10))
    fresh_db.execute_statement(
        "UPDATE messages SET context_visible = 0 WHERE id = ?", [request_id]
    )

    assert fresh_db.load_pydantic_messages() == []
    assert len(fresh_db.get_messages(limit=10)) == 2


def test_context_visibility_filters_automatic_event_queries(fresh_db):
    hidden = fresh_db.create_event("note", "hidden event", status="logged")
    visible = fresh_db.create_event("note", "visible event", status="logged")
    fresh_db.execute_statement(
        "UPDATE events SET context_visible = 0 WHERE id = ?", [hidden.id]
    )

    automatic = fresh_db.get_events_excluding_statuses([], limit=10)

    assert [event.id for event in automatic] == [visible.id]
    assert {event.id for event in fresh_db.get_events(limit=10)} == {
        hidden.id,
        visible.id,
    }


@pytest.mark.asyncio
async def test_run_agent_uses_legacy_history_only_for_empty_session(fresh_db, monkeypatch):
    captured = []
    fresh_db.save_pydantic_messages(
        [ModelRequest(parts=[UserPromptPart(content="legacy global context")])]
    )

    async def _fake_run(user_text, *args, **kwargs):
        captured.append(kwargs["message_history"])
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)
    kwargs = {
        "db": fresh_db,
        "origin_channel": "telegram",
        "user_id": "sender-a",
        "conversation_id": "chat-a",
    }
    await run_agent("first isolated turn", **kwargs)
    await run_agent("second isolated turn", **kwargs)

    assert "legacy global context" in str(captured[0])
    assert "legacy global context" not in str(captured[1])
    assert "first isolated turn" in str(captured[1])


@pytest.mark.asyncio
async def test_http_retention_does_not_fall_back_to_legacy_history(
    fresh_db, monkeypatch
):
    captured = []
    fresh_db.save_pydantic_messages(
        [ModelRequest(parts=[UserPromptPart(content="legacy global context")])]
    )

    async def _fake_run(user_text, *args, **kwargs):
        captured.append(kwargs["message_history"])
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)
    await run_agent(
        "first webhook turn",
        db=fresh_db,
        origin_channel="http",
        user_id="webhook_user",
        conversation_id="webhook_user",
    )

    assert "legacy global context" not in str(captured[0])


@pytest.mark.asyncio
async def test_run_agent_self_test_skips_legacy_history(fresh_db, monkeypatch):
    captured = []
    fresh_db.save_pydantic_messages(
        [ModelRequest(parts=[UserPromptPart(content="legacy global context")])]
    )

    async def _fake_run(user_text, *args, **kwargs):
        captured.append(kwargs["message_history"])
        return FakeRunResult(user_text, "OK")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)

    reply = await run_agent(
        "Diagnostic ping. Reply only with OK and do not call any tools.",
        db=fresh_db,
        origin_channel="self_test",
        user_id="self_test",
    )

    assert reply == "OK"
    assert "legacy global context" not in str(captured[0])


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
async def test_http_run_prioritizes_memory_facts_over_stale_history(
    fresh_db, tmp_path, monkeypatch
):
    captured = {}
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# Long-term memory\n\n- Current holdings: dividend ETF only\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    session_key = runtime.build_session_key(
        channel="http",
        user_id="webhook_user",
        conversation_id="webhook_user",
    )
    fresh_db.save_pydantic_messages(
        [
            ModelRequest(
                parts=[UserPromptPart(content="Old history: CR Land is a holding")]
            ),
            ModelResponse(parts=[TextPart(content="Tracking CR Land as a holding")]),
        ],
        session_key=session_key,
        channel="http",
    )

    async def _fake_run(user_text, *args, **kwargs):
        captured["history"] = kwargs["message_history"]
        return FakeRunResult(user_text, "ok")

    monkeypatch.setattr(runtime.agent, "run", _fake_run)
    await run_agent(
        "New market data",
        db=fresh_db,
        origin_channel="http",
        user_id="webhook_user",
        conversation_id="webhook_user",
    )

    history = captured["history"]
    assert "Old history: CR Land is a holding" in str(history[:-1])
    runtime_tail = str(history[-1])
    assert "Current holdings: dividend ETF only" in runtime_tail
    assert "`memory.md` is authoritative" in runtime_tail
    assert "Do not infer current holdings from tracked/watchlist items" in runtime_tail


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


def test_save_pydantic_messages_keeps_thinking_with_visible_response(fresh_db):
    """Usable reasoning is persisted when the response has visible content."""
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
    assert [part.part_kind for part in loaded[0].parts] == ["thinking", "text"]
    assert loaded[0].parts[0].content == "internal reasoning"
    assert loaded[0].parts[1].content == "visible"


def test_save_pydantic_messages_drops_empty_thinking_from_valid_response(fresh_db):
    """Empty reasoning is removed without discarding an otherwise valid response."""
    fresh_db.save_pydantic_messages(
        [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="   ",
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


def test_load_pydantic_messages_drops_incomplete_tool_call_chain(fresh_db):
    tool_call = ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="db_query",
                args={"sql": "select 1"},
                tool_call_id="call_missing",
            )
        ]
    )
    final_text = ModelResponse(parts=[TextPart(content="final answer")])

    fresh_db.save_message(
        "response",
        "",
        pydantic_ai_msg=bytes(ModelMessagesTypeAdapter.dump_json([tool_call])),
    )
    fresh_db.save_message("request", "human-readable tool result without blob")
    fresh_db.save_message(
        "response",
        "final answer",
        pydantic_ai_msg=bytes(ModelMessagesTypeAdapter.dump_json([final_text])),
    )

    loaded = fresh_db.load_pydantic_messages()

    assert len(loaded) == 1
    assert isinstance(loaded[0], ModelResponse)
    assert loaded[0].parts[0].content == "final answer"


def test_load_pydantic_messages_drops_orphan_retry_prompt(fresh_db):
    retry_prompt = ModelRequest(
        parts=[
            RetryPromptPart(
                content="Unknown tool name: 'memory_read'",
                tool_name="memory_read",
                tool_call_id="call_orphan",
            )
        ]
    )
    final_text = ModelResponse(parts=[TextPart(content="final answer")])

    fresh_db.save_message(
        "request",
        "tool retry without matching tool call",
        pydantic_ai_msg=bytes(ModelMessagesTypeAdapter.dump_json([retry_prompt])),
    )
    fresh_db.save_message(
        "response",
        "final answer",
        pydantic_ai_msg=bytes(ModelMessagesTypeAdapter.dump_json([final_text])),
    )

    loaded = fresh_db.load_pydantic_messages()

    assert len(loaded) == 1
    assert isinstance(loaded[0], ModelResponse)
    assert loaded[0].parts[0].content == "final answer"


def test_load_pydantic_messages_keeps_complete_tool_call_chain(fresh_db):
    fresh_db.save_pydantic_messages(
        [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="query the database first",
                        id="reasoning_content",
                        provider_name="deepseek",
                    ),
                    ToolCallPart(
                        tool_name="db_query",
                        args={"sql": "select 1"},
                        tool_call_id="call_ok",
                    ),
                ],
                provider_name="deepseek",
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="db_query",
                        content="[(1,)]",
                        tool_call_id="call_ok",
                    )
                ]
            ),
            ModelResponse(parts=[TextPart(content="final answer")]),
        ]
    )

    loaded = fresh_db.load_pydantic_messages()

    assert [msg.kind for msg in loaded] == ["response", "request", "response"]
    assert [part.part_kind for part in loaded[0].parts] == ["thinking", "tool-call"]
    assert loaded[0].parts[0].content == "query the database first"
    assert loaded[1].parts[0].part_kind == "tool-return"
    assert loaded[2].parts[0].content == "final answer"


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
    assert "record_event" in tool_names
    assert "update_event_summary" in tool_names
    assert "memory_view" in tool_names
    assert "memory_str_replace" in tool_names
    assert "memory_insert" in tool_names
    # Old blob-based tools must be fully removed, not left as aliases.
    assert "memory_read" not in tool_names
    assert "memory_update" not in tool_names


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

    source = inspect.getsource(runtime._run_agent_unlocked)
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


def test_load_pydantic_messages_time_window_keeps_complete_turns(fresh_db):
    session_key = "agent:secretary:http:conversation:webhook_user"
    fresh_db.save_pydantic_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="old request")]),
            ModelResponse(parts=[TextPart(content="orphan old response")]),
            ModelRequest(parts=[UserPromptPart(content="recent request")]),
            ModelResponse(parts=[TextPart(content="recent response")]),
        ],
        session_key=session_key,
        channel="http",
    )
    cutoff = datetime(2026, 7, 16)
    with fresh_db.get_session() as session:
        rows = session.query(Message).order_by(Message.id).all()
        rows[0].created_at = cutoff - timedelta(seconds=2)
        rows[1].created_at = cutoff + timedelta(seconds=1)
        rows[2].created_at = cutoff + timedelta(seconds=2)
        rows[3].created_at = cutoff + timedelta(seconds=3)
        recent_row_ids = [rows[2].id, rows[3].id]
        session.commit()

    loaded_row_ids = []
    loaded = fresh_db.load_pydantic_messages(
        session_key=session_key,
        include_legacy=False,
        created_after=cutoff,
        loaded_row_ids=loaded_row_ids,
    )
    contents = [
        part.content
        for message in loaded
        for part in getattr(message, "parts", [])
        if isinstance(part, (UserPromptPart, TextPart))
    ]

    assert contents == ["recent request", "recent response"]
    assert loaded_row_ids == recent_row_ids
    assert len(fresh_db.get_messages(limit=10)) == 4


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

    from channel_commands import build_status_text

    source = inspect.getsource(build_status_text)
    assert "get_items" not in source
    assert "limit=200" not in source
    assert "get_memory_file_path" in source


def test_default_schedule_prompts_follow_memory_events_design():
    """Scheduled prompts query durable context without forcing bookkeeping events."""
    from config import get_config

    schedules = get_config().schedules
    briefing_prompt = schedules["morning_briefing"].prompt
    trend_prompt = schedules["morning_trend_scan"].prompt
    review_prompt = schedules["review_reminder"].prompt
    stale_prompt = schedules["stale_check"].prompt
    consolidation_prompt = schedules["memory_consolidation"].prompt
    pending_prompt = schedules["pending_response_check"].prompt
    system_review_prompt = schedules["system_review"].prompt

    assert "memory.md" in stale_prompt
    assert "from items" not in stale_prompt.lower()
    assert "get_items" not in stale_prompt
    assert "open events" in stale_prompt.lower()
    assert "do not create any event" in stale_prompt.lower()
    assert "status='open'" in stale_prompt
    # NB: cron schedule times are user preferences living in the gitignored
    # config.yaml, not part of this design contract — asserting exact values
    # here breaks whenever a user retimes a task. This test only checks that
    # scheduled prompts point at memory.md/events instead of the removed
    # items table.
    assert "status='open'" in pending_prompt
    local_day_filter = "date(created_at,'+8 hours') = date('now','+8 hours')"
    assert local_day_filter in briefing_prompt
    assert local_day_filter in review_prompt
    assert "market_calendar" in trend_prompt
    assert "Do not claim live prices or market moves" in trend_prompt
    assert "source_channel='scheduled'" in pending_prompt
    assert "Resolve every older duplicate" in pending_prompt
    assert "Do not create any event for this check" in pending_prompt
    for proactive_prompt in (
        briefing_prompt,
        trend_prompt,
        review_prompt,
        stale_prompt,
    ):
        normalized = proactive_prompt.lower()
        assert "create an event" not in normalized
        assert "create a new event" not in normalized
        assert "db_execute to create" not in normalized
    assert "memory_view" in consolidation_prompt
    assert "secretary-core" in consolidation_prompt
    assert "memory.md rules" in consolidation_prompt
    assert "50KB" in consolidation_prompt or "50 KB" in consolidation_prompt
    assert "40KB" in consolidation_prompt or "40 KB" in consolidation_prompt
    assert (
        "短 bullet" in consolidation_prompt
        or "short bullets" in consolidation_prompt
    )
    assert "status != 'open'" in consolidation_prompt
    assert "context_visible=1" in consolidation_prompt
    assert "datetime('now','-7 days')" in consolidation_prompt
    assert "UPDATE events SET context_visible=0" in consolidation_prompt
    assert "UPDATE messages SET context_visible=0" in consolidation_prompt
    assert "Never DELETE historical rows" in consolidation_prompt
    assert "send_message" in consolidation_prompt
    assert (
        "不调 send_message" in consolidation_prompt
        or "do not call send_message" in consolidation_prompt.lower()
    )
    assert "NO_ACTION" in consolidation_prompt
    assert (
        "不要写维护报告" in consolidation_prompt
        or "write a maintenance report" in consolidation_prompt
    )
    assert len(consolidation_prompt) < 1800
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
    assert "sent_count=0 alone is not an anomaly" in system_review_prompt
    assert "do not call any tools" not in system_review_prompt.lower()
    assert "do not call send_message or mutation tools" in system_review_prompt.lower()
    assert len(system_review_prompt) < 3000
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
    assert result.status == "completed", f"unexpected force_compact result: {result}"

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
    monkeypatch.setattr(cfg.history, "context_tokens", 400)
    monkeypatch.setattr(cfg.history, "compress_threshold", 0.5)
    monkeypatch.setattr(cfg.history, "auto_compact", True)
    monkeypatch.setattr(cfg.history, "compact_min_active_messages", 4)
    monkeypatch.setattr(cfg.history, "compact_cooldown_minutes", 0)
    monkeypatch.setattr(compaction, "_last_auto_persist_compact_at", {})

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
    loaded_row_ids = []
    history = fresh_db.load_pydantic_messages(
        token_budget=100000,
        loaded_row_ids=loaded_row_ids,
    )

    outcome = await compaction.maybe_auto_persist_compact(
        fresh_db,
        history=history,
        loaded_row_ids=loaded_row_ids,
    )

    assert outcome is not None
    assert outcome.changed
    after = fresh_db.load_pydantic_messages(token_budget=100000)
    assert len(after) < len(history)
    assert any(
        "摘要" in getattr(part, "content", "")
        for part in getattr(after[0], "parts", [])
    )


@pytest.mark.asyncio
async def test_auto_compaction_cooldown_is_scoped_per_session(monkeypatch):
    """A recent compaction in session A must not suppress eligible session B."""
    import compaction

    cfg = compaction.get_config().history
    monkeypatch.setattr(cfg, "auto_compact", True)
    monkeypatch.setattr(cfg, "context_tokens", 100)
    monkeypatch.setattr(cfg, "compress_threshold", 0.5)
    monkeypatch.setattr(cfg, "compact_min_active_messages", 4)
    monkeypatch.setattr(cfg, "compact_cooldown_minutes", 120)
    monkeypatch.setattr(compaction, "_last_auto_persist_compact_at", {})
    monkeypatch.setattr(compaction, "_count_tokens", lambda _history: 60)

    calls = []

    async def fake_run_compaction(history, force=False, reason="auto"):
        calls.append(str(history[0]))
        return compaction.CompactOutcome(
            compacted=[history[-1]],
            changed=True,
            failed=False,
            error=None,
            before_messages=len(history),
            after_messages=1,
            before_tokens=10,
            after_tokens=1,
            reason=reason,
        )

    monkeypatch.setattr(compaction, "run_compaction", fake_run_compaction)
    monkeypatch.setattr(compaction, "persist_compacted_snapshot", lambda *a, **kw: 4)

    history_a = [
        ModelRequest(parts=[UserPromptPart(content=f"A-{i}")]) for i in range(4)
    ]
    history_b = [
        ModelRequest(parts=[UserPromptPart(content=f"B-{i}")]) for i in range(4)
    ]

    first_a = await compaction.maybe_auto_persist_compact(
        object(),
        history=history_a,
        session_key="session-a",
        loaded_row_ids=[1, 2, 3, 4],
    )
    first_b = await compaction.maybe_auto_persist_compact(
        object(),
        history=history_b,
        session_key="session-b",
        loaded_row_ids=[5, 6, 7, 8],
    )
    second_a = await compaction.maybe_auto_persist_compact(
        object(),
        history=history_a,
        session_key="session-a",
        loaded_row_ids=[1, 2, 3, 4],
    )

    assert first_a is not None and first_a.changed
    assert first_b is not None and first_b.changed
    assert second_a is None
    assert len(calls) == 2
    assert set(compaction._last_auto_persist_compact_at) == {"session-a", "session-b"}


@pytest.mark.asyncio
async def test_auto_compaction_failure_backoff_is_scoped(monkeypatch):
    import compaction

    cfg = compaction.get_config().history
    monkeypatch.setattr(cfg, "auto_compact", True)
    monkeypatch.setattr(cfg, "context_tokens", 10)
    monkeypatch.setattr(cfg, "compress_threshold", 0.5)
    monkeypatch.setattr(cfg, "compact_min_active_messages", 4)
    monkeypatch.setattr(cfg, "compact_cooldown_minutes", 0)
    monkeypatch.setattr(compaction, "_auto_compact_failures", {})
    monkeypatch.setattr(compaction, "_count_tokens", lambda _history: 10)

    calls = []

    async def fail(history, force=False, reason="auto"):
        calls.append(history[0].parts[0].content)
        return compaction.CompactOutcome(
            compacted=history,
            changed=False,
            failed=True,
            error="summary unavailable",
            before_messages=4,
            after_messages=4,
            before_tokens=10,
            after_tokens=10,
            reason=reason,
        )

    monkeypatch.setattr(compaction, "run_compaction", fail)
    history_a = [ModelRequest(parts=[UserPromptPart(content="A")])] * 4
    history_b = [ModelRequest(parts=[UserPromptPart(content="B")])] * 4

    assert await compaction.maybe_auto_persist_compact(
        object(), history=history_a, session_key="a", loaded_row_ids=[]
    )
    assert await compaction.maybe_auto_persist_compact(
        object(), history=history_a, session_key="a", loaded_row_ids=[]
    ) is None
    assert await compaction.maybe_auto_persist_compact(
        object(), history=history_b, session_key="b", loaded_row_ids=[]
    )
    assert calls == ["A", "B"]


def test_compaction_snapshot_only_archives_loaded_rows(fresh_db, monkeypatch):
    """Rows outside the replay budget must remain active and unsilently archived."""
    import compaction
    import memory

    monkeypatch.setattr(memory, "_estimate_msg_tokens", lambda _msgs: 1)
    session_key = "agent:secretary:telegram:conversation:bounded"
    for text in ("oldest", "middle", "newest"):
        fresh_db.save_pydantic_messages(
            [ModelRequest(parts=[UserPromptPart(content=text)])],
            session_key=session_key,
        )

    loaded_row_ids = []
    loaded = fresh_db.load_pydantic_messages(
        token_budget=2,
        session_key=session_key,
        include_legacy=False,
        loaded_row_ids=loaded_row_ids,
    )
    assert "oldest" not in str(loaded)

    compacted = [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content="Summary of previous conversation:\n\nrecent summary"
                )
            ]
        ),
        loaded[-1],
    ]
    outcome = compaction.CompactOutcome(
        compacted=compacted,
        changed=True,
        failed=False,
        error=None,
        before_messages=len(loaded),
        after_messages=len(compacted),
        before_tokens=2,
        after_tokens=1,
        reason="test",
    )

    fresh_db.replace_pydantic_messages_snapshot(
        outcome.compacted,
        archive_row_ids=loaded_row_ids,
        session_key=session_key,
    )
    after = fresh_db.load_pydantic_messages(
        token_budget=100,
        session_key=session_key,
        include_legacy=False,
    )

    assert "oldest" in str(after)
    assert "recent summary" in str(after)


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
    db.create_event(
        "remind",
        "运行时事件 full body should stay out of context",
        status="open",
        summary="运行时事件 sentinel",
    )
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
    assert "full body should stay out of context" not in text
    assert "Configured language: `en`" in text
    assert "default user-facing language: English" in text
    assert "shown 1 / configured 1" in text
    assert text.rfind("## Trusted Runtime Context") > text.rfind("## 已加载技能")

    reset_skills_loader()


@pytest.mark.asyncio
async def test_dynamic_context_warns_recorded_webhook_not_to_duplicate_event():
    from runtime import dynamic_context, SecretaryDeps

    deps = SecretaryDeps(
        db=Database(db_path=":memory:"),
        origin_channel="http",
        message_metadata={
            "record": "logged",
            "webhook_run_id": "hook_test",
            "recorded_event_id": 123,
        },
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    text = await dynamic_context(_Ctx(deps))

    assert "Webhook Record Notice" in text
    assert "already been recorded verbatim in `events`" in text
    assert "event id `123`" in text
    assert "status='logged'" in text
    assert "update_event_summary(123, summary)" in text
    assert "summarize, restate, reclassify" in text
    assert "cross-channel visibility" in text
    assert "distinct actionable follow-up" in text
    assert "Webhook Delivery Contract" in text
    assert "not a scheduled task" in text
    assert "`memory.md` is authoritative" in text
    assert "Do not infer current holdings from tracked/watchlist items" in text
    assert "put the user-visible reply in final output" in text
    assert "Do not call `send_message` to answer the current webhook" in text


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
async def test_memory_insert_adds_entry_after_line(tmp_path, monkeypatch):
    """memory_insert adds one bullet after a 1-based line, preserving the rest verbatim."""
    import runtime
    from runtime import memory_insert, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# 长期记忆\n\n"
        "## 用户偏好\n\n"
        "## 协作约定\n\n"
        "## 在追踪的事项\n"
        "- 已有追踪项\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_insert(
        _Ctx(deps),
        insert_line=8,
        insert_text="- 中东局势（伊朗、霍尔木兹海峡）",
    )

    assert "memory.md edited" in result
    assert memory_file.read_text(encoding="utf-8") == (
        "# 长期记忆\n\n"
        "## 用户偏好\n\n"
        "## 协作约定\n\n"
        "## 在追踪的事项\n"
        "- 已有追踪项\n"
        "- 中东局势（伊朗、霍尔木兹海峡）\n"
    )
    events = deps.db.get_agent_events()
    assert events[0].type == "memory_update"
    assert events[0].subject == "- 中东局势（伊朗、霍尔木兹海峡）"
    assert '"tool": "memory_insert"' in events[0].payload_json


@pytest.mark.asyncio
async def test_memory_insert_skips_exact_duplicate_entry(tmp_path, monkeypatch):
    """Re-inserting an identical entry (e.g. a retried run) must not duplicate it."""
    import runtime
    from runtime import memory_insert, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    original = "# 长期记忆\n\n## 在追踪的事项\n- 已有追踪项\n"
    memory_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_insert(_Ctx(deps), insert_line=4, insert_text="- 已有追踪项")

    assert "identical entry already exists" in result
    assert memory_file.read_text(encoding="utf-8") == original
    assert deps.db.get_agent_events() == []


@pytest.mark.asyncio
async def test_memory_insert_rejects_out_of_range_line(tmp_path, monkeypatch):
    import runtime
    from runtime import memory_insert, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    original = "# 长期记忆\n\n## 用户偏好\n"
    memory_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_insert(_Ctx(deps), insert_line=99, insert_text="- 新条目")

    assert "Error: invalid insert_line 99" in result
    assert memory_file.read_text(encoding="utf-8") == original
    assert deps.db.get_agent_events() == []


def test_daily_memory_backup_creates_one_snapshot_per_day(tmp_path, monkeypatch):
    import runtime
    from config import get_config

    cfg = _config_for_model("anthropic", "claude-sonnet-4-20250514")
    cfg.timezone = "UTC"
    monkeypatch.setattr(get_config, "_config", cfg, raising=False)
    monkeypatch.setattr(runtime, "MEMORY_FILE", tmp_path / "memory.md")
    monkeypatch.setattr(
        runtime,
        "_local_today_for_config",
        lambda: runtime.date(2026, 6, 29),
    )

    runtime.MEMORY_FILE.write_text("# Memory\n\nfirst version\n", encoding="utf-8")

    first = runtime.ensure_daily_memory_backup()
    runtime.MEMORY_FILE.write_text("# Memory\n\nsecond version\n", encoding="utf-8")
    second = runtime.ensure_daily_memory_backup()

    assert first == second
    assert first.name == "memory-2026-06-29.md"
    assert first.parent == tmp_path
    assert first.read_text(encoding="utf-8") == "# Memory\n\nfirst version\n"


def test_daily_memory_backup_follows_configured_memory_path(tmp_path, monkeypatch):
    import runtime
    from config import get_config

    memory_file = tmp_path / "notes" / "custom-memory.md"
    memory_file.parent.mkdir()
    memory_file.write_text("# Memory\n\nconfigured path\n", encoding="utf-8")

    cfg = _config_for_model("anthropic", "claude-sonnet-4-20250514")
    cfg.memory.path = str(memory_file)
    monkeypatch.setattr(get_config, "_config", cfg, raising=False)
    monkeypatch.setattr(
        runtime,
        "_local_today_for_config",
        lambda: runtime.date(2026, 6, 29),
    )

    backup = runtime.ensure_daily_memory_backup()

    assert backup == tmp_path / "notes" / "custom-memory-2026-06-29.md"
    assert backup.read_text(encoding="utf-8") == "# Memory\n\nconfigured path\n"


def test_daily_memory_backup_respects_config_switch(tmp_path, monkeypatch):
    import runtime
    from config import get_config

    cfg = _config_for_model("anthropic", "claude-sonnet-4-20250514")
    cfg.memory.backup_enabled = False
    monkeypatch.setattr(get_config, "_config", cfg, raising=False)
    monkeypatch.setattr(runtime, "MEMORY_FILE", tmp_path / "memory.md")

    runtime.MEMORY_FILE.write_text("# Memory\n", encoding="utf-8")

    assert runtime.ensure_daily_memory_backup() is None
    assert list(tmp_path.glob("memory-*.md")) == []


@pytest.mark.asyncio
async def test_memory_str_replace_rewrites_unique_snippet(tmp_path, monkeypatch):
    """A unique verbatim match is rewritten in place; the rest stays untouched."""
    import runtime
    from runtime import memory_str_replace, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# 长期记忆\n\n"
        "## 用户偏好\n"
        "- 喜欢喝茶\n"
        "- 早上不要打扰\n\n"
        "## 协作约定\n"
        "- 旧约定\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_str_replace(
        _Ctx(deps),
        old_str="- 喜欢喝茶",
        new_str="- 喜欢喝咖啡，不喝茶",
    )

    assert "memory.md edited" in result
    assert memory_file.read_text(encoding="utf-8") == (
        "# 长期记忆\n\n"
        "## 用户偏好\n"
        "- 喜欢喝咖啡，不喝茶\n"
        "- 早上不要打扰\n\n"
        "## 协作约定\n"
        "- 旧约定\n"
    )
    events = deps.db.get_agent_events()
    assert events[0].type == "memory_update"
    assert '"tool": "memory_str_replace"' in events[0].payload_json


@pytest.mark.asyncio
async def test_memory_str_replace_deletes_entry_with_empty_new_str(
    tmp_path, monkeypatch
):
    """Stale entries are removable: new_str='' with a trailing-newline anchor."""
    import runtime
    from runtime import memory_str_replace, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# 长期记忆\n\n"
        "## 在追踪的事项\n"
        "- 已平仓的旧持仓\n"
        "- 仍在跟踪的计划\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_str_replace(_Ctx(deps), old_str="- 已平仓的旧持仓\n")

    assert "memory.md edited" in result
    assert memory_file.read_text(encoding="utf-8") == (
        "# 长期记忆\n\n"
        "## 在追踪的事项\n"
        "- 仍在跟踪的计划\n"
    )


@pytest.mark.asyncio
async def test_memory_str_replace_rejects_zero_matches(tmp_path, monkeypatch):
    import runtime
    from runtime import memory_str_replace, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    original = "# 长期记忆\n\n## 用户偏好\n- 喜欢喝茶\n"
    memory_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_str_replace(
        _Ctx(deps), old_str="- 喜欢喝奶茶", new_str="- 改写"
    )

    assert "did not appear verbatim" in result
    assert memory_file.read_text(encoding="utf-8") == original
    assert deps.db.get_agent_events() == []


@pytest.mark.asyncio
async def test_memory_str_replace_rejects_ambiguous_matches(tmp_path, monkeypatch):
    """Multiple matches are refused with line numbers so the model can disambiguate."""
    import runtime
    from runtime import memory_str_replace, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    original = (
        "# 长期记忆\n\n"
        "## 用户偏好\n"
        "- 关注 AI\n\n"
        "## 在追踪的事项\n"
        "- 关注 AI\n"
    )
    memory_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_str_replace(_Ctx(deps), old_str="- 关注 AI", new_str="x")

    assert "appears 2 times" in result
    assert "lines 4, 7" in result
    assert memory_file.read_text(encoding="utf-8") == original
    assert deps.db.get_agent_events() == []


@pytest.mark.asyncio
async def test_memory_capacity_guard_blocks_growth_allows_prune(tmp_path, monkeypatch):
    """Writes past the injection cap are refused, but pruning stays possible."""
    import runtime
    from runtime import memory_insert, memory_str_replace, SecretaryDeps

    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# Memory\n\n## Tracked Items\n- old entry aaaaaaaaaa\n- keep\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)
    monkeypatch.setattr(runtime, "MEMORY_SOFT_CAP_CHARS", 10)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    original = memory_file.read_text(encoding="utf-8")
    grow = await memory_insert(_Ctx(deps), insert_line=4, insert_text="- new entry")
    assert "over the 10 char injection cap" in grow
    assert memory_file.read_text(encoding="utf-8") == original

    prune = await memory_str_replace(_Ctx(deps), old_str="- old entry aaaaaaaaaa\n")
    assert "memory.md edited" in prune
    assert "- old entry" not in memory_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_memory_view_shows_line_numbers_and_range(tmp_path, monkeypatch):
    import runtime
    from runtime import memory_view

    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# 长期记忆\n\n## 用户偏好\n- 喜欢喝茶\n", encoding="utf-8"
    )
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)

    full = await memory_view(None)
    assert "    1\t# 长期记忆" in full
    assert "    4\t- 喜欢喝茶" in full

    ranged = await memory_view(None, view_range=[3, -1])
    assert "    3\t## 用户偏好" in ranged
    assert "# 长期记忆" not in ranged


@pytest.mark.asyncio
async def test_memory_insert_materializes_scaffold_on_configured_path(
    tmp_path, monkeypatch
):
    """First write on a missing configured file creates the default scaffold."""
    from config import get_config
    from runtime import memory_insert, SecretaryDeps

    cfg = _config_for_model("anthropic", "claude-sonnet-4-20250514")
    cfg.memory.path = str(tmp_path / "configured-memory.md")
    monkeypatch.setattr(get_config, "_config", cfg, raising=False)

    deps = SecretaryDeps(db=Database(db_path=":memory:"))

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await memory_insert(
        _Ctx(deps),
        insert_line=6,
        insert_text="- Configured memory path works",
    )

    configured_memory = tmp_path / "configured-memory.md"
    assert "memory.md edited" in result
    text = configured_memory.read_text(encoding="utf-8")
    assert "## User Preferences" in text
    assert "- Configured memory path works" in text


def test_load_memory_md_marks_truncation(tmp_path, monkeypatch):
    """Prompt injection past the cap must be visibly truncated, never silent."""
    import runtime

    memory_file = tmp_path / "memory.md"
    memory_file.write_text("x" * 40, encoding="utf-8")
    monkeypatch.setattr(runtime, "MEMORY_FILE", memory_file)
    monkeypatch.setattr(runtime, "MEMORY_SOFT_CAP_CHARS", 10)

    injected = runtime._load_memory_md()
    assert injected.startswith("x" * 10)
    assert "truncated" in injected
    assert "memory_str_replace" in injected


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

    fresh_db.save_pydantic_messages(
        [ModelRequest(parts=[UserPromptPart(content="visibility candidate")])]
    )
    result = await db_execute(
        ctx,
        "UPDATE messages SET context_visible = 0 WHERE context_visible = 1",
    )
    assert "Statement executed successfully" in result
    assert fresh_db.load_pydantic_messages() == []
    assert fresh_db.get_messages(limit=1)[0].context_visible == 0

    result = await db_execute(
        ctx,
        "UPDATE messages SET content = 'changed' WHERE 1 = 0",
    )
    assert result.startswith("PERMISSION_DENIED")

    result = await db_execute(
        ctx,
        "UPDATE messages SET context_visible = 0, content = 'changed' WHERE 1 = 0",
    )
    assert result.startswith("PERMISSION_DENIED")

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
async def test_record_event_persists_source_metadata(fresh_db):
    from runtime import SecretaryDeps, record_event

    deps = SecretaryDeps(
        db=fresh_db,
        origin_channel="telegram",
        user_id="sender-123",
        conversation_id="-100987",
        reply_to_id="msg-42",
        thread_id="topic-7",
        session_key="agent:secretary:telegram:conversation:-100987:thread:topic-7",
        message_metadata={"chat_type": "supergroup"},
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    result = await record_event(
        _Ctx(deps),
        "remind",
        "明天复盘 OpenClaw 方案",
        status="open",
        summary="OpenClaw 复盘",
    )

    assert "Event recorded" in result
    event = fresh_db.get_events()[0]
    assert event.source_channel == "telegram"
    assert event.session_key == deps.session_key
    assert event.source_message_id == "msg-42"
    assert event.summary == "OpenClaw 复盘"
    assert "sender-123" in event.metadata_json
    assert "topic-7" in event.metadata_json


@pytest.mark.asyncio
async def test_update_event_summary_only_updates_allowed_webhook_event(fresh_db):
    from runtime import SecretaryDeps, update_event_summary

    allowed = fresh_db.create_event("note", "raw webhook payload")
    other = fresh_db.create_event("note", "other payload")
    deps = SecretaryDeps(
        db=fresh_db,
        origin_channel="http",
        message_metadata={"recorded_event_id": allowed.id},
    )

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    blocked = await update_event_summary(_Ctx(deps), other.id, "wrong target")
    assert "may only update summary" in blocked

    result = await update_event_summary(_Ctx(deps), allowed.id, "clean index line")
    assert "Event summary updated" in result

    events = {event.id: event for event in fresh_db.get_events(limit=10)}
    assert events[allowed.id].summary == "clean index line"
    assert events[allowed.id].content == "raw webhook payload"
    assert events[other.id].summary == "other payload"


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
async def test_file_permission_denial_is_structured_and_recorded(fresh_db, monkeypatch):
    """file_read/file_write permission failures should be stable for the LLM."""
    from config import get_config
    from runtime import file_read, file_write, SecretaryDeps

    cfg = _config_for_model("anthropic", "claude-sonnet-4-20250514")
    cfg.memory.path = "data/custom-memory.md"
    monkeypatch.setattr(get_config, "_config", cfg, raising=False)

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
    assert "allowed_alternative: memory_str_replace / memory_insert" in result

    result = await file_write(ctx, "data/custom-memory.md", "bad")
    assert result.startswith("PERMISSION_DENIED")
    assert "tool: file_write" in result
    assert "allowed_alternative: memory_str_replace / memory_insert" in result

    events = [
        event
        for event in fresh_db.get_agent_events()
        if event.type == "permission_denied"
    ]
    subjects = [event.subject for event in events]
    assert "file_read:protected_read_file" in subjects
    assert "file_write:protected_file" in subjects


def test_file_edit_tool_registered():
    """Generic files get the same anchored-edit model as memory.md."""
    tool_names = list(runtime.agent._function_toolset.tools.keys())
    assert "file_edit" in tool_names


@pytest.mark.asyncio
async def test_file_edit_replaces_unique_snippet(tmp_path, monkeypatch):
    """file_edit shares the memory editing core: unique verbatim anchor."""
    import runtime
    from runtime import file_edit, SecretaryDeps

    monkeypatch.setattr(runtime, "BASE_DIR", tmp_path)
    target = tmp_path / "data" / "notes.md"
    target.parent.mkdir()
    target.write_text("# Notes\n\n- old line\n- keep\n", encoding="utf-8")

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(SecretaryDeps(db=Database(db_path=":memory:")))

    result = await file_edit(
        ctx, "data/notes.md", old_str="- old line", new_str="- new line"
    )
    assert "data/notes.md edited" in result
    assert target.read_text(encoding="utf-8") == "# Notes\n\n- new line\n- keep\n"

    result = await file_edit(ctx, "data/notes.md", old_str="- missing")
    assert "did not appear verbatim in data/notes.md" in result
    assert "file_read" in result
    assert target.read_text(encoding="utf-8") == "# Notes\n\n- new line\n- keep\n"


@pytest.mark.asyncio
async def test_file_edit_denied_on_memory_path(fresh_db):
    """file_edit must respect the same protected-file policy as file_write."""
    from runtime import file_edit, SecretaryDeps

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(SecretaryDeps(db=fresh_db))

    result = await file_edit(ctx, "memory.md", old_str="x", new_str="y")
    assert result.startswith("PERMISSION_DENIED")
    assert "tool: file_edit" in result
    assert "allowed_alternative: memory_str_replace / memory_insert" in result


@pytest.mark.asyncio
async def test_file_write_refuses_overwrite_beyond_read_cap(tmp_path, monkeypatch):
    """Overwriting a file the model cannot fully read would drop the unseen tail."""
    import runtime
    from runtime import file_write, SecretaryDeps

    monkeypatch.setattr(runtime, "BASE_DIR", tmp_path)
    monkeypatch.setattr(runtime, "FILE_READ_CAP_CHARS", 10)
    target = tmp_path / "data" / "big.log"
    target.parent.mkdir()
    target.write_text("x" * 40, encoding="utf-8")

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    ctx = _Ctx(SecretaryDeps(db=Database(db_path=":memory:")))

    result = await file_write(ctx, "data/big.log", "tiny")
    assert "Error" in result
    assert "file_edit" in result
    assert target.read_text(encoding="utf-8") == "x" * 40

    result = await file_write(ctx, "data/big.log", "-more", mode="append")
    assert "successfully" in result
    assert target.read_text(encoding="utf-8") == "x" * 40 + "-more"


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


@pytest.mark.asyncio
async def test_feishu_inbound_message_reacts_then_replies_to_source_message():
    from channels.feishu_channel import FeishuChannel

    async def _handler(_msg):
        return "**done**"

    calls = []

    class FakeHttpResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class FakeHttpClient:
        async def post(self, path, json, headers):
            calls.append((path, json, headers))
            if path == "/open-apis/auth/v3/tenant_access_token/internal":
                return FakeHttpResponse(
                    {"code": 0, "tenant_access_token": "t-token", "expire": 7200}
                )
            return FakeHttpResponse({"code": 0, "msg": "ok"})

    class FakeSdkChannel:
        async def send(self, *args):
            raise AssertionError(f"unexpected fallback send: {args}")

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
        http_client_factory=FakeHttpClient,
    )
    chan._running = True
    chan._ready = True
    chan._sdk_channel = FakeSdkChannel()

    await chan._handle_message(FakeInbound())

    assert calls[0][0] == "/open-apis/auth/v3/tenant_access_token/internal"
    assert calls[1][0] == "/open-apis/im/v1/messages/om_msg/reactions"
    assert calls[1][1] == {"reaction_type": {"emoji_type": "THUMBSUP"}}
    assert calls[2][0] == "/open-apis/im/v1/messages/om_msg/reply"
    assert calls[2][1]["msg_type"] == "text"
    assert calls[2][1]["content"] == '{"text": "done"}'
    assert calls[2][2]["Authorization"] == "Bearer t-token"


@pytest.mark.asyncio
async def test_feishu_inbound_message_falls_back_to_send_when_reply_fails():
    from channels.feishu_channel import FeishuChannel

    async def _handler(_msg):
        return "ok"

    sent = []

    class FakeHttpResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class FakeHttpClient:
        async def post(self, path, json, headers):
            if path == "/open-apis/auth/v3/tenant_access_token/internal":
                return FakeHttpResponse(
                    {"code": 0, "tenant_access_token": "t-token", "expire": 7200}
                )
            if path.endswith("/reply"):
                return FakeHttpResponse({"code": 230050, "msg": "not visible"})
            return FakeHttpResponse({"code": 0, "msg": "ok"})

    class FakeSdkChannel:
        async def send(self, *args):
            sent.append(args)

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
        http_client_factory=FakeHttpClient,
    )
    chan._running = True
    chan._ready = True
    chan._sdk_channel = FakeSdkChannel()

    await chan._handle_message(FakeInbound())

    assert sent == [
        (
            "oc_chat",
            {"text": "ok"},
            {"receive_id_type": "chat_id"},
        )
    ]


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


@pytest.mark.asyncio
async def test_http_webhook_record_logged_creates_event(fresh_db):
    import httpx
    from channels.http_channel import HTTPChannel

    handled = asyncio.Event()
    seen = []

    async def _handler(msg):
        seen.append(msg)
        handled.set()
        return "ok"

    chan = HTTPChannel(
        token="secret",
        message_handler=_handler,
        event_recorder=fresh_db.create_event,
    )

    transport = httpx.ASGITransport(app=chan.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/hooks",
            json={
                "message": "AAPL 放量突破失败，RS 转弱",
                "summary": "AAPL 突破失败",
                "record": "logged",
            },
            headers={"x-webhook-token": "secret"},
        )

    assert response.status_code == 202
    run_id = response.json()["runId"]
    await asyncio.wait_for(handled.wait(), timeout=1)

    event = fresh_db.get_events(limit=1)[0]
    assert event.type == "note"
    assert event.status == "logged"
    assert event.content == "AAPL 放量突破失败，RS 转弱"
    assert event.summary == "AAPL 突破失败"
    assert event.source_channel == "http"
    assert event.session_key == "agent:secretary:http:conversation:webhook_user"
    assert run_id in event.metadata_json
    assert seen[0].metadata["record"] == "logged"
    assert seen[0].metadata["webhook_run_id"] == run_id
    assert seen[0].metadata["recorded_event_id"] == event.id
    assert seen[0].metadata["recorded_event_summary"] == "AAPL 突破失败"
    assert seen[0].metadata["summary_supplied"] is True


@pytest.mark.asyncio
async def test_http_messages_route_is_registered_without_webhook_record():
    import httpx
    from channels.http_channel import HTTPChannel

    async def _handler(_msg):
        return "ok"

    chan = HTTPChannel(token="secret", message_handler=_handler)
    await chan.send("queued", user_id="target")

    transport = httpx.ASGITransport(app=chan.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/messages",
            headers={"x-webhook-token": "secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"messages": [{"text": "queued", "user_id": "target"}]}


@pytest.mark.asyncio
async def test_http_webhook_record_open_creates_open_event(fresh_db):
    import httpx
    from channels.http_channel import HTTPChannel

    handled = asyncio.Event()

    async def _handler(_msg):
        handled.set()
        return "ok"

    chan = HTTPChannel(
        token="secret",
        message_handler=_handler,
        event_recorder=fresh_db.create_event,
    )

    transport = httpx.ASGITransport(app=chan.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/hooks",
            json={"message": "需要收盘后复查 AAPL 20MA", "record": "open"},
            headers={"x-webhook-token": "secret"},
        )

    assert response.status_code == 202
    await asyncio.wait_for(handled.wait(), timeout=1)

    event = fresh_db.get_events(limit=1)[0]
    assert event.type == "note"
    assert event.status == "open"
    assert event.content == "需要收盘后复查 AAPL 20MA"
    assert event.summary == "需要收盘后复查 AAPL 20MA"


@pytest.mark.asyncio
async def test_http_webhook_record_false_skips_event(fresh_db):
    import httpx
    from channels.http_channel import HTTPChannel

    handled = asyncio.Event()

    async def _handler(_msg):
        handled.set()
        return "ok"

    chan = HTTPChannel(
        token="secret",
        message_handler=_handler,
        event_recorder=fresh_db.create_event,
    )

    transport = httpx.ASGITransport(app=chan.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/hooks",
            json={"message": "普通 webhook 输入", "record": False},
            headers={"x-webhook-token": "secret"},
        )

    assert response.status_code == 202
    await asyncio.wait_for(handled.wait(), timeout=1)
    assert fresh_db.get_events(limit=1) == []


@pytest.mark.asyncio
async def test_http_webhook_rejects_invalid_record(fresh_db):
    import httpx
    from channels.http_channel import HTTPChannel

    handled = False

    async def _handler(_msg):
        nonlocal handled
        handled = True
        return "ok"

    chan = HTTPChannel(
        token="secret",
        message_handler=_handler,
        event_recorder=fresh_db.create_event,
    )

    transport = httpx.ASGITransport(app=chan.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/hooks",
            json={"message": "bad", "record": "resolved"},
            headers={"x-webhook-token": "secret"},
        )

    assert response.status_code == 422
    assert handled is False
    assert fresh_db.get_events(limit=1) == []


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
    assert usage["last_request_input_tokens"] == 13
    assert usage["origin"] == "cli"
    assert "at" in usage


@pytest.mark.asyncio
async def test_last_usage_is_scoped_to_conversation(fresh_db, fake_agent_run):
    await runtime.run_agent(
        "alpha", db=fresh_db, origin_channel="telegram", conversation_id="chat-a"
    )
    await runtime.run_agent(
        "beta", db=fresh_db, origin_channel="telegram", conversation_id="chat-b"
    )

    key_a = runtime.build_session_key(
        channel="telegram", conversation_id="chat-a"
    )
    key_b = runtime.build_session_key(
        channel="telegram", conversation_id="chat-b"
    )
    assert runtime.get_last_usage(key_a)["origin"] == "telegram"
    assert runtime.get_last_usage(key_b)["origin"] == "telegram"
    assert key_a != key_b


@pytest.mark.asyncio
async def test_run_agent_serializes_same_session(monkeypatch):
    active = 0
    max_active = 0

    async def fake_unlocked(*args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return "ok"

    monkeypatch.setattr(runtime, "_run_agent_unlocked", fake_unlocked)
    db = object()
    await asyncio.gather(
        runtime.run_agent("one", db=db, conversation_id="same"),
        runtime.run_agent("two", db=db, conversation_id="same"),
    )
    assert max_active == 1


@pytest.mark.asyncio
async def test_manual_compaction_uses_same_session_lock(monkeypatch):
    from channel_commands import CommandScope, compact_conversation
    from compaction import CompactCommandResult
    from session_locks import get_session_lock
    import compaction

    scope = CommandScope(channel="telegram", user_id="u", conversation_id="chat")
    called = asyncio.Event()

    async def fake_force_compact(*args, **kwargs):
        called.set()
        return CompactCommandResult(status="not_needed")

    monkeypatch.setattr(compaction, "force_compact", fake_force_compact)
    async with get_session_lock(scope.session_key()):
        task = asyncio.create_task(compact_conversation(scope=scope, lang="en"))
        await asyncio.sleep(0)
        assert not called.is_set()
    assert await task == "ℹ️ Recent history is already within budget; no compaction needed"


def test_single_run_request_guard_truncates_oversized_tool_output(monkeypatch):
    cfg = runtime.get_config()
    monkeypatch.setattr(cfg.history, "context_tokens", 10000)
    monkeypatch.setattr(cfg.llm, "max_tokens", 1000)
    monkeypatch.setattr(cfg.history, "compact_tool_output_max_chars", 1000)

    @dataclass
    class RequestContext:
        messages: list

    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="large_result",
                    tool_call_id="call-1",
                    content="x" * 100000,
                )
            ]
        )
    ]
    request = RequestContext(messages=messages)
    ctx = SimpleNamespace(deps=SimpleNamespace(session_key="guard-test"))

    guarded = runtime._guard_model_request_context(ctx, request)

    assert guarded is not request
    assert len(guarded.messages[0].parts[0].content) < 100000


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


def test_usage_payload_records_per_request_cache_metrics():
    """Run totals retain a chronological cache breakdown for each provider request."""
    class TotalUsage:
        input_tokens = 300
        output_tokens = 20
        requests = 2
        cache_read_tokens = 0
        cache_write_tokens = 0
        details = {"prompt_cache_hit_tokens": 170, "prompt_cache_miss_tokens": 130}

    class FirstUsage:
        input_tokens = 120
        cache_read_tokens = 0
        cache_write_tokens = 0
        details = {"prompt_cache_hit_tokens": 20, "prompt_cache_miss_tokens": 100}

    class FollowUsage:
        input_tokens = 180
        cache_read_tokens = 0
        cache_write_tokens = 0
        details = {"prompt_cache_hit_tokens": 150, "prompt_cache_miss_tokens": 30}

    payload = runtime._build_usage_payload(
        TotalUsage(),
        origin_channel="test",
        at="now",
        last_request_usage=FollowUsage(),
        request_usages=[FirstUsage(), FollowUsage()],
    )

    assert payload["last_request_input_tokens"] == 180
    assert payload["request_cache_metrics"] == [
        {
            "request": 1,
            "input_tokens": 120,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_tokens": 20,
            "cache_miss_tokens": 100,
            "cache_hit_ratio": 1 / 6,
        },
        {
            "request": 2,
            "input_tokens": 180,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_tokens": 150,
            "cache_miss_tokens": 30,
            "cache_hit_ratio": 5 / 6,
        },
    ]


def test_telegram_status_surfaces_cache_metrics():
    """Telegram /status should show recent cache usage when available."""
    import inspect
    from channel_commands import build_status_text

    source = inspect.getsource(build_status_text)
    assert "command.status" in source
    assert "command.status.cache_metrics" in source
    assert "cache_hit_tokens" in source
    assert "cache_write_tokens" in source


def test_status_and_compact_business_logic_is_channel_independent():
    """Channel adapters may extract identity and send, but must not own command logic."""
    import inspect
    from channels.feishu_channel import FeishuChannel
    from channels.telegram_channel import TelegramChannel

    status_sources = (
        inspect.getsource(TelegramChannel._status_command),
        inspect.getsource(FeishuChannel._status_command),
    )
    compact_sources = (
        inspect.getsource(TelegramChannel._compact_command),
        inspect.getsource(FeishuChannel._compact_command),
    )

    assert all("build_status_text" in source for source in status_sources)
    assert all("get_last_usage" not in source for source in status_sources)
    assert all("load_pydantic_messages" not in source for source in status_sources)
    assert all("compact_conversation" in source for source in compact_sources)
    assert all("force_compact" not in source for source in compact_sources)


def test_status_uses_current_session_custom_memory_and_last_request_usage(
    fresh_db, tmp_path, monkeypatch
):
    from channel_commands import CommandScope, build_status_text
    from config import get_config
    import memory

    alpha = runtime.build_session_key(
        channel="telegram", user_id="a", conversation_id="chat-a"
    )
    beta = runtime.build_session_key(
        channel="telegram", user_id="b", conversation_id="chat-b"
    )
    fresh_db.save_pydantic_messages(
        [ModelRequest(parts=[UserPromptPart(content="alpha")])], session_key=alpha
    )
    fresh_db.save_pydantic_messages(
        [ModelRequest(parts=[UserPromptPart(content="beta")])], session_key=beta
    )
    monkeypatch.setattr(memory, "_db_instance", fresh_db)
    fresh_db.create_agent_event(
        event_type="run_finished",
        origin="telegram",
        payload={
            "usage": {
                "cache_hit_tokens": 60,
                "cache_miss_tokens": 40,
            }
        },
    )
    fresh_db.create_agent_event(
        event_type="run_finished",
        origin="http",
        payload={
            "usage": {
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 100,
            }
        },
    )

    memory_file = tmp_path / "custom-memory.md"
    memory_file.write_text("custom memory", encoding="utf-8")
    monkeypatch.setattr(get_config().memory, "path", str(memory_file))
    monkeypatch.setattr(
        runtime,
        "_last_usage_by_session",
        {alpha: {
            "input_tokens": 150,
            "last_request_input_tokens": 30,
            "requests": 5,
            "cache_hit_tokens": 100,
            "cache_miss_tokens": 50,
            "cache_read_tokens": 100,
            "cache_write_tokens": 0,
            "cache_hit_ratio": 2 / 3,
            "request_cache_metrics": [
                {
                    "request": 1,
                    "cache_hit_tokens": 80,
                    "cache_miss_tokens": 40,
                },
                {
                    "request": 2,
                    "cache_hit_tokens": 20,
                    "cache_miss_tokens": 10,
                },
            ],
            "at": "now",
            "origin": "telegram",
        }},
    )

    text = build_status_text(
        scope=CommandScope(
            channel="telegram", user_id="a", conversation_id="chat-a"
        ),
        lang="en",
        peer_channel_names=["telegram", "http"],
    )

    assert "2 / 1" in text
    assert "30 /" in text
    assert "150 /" not in text
    assert f"{memory_file.stat().st_size} bytes" in text
    assert "first request: hit `80`, miss `40`" in text
    assert "follow-ups (1): hit `20`, miss `10`" in text
    assert "24h all `30.0%`" in text
    assert "`telegram` `60.0%`" in text


@pytest.mark.asyncio
async def test_compact_command_returns_localized_structured_result(
    fresh_db, monkeypatch
):
    from channel_commands import CommandScope, compact_conversation
    from compaction import CompactCommandResult
    import compaction
    import memory

    monkeypatch.setattr(memory, "_db_instance", fresh_db)

    async def fake_force_compact(db, session_key=None):
        return CompactCommandResult(status="failed", error="boom")

    monkeypatch.setattr(compaction, "force_compact", fake_force_compact)
    text = await compact_conversation(
        scope=CommandScope(
            channel="feishu",
            user_id="sender",
            conversation_id="chat",
            thread_id="topic-7",
        ),
        lang="zh",
    )

    assert text == "❌ 压缩失败：boom"
    assert "完成" not in text


@pytest.mark.asyncio
async def test_feishu_commands_preserve_topic_scope(monkeypatch):
    from channels.feishu_channel import FeishuChannel

    async def handler(_message):
        return "ok"

    channel = FeishuChannel(
        app_id="id",
        app_secret="secret",
        default_chat_id="chat",
        message_handler=handler,
    )
    captured = {}

    async def fake_compact(chat_id, scope):
        captured["chat_id"] = chat_id
        captured["scope"] = scope

    monkeypatch.setattr(channel, "_compact_command", fake_compact)
    conversation = type("Conversation", (), {"thread_id": "topic-7"})()
    msg = type("Message", (), {"conversation": conversation})()

    await channel._handle_command("compact", msg, "chat", "sender")

    assert captured["chat_id"] == "chat"
    assert captured["scope"].thread_id == "topic-7"
    assert captured["scope"].session_key().endswith(":thread:topic-7")


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
