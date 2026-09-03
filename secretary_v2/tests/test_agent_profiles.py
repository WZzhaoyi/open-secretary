"""Regression tests for bounded webhook and scheduled agent profiles."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai.models.test import TestModel

import main
import runtime
from agent_profiles import (
    INTERACTIVE_PROFILE,
    SCHEDULED_MAINTENANCE_PROFILE,
    SCHEDULED_NOTIFICATION_PROFILE,
    WEBHOOK_PROFILE,
    ScheduledNotificationOutput,
    WebhookAgentOutput,
    WebhookEventAction,
)
from channels.base import IncomingMessage
from memory import Database


class _Usage:
    input_tokens = 20
    output_tokens = 5
    requests = 1
    cache_read_tokens = 0
    cache_write_tokens = 0
    details = {}


class _StructuredResult:
    def __init__(self, prompt, output):
        self.prompt = prompt
        self.output = output

    def new_messages(self):
        return [
            ModelRequest(parts=[UserPromptPart(content=self.prompt)]),
            ModelResponse(parts=[TextPart(content=self.output.model_dump_json())]),
        ]

    def all_messages(self):
        return [
            ModelRequest(parts=[UserPromptPart(content=self.prompt)]),
            ModelResponse(
                parts=[TextPart(content=self.output.model_dump_json())],
                usage=RequestUsage(input_tokens=20, output_tokens=5),
            ),
        ]

    def usage(self):
        return _Usage()


@pytest.fixture
def profile_db(tmp_path):
    return Database(db_path=str(tmp_path / "profiles.db"))


def test_profile_tool_allowlists_are_deterministic_and_fail_closed():
    tool_defs = [
        SimpleNamespace(name=name)
        for name in (
            "db_query",
            "db_execute",
            "memory_view",
            "memory_str_replace",
            "memory_insert",
            "send_message",
            "shell",
        )
    ]

    def visible(profile):
        ctx = SimpleNamespace(deps=SimpleNamespace(agent_profile=profile))
        return {tool.name for tool in runtime._prepare_profile_tools(ctx, tool_defs)}

    assert visible(INTERACTIVE_PROFILE) == {tool.name for tool in tool_defs}
    assert visible(WEBHOOK_PROFILE) == set()
    assert visible(SCHEDULED_NOTIFICATION_PROFILE) == set()
    assert visible(SCHEDULED_MAINTENANCE_PROFILE) == {
        "db_query",
        "db_execute",
        "memory_view",
        "memory_str_replace",
        "memory_insert",
    }
    assert visible("unknown_profile") == set()


@pytest.mark.asyncio
async def test_webhook_profile_is_one_model_request_with_no_function_tools(profile_db):
    model = TestModel()

    with runtime.agent.override(model=model):
        await runtime.run_webhook_agent(
            "bounded ingest",
            profile_db,
            user_id="webhook_user",
            conversation_id="webhook_user",
            message_metadata={"record": "logged"},
        )

    assert model.last_model_request_parameters.function_tools == []
    assert [tool.name for tool in model.last_model_request_parameters.output_tools] == [
        "final_result"
    ]
    finished = next(
        event for event in profile_db.get_agent_events() if event.type == "run_finished"
    )
    payload = json.loads(finished.payload_json)
    assert payload["agent_profile"] == WEBHOOK_PROFILE
    assert payload["usage"]["requests"] == 1
    assert profile_db.load_pydantic_messages() == []


@pytest.mark.asyncio
async def test_webhook_profile_applies_summary_and_actions_without_tool_loop(
    profile_db, monkeypatch
):
    recorded = profile_db.create_event(
        "note",
        "raw market webhook",
        status="logged",
        source_channel="http",
    )
    captured = {}
    output = WebhookAgentOutput(
        reply="concise reply",
        event_summary="market signal weakened",
        events=[
            WebhookEventAction(
                event_type="check",
                status="open",
                summary="review after close",
                content="Review the signal after close",
            )
        ],
    )

    async def fake_run(prompt, *args, **kwargs):
        captured.update(kwargs)
        return _StructuredResult(prompt, output)

    monkeypatch.setattr(runtime.agent, "run", fake_run)

    reply = await runtime.run_webhook_agent(
        "raw market webhook",
        profile_db,
        user_id="webhook_user",
        conversation_id="webhook_user",
        message_metadata={
            "record": "logged",
            "recorded_event_id": recorded.id,
            "summary_supplied": False,
            "webhook_run_id": "hook_test",
        },
    )

    assert reply == "concise reply"
    assert captured["output_type"] is WebhookAgentOutput
    assert captured["deps"].agent_profile == WEBHOOK_PROFILE
    assert profile_db.load_pydantic_messages() == []
    rows = profile_db.execute_query(
        "SELECT type,status,summary,content FROM events ORDER BY id"
    )
    assert rows == [
        {
            "type": "note",
            "status": "logged",
            "summary": "market signal weakened",
            "content": "raw market webhook",
        },
        {
            "type": "check",
            "status": "open",
            "summary": "review after close",
            "content": "Review the signal after close",
        },
    ]
    assert {
        event.type for event in profile_db.get_agent_events()
    } >= {"webhook_summary_updated", "webhook_actions_applied", "run_finished"}


@pytest.mark.asyncio
async def test_scheduled_notification_resolves_only_prefetched_candidates(
    profile_db, monkeypatch
):
    candidate = profile_db.create_event(
        "remind",
        "reply needed",
        status="open",
        source_channel="scheduled",
    )
    unrelated = profile_db.create_event(
        "check",
        "manual check",
        status="open",
        source_channel="telegram",
    )
    profile_db.create_event(
        "response",
        "user already answered",
        status="logged",
        source_channel="telegram",
    )
    captured = {}
    output = ScheduledNotificationOutput(
        should_notify=False,
        message="",
        resolve_event_ids=[candidate.id, unrelated.id, 999999],
    )

    async def fake_run(prompt, *args, **kwargs):
        captured.update(kwargs)
        return _StructuredResult(prompt, output)

    monkeypatch.setattr(runtime.agent, "run", fake_run)

    decision = await runtime.run_scheduled_notification_agent(
        "check pending responses",
        profile_db,
        task_id="pending_response_check",
    )

    assert decision.resolve_event_ids == [candidate.id]
    assert captured["output_type"] is ScheduledNotificationOutput
    assert captured["deps"].agent_profile == SCHEDULED_NOTIFICATION_PROFILE
    assert "Prefetched Scheduled Data" in str(captured["message_history"])
    statuses = {
        row["id"]: row["status"]
        for row in profile_db.execute_query("SELECT id,status FROM events")
    }
    assert statuses[candidate.id] == "resolved"
    assert statuses[unrelated.id] == "open"
    assert profile_db.load_pydantic_messages() == []


@pytest.mark.asyncio
async def test_scheduled_handler_delivers_structured_notification_once(
    profile_db, monkeypatch
):
    sent = []

    class _Channel:
        async def send(self, text, user_id=None):
            sent.append((text, user_id))

    async def fake_notification_run(*args, **kwargs):
        assert kwargs["task_id"] == "review_reminder"
        return ScheduledNotificationOutput(
            should_notify=True,
            message="review now",
        )

    monkeypatch.setattr(main, "run_scheduled_notification_agent", fake_notification_run)
    app = main.SecretaryApp.__new__(main.SecretaryApp)
    app.db = profile_db
    app.config = SimpleNamespace(
        channels=SimpleNamespace(default_outgoing="telegram")
    )
    app.channels = {"telegram": _Channel()}
    app.scheduler = None
    app.subagent_registry = None
    app._collect_skill_content = lambda _text: ""

    response = await app._handle_scheduled_message(
        IncomingMessage(
            text="review prompt",
            channel="scheduled",
            user_id="scheduler",
            metadata={"task_id": "review_reminder"},
        )
    )

    assert response == "NO_ACTION"
    assert sent == [("review now", None)]
    audit = next(
        event for event in profile_db.get_agent_events() if event.type == "send_message"
    )
    assert audit.origin == "scheduled"


def test_resolve_open_events_is_exact_and_idempotent(profile_db):
    first = profile_db.create_event("check", "first", status="open")
    second = profile_db.create_event("check", "second", status="logged")

    assert profile_db.resolve_open_events([first.id, second.id, first.id]) == 1
    assert profile_db.resolve_open_events([first.id]) == 0
