import asyncio
import json

import pytest

from config import (
    AgentSubagentConfig,
    ClaudeSubagentConfig,
    CodexSubagentConfig,
    SubagentConfig,
    get_config,
    reset_config,
)
from memory import Database
from subagent_runs import (
    SubAgentRunManager,
    SubAgentStage,
    _completion_summary,
    load_subagent_definition,
    parse_subagent_shortcut,
)
from subagents import (
    SubAgentResult,
    SubAgentRunner,
    _bash_patterns_from_tools,
    _command_allowed_by_patterns,
    extract_subagent_text,
)


class FakeRunner:
    def choose_engine(self, requested=None):
        return requested or "claude"

    async def run(self, engine, prompt, cwd=None, timeout=1800):
        await asyncio.sleep(0)
        return SubAgentResult(
            engine=engine,
            prompt=prompt,
            command=[engine],
            cwd=".",
            exit_code=0,
            stdout=f"{engine} completed: {prompt[:40]}",
            stderr="",
        )


def test_subagent_command_templates(monkeypatch):
    reset_config()
    cfg = get_config()
    cfg.subagent = SubagentConfig()
    monkeypatch.setattr("subagents.get_config", lambda: cfg)
    runner = SubAgentRunner()

    codex = runner.build_command(
        "codex", "研究 AI 产业链", output_last_message="/tmp/final.txt"
    )
    assert codex[0] == "codex"
    assert "--search" in codex
    assert "--ask-for-approval" in codex
    assert codex[codex.index("--ask-for-approval") + 1] == "never"
    assert "exec" in codex
    assert "--sandbox" in codex
    assert codex[codex.index("--sandbox") + 1] == "read-only"
    assert "--output-last-message" in codex
    assert codex[codex.index("--output-last-message") + 1] == "/tmp/final.txt"
    assert codex[-1] == "研究 AI 产业链"

    claude = runner.build_command("claude", "研究 AI 产业链")
    assert claude[:2] == ["claude", "-p"]
    assert "--allowedTools" in claude
    allowed_tools = claude[claude.index("--allowedTools") + 1]
    assert "WebSearch,WebFetch,Read" in allowed_tools
    assert "Bash(" not in allowed_tools
    assert "--model" in claude
    assert "sonnet" in claude
    assert "--effort" in claude
    assert "high" in claude

    agent = runner.build_command("agent", "研究 AI 产业链")
    assert agent[:3] == ["internal-agent", "--provider", cfg.llm.provider]
    assert "--model" in agent
    assert cfg.llm.model in agent
    assert "--allowedTools" in agent
    assert "Bash(opencli gemini *)" in agent[agent.index("--allowedTools") + 1]
    cfg.llm.effort = "max"
    agent_with_effort = runner.build_command("agent", "研究 AI 产业链")
    assert "--effort" in agent_with_effort
    assert agent_with_effort[agent_with_effort.index("--effort") + 1] == "max"
    reset_config()


def test_subagent_command_templates_respect_research_config(monkeypatch):
    reset_config()
    cfg = get_config()
    cfg.subagent = SubagentConfig(
        default_engine="codex",
        codex=CodexSubagentConfig(
            model="gpt-5.5",
            enable_search=False,
            sandbox="workspace-write",
            approval_policy="on-request",
            config_overrides=["sandbox_workspace_write.network_access=true"],
        ),
        claude=ClaudeSubagentConfig(
            model="opus",
            effort="xhigh",
            allowed_bash=["longbridge quote *"],
            disallowed_tools=["Bash(longbridge order *)"],
        ),
    )
    monkeypatch.setattr("subagents.get_config", lambda: cfg)

    runner = SubAgentRunner()
    codex = runner.build_command("codex", "研究 AI 产业链")
    assert "--search" not in codex
    assert codex[codex.index("-c") + 1] == "sandbox_workspace_write.network_access=true"
    assert codex[codex.index("--ask-for-approval") + 1] == "on-request"
    assert codex[codex.index("--sandbox") + 1] == "workspace-write"
    assert codex[codex.index("--model") + 1] == "gpt-5.5"

    claude = runner.build_command("claude", "研究 AI 产业链")
    allowed_tools = claude[claude.index("--allowedTools") + 1]
    assert "Bash(longbridge quote *)" in allowed_tools
    assert claude[claude.index("--disallowedTools") + 1] == "Bash(longbridge order *)"
    assert claude[claude.index("--model") + 1] == "opus"
    assert claude[claude.index("--effort") + 1] == "xhigh"
    reset_config()


def test_subagent_choose_engine_falls_back_to_internal_agent(monkeypatch):
    reset_config()
    cfg = get_config()
    cfg.subagent = SubagentConfig(default_engine="claude", fallback_engine="agent")
    monkeypatch.setattr("subagents.get_config", lambda: cfg)
    monkeypatch.setattr("subagents.shutil.which", lambda _engine: None)

    runner = SubAgentRunner()
    assert runner.is_available("claude") is False
    assert runner.is_available("codex") is False
    assert runner.is_available("agent") is True
    assert runner.choose_engine() == "agent"
    reset_config()


def test_subagent_choose_engine_can_disable_internal_fallback(monkeypatch):
    reset_config()
    cfg = get_config()
    cfg.subagent = SubagentConfig(
        default_engine="claude",
        fallback_engine="agent",
        agent=AgentSubagentConfig(enabled=False),
    )
    monkeypatch.setattr("subagents.get_config", lambda: cfg)
    monkeypatch.setattr("subagents.shutil.which", lambda _engine: None)

    runner = SubAgentRunner()
    with pytest.raises(RuntimeError, match="No subagent engine is available"):
        runner.choose_engine()
    reset_config()


def test_subagent_explicit_missing_cli_does_not_fall_back(monkeypatch):
    reset_config()
    cfg = get_config()
    cfg.subagent = SubagentConfig(fallback_engine="agent")
    monkeypatch.setattr("subagents.get_config", lambda: cfg)
    monkeypatch.setattr("subagents.shutil.which", lambda _engine: None)

    runner = SubAgentRunner()
    with pytest.raises(RuntimeError, match="codex CLI is not installed"):
        runner.choose_engine("codex")
    assert runner.choose_engine("agent") == "agent"
    reset_config()


@pytest.mark.asyncio
async def test_internal_agent_subagent_run_is_isolated(monkeypatch, tmp_path):
    reset_config()
    cfg = get_config()
    cfg.subagent = SubagentConfig(default_engine="agent", fallback_engine="agent")
    monkeypatch.setattr("subagents.get_config", lambda: cfg)
    monkeypatch.setattr("subagents._build_internal_model", lambda: object())

    calls = []

    class FakeAgent:
        def __init__(self, model, system_prompt):
            calls.append(("init", model, system_prompt))

        def tool_plain(self, *args, **kwargs):
            def decorator(func):
                calls.append(("tool", kwargs.get("name")))
                return func

            return decorator

        async def run(self, prompt):
            calls.append(("run", prompt))

            class Result:
                output = "isolated stage output"

            return Result()

    monkeypatch.setattr("subagents.Agent", FakeAgent)

    result = await SubAgentRunner(base_dir=tmp_path).run(
        "agent",
        "stage prompt",
        timeout=1,
    )

    assert result.ok
    assert result.engine == "agent"
    assert result.command[0] == "internal-agent"
    assert result.stdout == "isolated stage output"
    assert calls[0][0] == "init"
    assert calls[0][2].startswith(cfg.subagent.agent.system_prompt)
    assert ("tool", "bash") in calls
    assert calls[-1] == ("run", "stage prompt")
    reset_config()


def test_internal_agent_bash_allowlist_patterns():
    assert _bash_patterns_from_tools(["WebSearch", "Bash(opencli gemini *)"]) == [
        "opencli gemini *"
    ]
    assert _command_allowed_by_patterns("opencli list -f yaml", ["opencli list*"])
    assert _command_allowed_by_patterns("opencli gemini search AI", ["opencli gemini *"])
    assert not _command_allowed_by_patterns("opencli external docker ps", ["opencli gemini *"])


@pytest.mark.asyncio
async def test_internal_agent_bash_denies_unlisted_command(tmp_path):
    result = await SubAgentRunner(base_dir=tmp_path)._run_internal_bash(
        command="echo hello",
        work_dir=tmp_path,
        timeout=1,
        allowed_patterns=["opencli *"],
    )

    assert "PERMISSION_DENIED" in result
    assert "command_not_allowlisted" in result


@pytest.mark.asyncio
async def test_internal_agent_bash_still_uses_shell_guardrails(tmp_path):
    result = await SubAgentRunner(base_dir=tmp_path)._run_internal_bash(
        command="sudo ls",
        work_dir=tmp_path,
        timeout=1,
        allowed_patterns=["sudo *"],
    )

    assert "PERMISSION_DENIED" in result
    assert "hard_deny_command" in result


@pytest.mark.asyncio
async def test_internal_agent_bash_respects_disallowed_tools(tmp_path):
    result = await SubAgentRunner(base_dir=tmp_path)._run_internal_bash(
        command="opencli external docker ps",
        work_dir=tmp_path,
        timeout=1,
        allowed_patterns=["opencli *"],
        disallowed_patterns=["opencli external *"],
    )

    assert "PERMISSION_DENIED" in result
    assert "command_disallowed" in result
    assert "subagent.agent.disallowed_tools" in result


def test_extract_subagent_text_from_claude_json():
    stdout = (
        '{"type":"result","result":"# 报告\\n\\n正文",'
        '"session_id":"abc","usage":{"input_tokens":1}}'
    )
    assert extract_subagent_text("claude", stdout) == "# 报告\n\n正文"


def test_extract_subagent_text_from_codex_jsonl():
    stdout = "\n".join(
        [
            '{"type":"usage","tokens":10}',
            '{"type":"message","role":"assistant","content":"初稿"}',
            '{"type":"final_output","message":"最终报告"}',
        ]
    )
    assert extract_subagent_text("codex", stdout) == "最终报告"


def test_summary_text_prefers_clean_result_over_machine_metadata():
    result = SubAgentResult(
        engine="claude",
        prompt="p",
        command=["claude"],
        cwd=".",
        exit_code=0,
        stdout='{"result":"干净报告","session_id":"leak"}',
        stderr="",
    )
    assert result.summary_text() == "干净报告"


def test_deep_research_definition_loads_stage_prompts():
    definition = load_subagent_definition("deep_research")

    assert definition.name == "deep_research"
    assert definition.kind == "research"
    assert definition.id_prefix == "research"
    assert definition.artifact_dir == "research"
    assert definition.main_stage == "report"
    assert definition.default_engine == "claude"
    assert "交易机会和行业分析研究员" in definition.base_contract
    assert definition.stages == ["scout", "bull_case", "bear_case", "report"]
    assert "阶段：scout" in definition.prompt_template("scout")
    assert set(definition.prompt_templates) == {
        "scout",
        "bull_case",
        "bear_case",
        "report",
    }
    assert "{{topic}}" in definition.prompt_template("report")


@pytest.mark.asyncio
async def test_subagent_run_manager_runs_multistage_job(test_db, tmp_path):
    notifications = []

    async def notify(channel, text, user_id=None, artifact_path=None):
        notifications.append((channel, text, user_id, artifact_path))

    manager = SubAgentRunManager(
        db=test_db,
        runner=FakeRunner(),
        notifier=notify,
        artifact_dir=tmp_path,
        agent_name="deep_research",
    )

    job_id = manager.start(
        input_payload={"topic": "机器人行业机会"},
        subject="机器人行业机会",
        engine="codex",
    )
    task = manager._tasks[job_id]
    await task

    job = test_db.get_subagent_run(job_id)
    assert job.status == "succeeded"
    assert job.agent_name == "deep_research"
    assert job.agent_kind == "research"
    assert job.engine == "codex"
    assert job.input_payload == {"topic": "机器人行业机会"}
    assert job.subject == "机器人行业机会"
    assert job.artifact_path
    assert "codex completed" in job.result
    assert (tmp_path / f"{job_id}.md").exists()
    assert notifications[-1][0] == "cli"
    assert notifications[-1][3] == job.artifact_path
    stage_rows = json.loads(job.stages_json)
    assert [stage["name"] for stage in stage_rows] == manager.definition.stages


@pytest.mark.asyncio
async def test_subagent_run_manager_supports_legacy_notifier_signature(test_db, tmp_path):
    notifications = []

    async def notify(channel, text, user_id=None):
        notifications.append((channel, text, user_id))

    manager = SubAgentRunManager(
        db=test_db,
        runner=FakeRunner(),
        notifier=notify,
        artifact_dir=tmp_path,
        agent_name="deep_research",
    )

    await manager._notify("cli", "cli_user", "done", artifact_path=str(tmp_path / "report.md"))

    assert notifications == [("cli", "done", "cli_user")]


@pytest.mark.asyncio
async def test_subagent_run_manager_resumes_incomplete_jobs(test_db, tmp_path):
    test_db.create_subagent_run(
        run_id="research_resume",
        agent_name="deep_research",
        agent_kind="research",
        engine="claude",
        input_payload={"topic": "A股石油股预期差"},
        subject="A股石油股预期差",
        origin_channel="cli",
        user_id="cli_user",
    )
    test_db.update_subagent_run("research_resume", status="running")
    manager = SubAgentRunManager(
        db=test_db,
        runner=FakeRunner(),
        artifact_dir=tmp_path,
        agent_name="deep_research",
    )

    resumed = manager.resume_incomplete()
    assert resumed == ["research_resume"]
    assert "research_resume" in manager._tasks
    await manager._tasks["research_resume"]

    job = test_db.get_subagent_run("research_resume")
    assert job.status == "succeeded"
    assert (tmp_path / "research_resume.md").exists()


def test_subagent_run_database_roundtrip(test_db):
    job = test_db.create_subagent_run(
        run_id="research_test",
        agent_name="deep_research",
        agent_kind="research",
        engine="claude",
        input_payload={"topic": "港股创新药行业"},
        subject="港股创新药行业",
        origin_channel="cli",
        user_id="cli_user",
    )
    assert job.id == "research_test"
    assert job.agent_name == "deep_research"
    assert job.subject == "港股创新药行业"

    test_db.update_subagent_run("research_test", status="running")
    fetched = test_db.get_subagent_run("research_test")
    assert fetched.status == "running"
    assert test_db.list_subagent_runs(agent_name="deep_research", limit=1)[0].id == "research_test"


def test_research_done_message_uses_brief_not_tail(test_db, tmp_path):
    test_db.create_subagent_run(
        run_id="research_brief",
        agent_name="deep_research",
        agent_kind="research",
        engine="claude",
        input_payload={"topic": "全球铝供需"},
        subject="全球铝供需",
        origin_channel="cli",
        user_id="cli_user",
    )
    report = (
        "一、 一句话结论\n"
        "铝价的预期差主要来自中东供给风险与中国库存压力的拉扯。\n\n"
        "二、 交易/行业假设\n"
        "正方看供给冲击，反方看已 price-in 和中国增供。\n\n"
        + ("正文\n" * 200)
        + "| S15 | 来源表 | 免责声明附近内容 |\n"
    )
    artifact = tmp_path / "research_brief.md"
    artifact.write_text(report, encoding="utf-8")
    test_db.update_subagent_run(
        "research_brief",
        status="succeeded",
        result=report,
        artifact_path=str(artifact),
    )
    manager = SubAgentRunManager(db=test_db, runner=FakeRunner(), artifact_dir=tmp_path, agent_name="deep_research")

    message = manager._done_message("research_brief")
    assert "Summary:" in message
    assert "铝价的预期差主要来自" in message
    assert "| S15 |" not in message
    assert "Artifact:" in message


def test_research_brief_prefers_explicit_brief_section():
    report = (
        "# Research research_x\n\n"
        "## 主报告\n"
        "## 简报\n"
        "- 结论：只保留真正适合通知的内容。\n"
        "- 下一步跟踪：验证价格和库存。\n\n"
        "一、 一句话结论\n"
        "这里是较长正文，不应进入简报。\n"
    )

    brief = _completion_summary(report)
    assert "只保留真正适合通知的内容" in brief
    assert "较长正文" not in brief


def test_research_artifact_puts_main_report_before_appendix(test_db, tmp_path):
    manager = SubAgentRunManager(db=test_db, runner=FakeRunner(), artifact_dir=tmp_path, agent_name="deep_research")
    stages = [
        SubAgentStage(name="scout", status="succeeded", prompt="", output="scout notes"),
        SubAgentStage(name="bull_case", status="succeeded", prompt="", output="bull notes"),
        SubAgentStage(name="bear_case", status="succeeded", prompt="", output="bear notes"),
        SubAgentStage(name="report", status="succeeded", prompt="", output="final report"),
    ]

    test_db.create_subagent_run(
        run_id="research_artifact",
        agent_name="deep_research",
        agent_kind="research",
        engine="claude",
        input_payload={"topic": "铝行业"},
        subject="铝行业",
    )
    job = test_db.get_subagent_run("research_artifact")
    path = manager._write_artifact(job, stages)
    text = path.read_text(encoding="utf-8")

    assert text.index("## Result") < text.index("## Stage Outputs")
    assert text.index("final report") < text.index("### scout")
    assert "### bull_case" in text
    assert "### bear_case" in text


def test_parse_subagent_shortcut_routes_status_without_llm():
    import re

    pattern = re.compile(r"\bresearch_[0-9a-fA-F]{6,32}\b")
    list_words = ("最近研究任务", "研究任务列表", "列出研究任务", "list research")

    assert parse_subagent_shortcut("查看 research_ce4ab3afb8 的状态", pattern, list_words) == (
        "status",
        "research_ce4ab3afb8",
    )
    assert parse_subagent_shortcut("取消 research_ce4ab3afb8", pattern, list_words) == (
        "cancel",
        "research_ce4ab3afb8",
    )
    assert parse_subagent_shortcut("查看最近研究任务", pattern, list_words) == ("list", None)
    assert parse_subagent_shortcut("研究 A股石油股预期差", pattern, list_words) is None


def test_research_prompt_prefers_native_search_and_readonly_market_cli(test_db, tmp_path):
    manager = SubAgentRunManager(db=test_db, runner=FakeRunner(), artifact_dir=tmp_path, agent_name="deep_research")
    prompt = manager._render_stage_prompt(
        "scout", {"topic": "A股石油股预期差", "language": "en"}, []
    )

    assert "WebSearch/WebFetch" in prompt
    assert "当前已授权的只读 CLI" in prompt
    assert "不要调用未授权的 Bash" in prompt
    assert "交易、订单、账户、资产" in prompt
    assert "language: en" in prompt
    assert "{{language}}" not in prompt


def test_research_prompt_defaults_language_to_auto(test_db, tmp_path):
    manager = SubAgentRunManager(
        db=test_db, runner=FakeRunner(), artifact_dir=tmp_path, agent_name="deep_research"
    )
    prompt = manager._render_stage_prompt("scout", {"topic": "A股石油股预期差"}, [])

    assert "language: auto" in prompt
    assert "{{language}}" not in prompt


def test_research_prompts_use_bull_and_bear_cases(test_db, tmp_path):
    manager = SubAgentRunManager(db=test_db, runner=FakeRunner(), artifact_dir=tmp_path, agent_name="deep_research")
    scout_stage = SubAgentStage(
        name="scout", status="succeeded", prompt="", output="scout"
    )
    bull_stage = SubAgentStage(
        name="bull_case", status="succeeded", prompt="", output="bull"
    )
    scout_prompt = manager._render_stage_prompt("scout", {"topic": "A股石油股预期差"}, [])
    bull_prompt = manager._render_stage_prompt(
        "bull_case", {"topic": "A股石油股预期差"}, [scout_stage]
    )
    bear_prompt = manager._render_stage_prompt(
        "bear_case", {"topic": "A股石油股预期差"}, [scout_stage, bull_stage]
    )
    report_prompt = manager._render_stage_prompt(
        "report", {"topic": "A股石油股预期差"}, [scout_stage, bull_stage]
    )

    assert "1200-1800 字" in scout_prompt
    assert "候选假设最多 3 条" in scout_prompt
    assert "bull_case / 正方研究" in bull_prompt
    assert "存在预期差或交易机会" in bull_prompt
    assert "关键证据最多 8 条" in bull_prompt
    assert "bear_case / 反方研究、事实核验与压力测试" in bear_prompt
    assert "正方关键事实核验（最高优先级）" in bear_prompt
    assert "事实核验最多 6 条正方关键事实" in bear_prompt
    assert "最强反方论点与证据" in bear_prompt
    assert "已 price-in" in bear_prompt
    assert "最强反方证据最多 8 条" in bear_prompt
    assert "## 简报" in report_prompt
    assert "3000-5000 字" in report_prompt
    assert "正方证据最多 5 条" in report_prompt
    assert "来源表最多 10 条" in report_prompt
    assert "事实核验与压力测试" in report_prompt
