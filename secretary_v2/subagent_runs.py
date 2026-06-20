"""Subagent run orchestration built on local CLI sidecars."""

import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional
import re
import yaml

from guardrails import BASE_DIR, truncate_output
from memory import Database
from subagents import SubAgentRunner, extract_subagent_text

logger = logging.getLogger(__name__)

NotifyFn = Callable[..., Awaitable[None]]

# All new subagent runs share one prefix; the run's kind lives in the DB row
# (agent_name/agent_kind), not in the id. This keeps id-based lifecycle ops
# (status/cancel/resume) kind-agnostic. `research_` is matched only so jobs
# created before prefix unification stay addressable — it is a legacy id
# prefix, not a reference to any specific subagent type.
DEFAULT_ID_PREFIX = "run"
SUBAGENT_ID_RE = re.compile(r"\b(?:run|research)_[0-9a-fA-F]{6,32}\b")
SUBAGENT_LIST_WORDS = (
    "后台任务列表",
    "子任务列表",
    "列出后台任务",
    "列出子任务",
    "list subagents",
    "list runs",
    "最近研究任务",
    "研究任务列表",
    "列出研究任务",
    "list research",
)


def _truncate_head(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars].rstrip()
        + f"\n\n... (truncated, first {max_chars} chars) ..."
    )


def _completion_summary(text: str, max_chars: int = 2200) -> str:
    """Return a notification-sized summary from generated output."""
    text = (text or "").strip()
    if not text:
        return "Output generated, but no summary was extracted."

    first_section = _extract_first_markdown_section(text)
    if first_section:
        return _truncate_head(first_section, max_chars)

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line.startswith("... (truncated"):
            continue
        if line.startswith("# ") or line.startswith("- Agent:") or line.startswith("- Engine:"):
            continue
        if line.startswith("- Subject:") or line.startswith("- Generated at:") or line.startswith("Status:"):
            continue
        if line.startswith("## "):
            lines = []
            continue
        if re.match(r"^[一二三四五六七八九十]+[、.．]\s*", line) and lines:
            break
        if line.startswith("|"):
            break
        lines.append(line)
        candidate = "\n".join(lines).strip()
        if len(candidate) >= max_chars:
            return _truncate_head(candidate, max_chars)

    brief = "\n".join(lines).strip()
    return _truncate_head(brief or text, max_chars)


def _extract_first_markdown_section(text: str) -> str:
    heading_re = re.compile(r"^#{1,6}\s+(.+?)\s*$")
    lines: List[str] = []
    collecting = False
    for raw_line in text.splitlines():
        if heading_re.match(raw_line.strip()):
            if collecting:
                break
            collecting = True
            continue
        if collecting:
            if re.match(r"^[一二三四五六七八九十]+[、.．]\s*", raw_line.strip()):
                break
            lines.append(raw_line.rstrip())
    return "\n".join(lines).strip()


def parse_subagent_shortcut(
    text: str,
    id_pattern: re.Pattern[str],
    list_words: tuple[str, ...],
    cancel_words: tuple[str, ...] = ("取消", "停止", "终止", "cancel", "stop"),
    status_words: tuple[str, ...] = ("查看", "查询", "状态", "进度", "status", "progress"),
    resume_words: tuple[str, ...] = ("继续", "恢复", "重跑", "重试", "resume", "retry", "rerun"),
) -> Optional[tuple[str, Optional[str]]]:
    """Parse deterministic subagent run commands before the LLM sees them.

    This prevents status/cancel requests from being misread as requests to
    start a replacement background run.
    """
    normalized = text.strip().lower()
    match = id_pattern.search(normalized)
    job_id = match.group(0) if match else None

    if job_id and any(word in normalized for word in cancel_words):
        return ("cancel", job_id)
    if job_id and any(word in normalized for word in resume_words):
        return ("resume", job_id)
    if job_id and any(word in normalized for word in status_words):
        return ("status", job_id)
    if any(word in normalized for word in list_words):
        return ("list", None)
    return None


@dataclass
class SubAgentStage:
    name: str
    status: str
    prompt: str
    output: str = ""
    exit_code: Optional[int] = None
    error: Optional[str] = None


@dataclass
class SubAgentDefinition:
    name: str
    kind: str
    description: str
    required_inputs: List[str]
    artifact_dir: Optional[str]
    main_stage: Optional[str]
    default_engine: Optional[str]
    base_contract: str
    stages: List[str]
    prompt_templates: Dict[str, str]
    root_dir: Path

    def prompt_template(self, stage: str) -> str:
        if stage in self.prompt_templates:
            return self.prompt_templates[stage]
        path = self.root_dir / f"{stage}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Missing prompt template for stage '{stage}'")


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def load_subagent_definition(name: str, base_dir: Optional[Path] = None) -> SubAgentDefinition:
    """Load a lightweight subagent definition from subagent_defs/<name>/AGENT.md."""
    root = Path(base_dir or (BASE_DIR / "subagent_defs")) / name
    agent_path = root / "AGENT.md"
    if not agent_path.exists():
        raise FileNotFoundError(f"Subagent definition not found: {agent_path}")

    text = agent_path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"Subagent definition missing YAML frontmatter: {agent_path}")
    data = yaml.safe_load(match.group(1)) or {}
    stages_data = data.get("stages") or []
    prompt_templates: Dict[str, str] = {}
    if isinstance(stages_data, dict):
        stages = list(stages_data.keys())
        prompt_templates = {
            str(stage): str(template)
            for stage, template in stages_data.items()
            if isinstance(stage, str) and isinstance(template, str) and template.strip()
        }
    else:
        stages = stages_data
    if not isinstance(stages, list) or not stages or not all(
        isinstance(s, str) and s for s in stages
    ):
        raise ValueError(f"Subagent definition has invalid stages: {agent_path}")
    missing = [
        stage
        for stage in stages
        if stage not in prompt_templates and not (root / f"{stage}.md").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Subagent definition missing prompt templates {missing}: {root}"
        )
    required_inputs = [
        str(x).strip()
        for x in (data.get("required_inputs") or [])
        if str(x).strip()
    ]
    return SubAgentDefinition(
        name=str(data.get("name") or name),
        kind=str(data.get("kind") or ""),
        description=str(data.get("description") or ""),
        required_inputs=required_inputs,
        artifact_dir=data.get("artifact_dir"),
        main_stage=data.get("main_stage"),
        default_engine=data.get("default_engine"),
        base_contract=str(data.get("base_contract") or ""),
        stages=stages,
        prompt_templates=prompt_templates,
        root_dir=root,
    )


def discover_definitions(
    base_dir: Optional[Path] = None,
) -> Dict[str, SubAgentDefinition]:
    """Discover every subagent_defs/<name>/AGENT.md definition.

    Convention-based discovery (à la OpenCode's agent directory): the directory
    name is the identifier, no central registry list. A malformed definition is
    skipped with a warning rather than failing all discovery.
    """
    root = Path(base_dir or (BASE_DIR / "subagent_defs"))
    definitions: Dict[str, SubAgentDefinition] = {}
    if not root.exists():
        return definitions
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "AGENT.md").exists():
            continue
        try:
            definition = load_subagent_definition(child.name, base_dir=root)
        except Exception as e:
            logger.warning("Skipping subagent definition %s: %s", child.name, e)
            continue
        definitions[definition.name] = definition
    return definitions


def _render_template(template: str, values: Dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


class SubAgentRunManager:
    """Manage asynchronous multi-stage subagent jobs."""

    def __init__(
        self,
        db: Database,
        runner: Optional[SubAgentRunner] = None,
        notifier: Optional[NotifyFn] = None,
        artifact_dir: Optional[Path] = None,
        definition: Optional[SubAgentDefinition] = None,
        agent_name: Optional[str] = None,
    ):
        if definition is None and not agent_name:
            raise ValueError("agent_name is required when definition is not provided")
        self.db = db
        self.runner = runner or SubAgentRunner()
        self.notifier = notifier
        self.definition = definition or load_subagent_definition(agent_name)
        default_artifact_dir = BASE_DIR / "subagent_runs" / self.definition.name
        if self.definition.artifact_dir:
            default_artifact_dir = BASE_DIR / self.definition.artifact_dir
        self.artifact_dir = Path(artifact_dir or default_artifact_dir).resolve()
        self._tasks: Dict[str, asyncio.Task] = {}

    def start(
        self,
        input_payload: Dict[str, str],
        subject: Optional[str] = None,
        engine: Optional[str] = None,
        origin_channel: str = "cli",
        user_id: Optional[str] = None,
    ) -> str:
        if not input_payload:
            raise ValueError("input_payload is required")
        missing = [
            key
            for key in self.definition.required_inputs
            if not str(input_payload.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"missing required inputs for '{self.definition.name}': "
                f"{', '.join(missing)}"
            )
        subject_text = (subject or self._subject_from_payload(input_payload)).strip()
        if not subject_text:
            raise ValueError("subject is required")
        chosen = self.runner.choose_engine(engine)
        job_id = f"{DEFAULT_ID_PREFIX}_{uuid.uuid4().hex[:10]}"
        self.db.create_subagent_run(
            run_id=job_id,
            agent_name=self.definition.name,
            agent_kind=self.definition.kind,
            engine=chosen,
            input_payload=input_payload,
            subject=subject_text,
            origin_channel=origin_channel,
            user_id=user_id,
        )
        task = asyncio.create_task(
            self._run_job(job_id),
            name=f"subagent:{job_id}",
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda t, jid=job_id: self._tasks.pop(jid, None))
        return job_id

    def resume_incomplete(self, limit: int = 20) -> List[str]:
        """Restart persisted pending/running jobs after a service restart.

        In-memory asyncio tasks disappear when the process exits. We keep the
        job rows durable and simply replay incomplete jobs from the beginning.
        """
        jobs = self.db.list_subagent_runs_by_status(
            ["pending", "running"],
            agent_name=self.definition.name,
            limit=limit,
        )
        resumed: List[str] = []
        for job in jobs:
            if job.id in self._tasks:
                continue
            self.db.update_subagent_run(
                job.id,
                status="pending",
                error="resumed after service restart; previous in-memory task was lost",
            )
            task = asyncio.create_task(
                self._run_job(job.id),
                name=f"subagent:{job.id}:resumed",
            )
            self._tasks[job.id] = task
            task.add_done_callback(lambda t, jid=job.id: self._tasks.pop(jid, None))
            resumed.append(job.id)
        if resumed:
            self.db.create_event(
                "subagent",
                f"Resumed incomplete subagent runs: {', '.join(resumed)}",
            )
        return resumed

    def resume(self, job_id: str) -> bool:
        """Resume a persisted job from its first non-succeeded stage."""
        if job_id in self._tasks:
            return True
        job = self.db.get_subagent_run(job_id)
        if not job or job.agent_name != self.definition.name:
            return False
        if job.status == "succeeded":
            return False
        self.db.update_subagent_run(
            job_id,
            status="pending",
            error="resume requested; retrying from first incomplete stage",
            artifact_path=None,
            result=None,
        )
        task = asyncio.create_task(
            self._run_job(job_id),
            name=f"subagent:{job_id}:resumed",
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda t, jid=job_id: self._tasks.pop(jid, None))
        self.db.create_event("subagent", f"Resume requested for subagent run {job_id}")
        return True

    def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            self.db.update_subagent_run(job_id, status="cancelled", error="cancelled")
            return True
        job = self.db.get_subagent_run(job_id)
        if job and job.status in ("pending", "running"):
            self.db.update_subagent_run(job_id, status="cancelled", error="cancelled")
            return True
        return False

    def status_text(self, job_id: str) -> str:
        job = self.db.get_subagent_run(job_id)
        if not job:
            return f"Subagent run `{job_id}` not found"
        return self._format_job(job)

    def list_text(self, limit: int = 10) -> str:
        jobs = self.db.list_subagent_runs(
            agent_name=self.definition.name,
            limit=limit,
        )
        if not jobs:
            return "No subagent runs"
        return "\n".join(self._format_job(job, compact=True) for job in jobs)

    def _record_agent_event(
        self,
        event_type: str,
        job_id: str,
        subject: str,
        payload: Optional[dict] = None,
    ) -> None:
        try:
            self.db.create_agent_event(
                event_type,
                origin="subagent",
                run_id=job_id,
                subject=subject,
                payload=payload,
            )
        except Exception as e:
            logger.warning("Failed to record subagent event %s: %s", event_type, e)

    async def stop(self) -> None:
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_job(self, job_id: str) -> None:
        job = self.db.get_subagent_run(job_id)
        if not job:
            return

        stages = self._load_resumable_stages(job.stages_json)
        self.db.update_subagent_run(job_id, status="running")
        self.db.create_event(
            "subagent",
            f"Subagent run {job_id} started: agent={job.agent_name} engine={job.engine} subject={job.subject}",
        )

        try:
            for stage_name in self.definition.stages[len(stages):]:
                prompt = self._render_stage_prompt(stage_name, job.input_payload, stages)
                await self._run_stage(job_id, stages, stage_name, prompt)

            main_stage_name = self._main_stage_name()
            final = next(
                (stage for stage in reversed(stages) if stage.name == main_stage_name),
                stages[-1],
            )

            artifact = self._write_artifact(job, stages)
            result = final.output.strip() or self._stage_digest(stages, max_chars=12000)
            self.db.update_subagent_run(
                job_id,
                status="succeeded",
                stages_json=self._stages_json(stages),
                artifact_path=str(artifact),
                result=result,
            )
            self.db.create_event(
                "subagent",
                f"Subagent run {job_id} completed: subject={job.subject}\nartifact={artifact}",
            )
            await self._notify(
                job.origin_channel,
                job.user_id,
                self._done_message(job_id),
                artifact_path=str(artifact),
            )
        except asyncio.CancelledError:
            self.db.update_subagent_run(
                job_id,
                status="cancelled",
                stages_json=self._stages_json(stages),
                error="cancelled",
            )
            self.db.create_event("subagent", f"Subagent run {job_id} cancelled: subject={job.subject}")
            await self._notify(job.origin_channel, job.user_id, f"Subagent run `{job_id}` cancelled.")
            raise
        except Exception as e:
            logger.exception("Subagent run %s failed", job_id)
            self.db.update_subagent_run(
                job_id,
                status="failed",
                stages_json=self._stages_json(stages),
                error=f"{type(e).__name__}: {e}",
            )
            self.db.create_event(
                "subagent",
                f"Subagent run {job_id} failed: {type(e).__name__}: {e}",
            )
            await self._notify(
                job.origin_channel,
                job.user_id,
                f"Subagent run `{job_id}` failed: {type(e).__name__}: {e}",
            )

    async def _run_stage(
        self,
        job_id: str,
        stages: List[SubAgentStage],
        name: str,
        prompt: str,
    ) -> SubAgentStage:
        job = self.db.get_subagent_run(job_id)
        if not job:
            raise RuntimeError(f"job {job_id} disappeared")
        stage = SubAgentStage(name=name, status="running", prompt=prompt)
        stages.append(stage)
        self.db.update_subagent_run(job_id, stages_json=self._stages_json(stages))
        self._record_agent_event(
            "subagent_step_started",
            job_id,
            name,
            payload={
                "engine": job.engine,
                "subject": job.subject,
                "prompt_chars": len(prompt or ""),
            },
        )

        try:
            result = await self.runner.run(job.engine, prompt, timeout=1800)
        except Exception as e:
            stage.status = "failed"
            stage.error = f"{type(e).__name__}: {e}"
            self.db.update_subagent_run(job_id, stages_json=self._stages_json(stages))
            self._record_agent_event(
                "subagent_step_finished",
                job_id,
                name,
                payload={
                    "status": "failed",
                    "error": stage.error,
                },
            )
            raise
        stage.exit_code = result.exit_code
        stage.output = extract_subagent_text(job.engine, result.stdout) or result.stderr.strip()
        if result.ok:
            stage.status = "succeeded"
        else:
            stage.status = "failed"
            stage.error = (
                "sub-agent timed out"
                if result.timed_out
                else truncate_output(result.stderr or result.stdout, 2000)
            )
            self._record_agent_event(
                "subagent_step_finished",
                job_id,
                name,
                payload={
                    "status": "failed",
                    "exit_code": stage.exit_code,
                    "timed_out": result.timed_out,
                    "error": stage.error,
                },
            )
            raise RuntimeError(f"{job.engine} stage {name} failed: {stage.error}")
        self.db.update_subagent_run(job_id, stages_json=self._stages_json(stages))
        self._record_agent_event(
            "subagent_step_finished",
            job_id,
            name,
            payload={
                "status": "succeeded",
                "exit_code": stage.exit_code,
                "output_chars": len(stage.output or ""),
            },
        )
        return stage

    async def _notify(
        self,
        channel: str,
        user_id: Optional[str],
        text: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        if self.notifier is None:
            return
        try:
            if self._notifier_accepts_artifact_path():
                await self.notifier(channel, text, user_id, artifact_path)
            else:
                await self.notifier(channel, text, user_id)
        except Exception as e:
            logger.error("Failed to send subagent notification: %s", e)

    def _notifier_accepts_artifact_path(self) -> bool:
        if self.notifier is None:
            return False
        try:
            params = inspect.signature(self.notifier).parameters
        except (TypeError, ValueError):
            return True
        if "artifact_path" in params:
            return True
        values = list(params.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in values):
            return True
        positional = [
            p
            for p in values
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 4

    def _write_artifact(self, job, stages: List[SubAgentStage]) -> Path:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / f"{job.id}.md"
        sections = [
            f"# Subagent Run {job.id}",
            f"- Agent: `{job.agent_name}`",
            f"- Engine: `{job.engine}`",
            f"- Subject: {job.subject or ''}",
            f"- Generated at: {datetime.now().isoformat()}",
            "",
        ]

        main_stage_name = self._main_stage_name()
        main_stage = next((s for s in stages if s.name == main_stage_name), None)
        if main_stage:
            sections.extend(
                [
                    "## Result",
                    f"Stage: `{main_stage.name}`",
                    f"Status: `{main_stage.status}`",
                    "",
                    main_stage.output.strip() or main_stage.error or "",
                    "",
                    "## Stage Outputs",
                    "",
                ]
            )
            appendix_stages = [s for s in stages if s.name != main_stage.name]
        else:
            sections.extend(["## Stage Outputs", ""])
            appendix_stages = stages

        for stage in appendix_stages:
            sections.extend(
                [
                    f"### {stage.name}",
                    f"Status: `{stage.status}`",
                    "",
                    stage.output.strip() or stage.error or "",
                    "",
                ]
            )
        path.write_text("\n".join(sections), encoding="utf-8")
        return path

    def _done_message(self, job_id: str) -> str:
        job = self.db.get_subagent_run(job_id)
        if not job:
            return f"Subagent run `{job_id}` completed."
        summary = self._completion_summary(job)
        artifact = f"\n\nArtifact: `{job.artifact_path}`" if job.artifact_path else ""
        return f"Subagent run `{job_id}` completed.\n\nSummary:\n{summary}{artifact}"

    def _format_job(self, job, compact: bool = False) -> str:
        if compact:
            return f"- `{job.id}` {job.status} {job.engine}: {truncate_output(job.subject or '', 80)}"
        lines = [
            f"Subagent run `{job.id}`",
            f"- Agent: `{job.agent_name}`",
            f"- Status: `{job.status}`",
            f"- Engine: `{job.engine}`",
            f"- Subject: {job.subject or ''}",
            f"- Created: {job.created_at}",
        ]
        if job.artifact_path:
            lines.append(f"- Artifact: `{job.artifact_path}`")
        if job.id in self._tasks:
            lines.append("- Background task: `active`")
        elif job.status in ("pending", "running"):
            lines.append("- Background task: `missing`")
        if job.error:
            lines.append(f"- Error: {job.error}")
        if job.result:
            lines.extend(["", "Summary:", self._completion_summary(job, max_chars=3000)])
        return "\n".join(lines)

    def _completion_summary(self, job, max_chars: int = 2200) -> str:
        text = (job.result or "").strip()
        if not text and job.artifact_path:
            try:
                text = Path(job.artifact_path).read_text(encoding="utf-8")
            except OSError:
                text = ""
        return _completion_summary(text, max_chars=max_chars)

    def _stages_json(self, stages: List[SubAgentStage]) -> str:
        return json.dumps([asdict(s) for s in stages], ensure_ascii=False)

    def _load_resumable_stages(self, stages_json: str) -> List[SubAgentStage]:
        """Return the contiguous succeeded stage prefix that can be reused."""
        try:
            raw_stages = json.loads(stages_json or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(raw_stages, list):
            return []

        reusable: List[SubAgentStage] = []
        for expected_name, raw in zip(self.definition.stages, raw_stages):
            if not isinstance(raw, dict):
                break
            if raw.get("name") != expected_name or raw.get("status") != "succeeded":
                break
            reusable.append(
                SubAgentStage(
                    name=str(raw.get("name") or ""),
                    status="succeeded",
                    prompt=str(raw.get("prompt") or ""),
                    output=str(raw.get("output") or ""),
                    exit_code=raw.get("exit_code"),
                    error=raw.get("error"),
                )
            )
        return reusable

    def _stage_digest(self, stages: List[SubAgentStage], max_chars: int = 16000) -> str:
        chunks = []
        for stage in stages:
            chunks.append(f"## {stage.name}\n{stage.output or stage.error or ''}")
        return truncate_output("\n\n".join(chunks), max_chars)

    def _subject_from_payload(self, input_payload: Dict[str, str]) -> str:
        for key in ("subject", "prompt", "query", "title"):
            value = input_payload.get(key)
            if value:
                return str(value)
        first = next(iter(input_payload.values()), "")
        return str(first)

    def _main_stage_name(self) -> str:
        if self.definition.main_stage:
            return self.definition.main_stage
        return self.definition.stages[-1]

    def _render_stage_prompt(
        self, stage_name: str, input_payload: Dict[str, str], stages: List[SubAgentStage]
    ) -> str:
        template = self.definition.prompt_template(stage_name)
        subject = self._subject_from_payload(input_payload)
        values = {
            "base_contract": self.definition.base_contract,
            "subject": subject,
            "evidence": self._stage_digest(stages),
            "language": "auto",
        }
        values.update({key: str(value) for key, value in input_payload.items()})
        values.update({stage.name: truncate_output(stage.output or "", 6000) for stage in stages})
        return _render_template(template, values)


class SubAgentRegistry:
    """Route subagent lifecycle across all discovered kinds.

    Each kind keeps its own per-definition ``SubAgentRunManager`` (stages,
    artifact dir, in-memory tasks). The registry only dispatches:

    - ``start`` routes by ``agent_name``.
    - everything else routes by run id: the owning manager is found via the DB
      row's ``agent_name``. Because run ids share one prefix, id-based ops are
      kind-agnostic — adding a new subagent type is just a new definition file.

    Concurrency: ``max_concurrent`` is enforced strictly on ``start`` (the path
    the model drives, where runaway spawning matters). ``resume`` is also
    capped. ``resume_incomplete`` is crash recovery for already-persisted jobs;
    it respects remaining capacity between managers but a single manager may
    replay its own incomplete batch, so recovery can briefly exceed the live
    cap. That trade-off keeps v1 free of a job queue.
    """

    def __init__(
        self,
        db: Database,
        notifier: Optional[NotifyFn] = None,
        runner: Optional[SubAgentRunner] = None,
        max_concurrent: int = 1,
        base_dir: Optional[Path] = None,
    ):
        self.db = db
        self.max_concurrent = max(int(max_concurrent), 1)
        definitions = discover_definitions(base_dir)
        if not definitions:
            raise ValueError("No subagent definitions discovered under subagent_defs/")
        shared_runner = runner or SubAgentRunner()
        self._managers: Dict[str, SubAgentRunManager] = {
            name: SubAgentRunManager(
                db=db,
                runner=shared_runner,
                notifier=notifier,
                definition=definition,
            )
            for name, definition in definitions.items()
        }

    def agent_catalog(self) -> List[Dict[str, object]]:
        """Routing hints for the model: name, kind, description, required inputs."""
        return [
            {
                "name": m.definition.name,
                "kind": m.definition.kind,
                "description": m.definition.description,
                "required_inputs": list(m.definition.required_inputs),
            }
            for m in self._managers.values()
        ]

    def _active_count(self) -> int:
        return sum(len(m._tasks) for m in self._managers.values())

    def start(self, agent_name: str, input_payload: Dict[str, str], **kwargs) -> str:
        manager = self._managers.get(agent_name)
        if manager is None:
            available = ", ".join(sorted(self._managers))
            raise ValueError(
                f"Unknown subagent '{agent_name}'. Available: {available}"
            )
        if self._active_count() >= self.max_concurrent:
            raise RuntimeError(
                f"Subagent concurrency limit reached ({self.max_concurrent}); "
                "wait for a running job to finish."
            )
        return manager.start(input_payload=input_payload, **kwargs)

    def _owner(self, job_id: str) -> Optional[SubAgentRunManager]:
        run = self.db.get_subagent_run(job_id)
        if not run:
            return None
        return self._managers.get(run.agent_name)

    def status_text(self, job_id: str) -> str:
        manager = self._owner(job_id)
        return (
            manager.status_text(job_id)
            if manager
            else f"Subagent run `{job_id}` not found"
        )

    def cancel(self, job_id: str) -> bool:
        manager = self._owner(job_id)
        return bool(manager and manager.cancel(job_id))

    def resume(self, job_id: str) -> bool:
        manager = self._owner(job_id)
        if not manager:
            return False
        # Already-running jobs resume in place without consuming new capacity.
        if job_id not in manager._tasks and self._active_count() >= self.max_concurrent:
            return False
        return manager.resume(job_id)

    def list_text(self, limit: int = 10) -> str:
        runs = self.db.list_subagent_runs(limit=limit)
        if not runs:
            return "No subagent runs"
        lines = []
        for run in runs:
            manager = self._managers.get(run.agent_name)
            if manager:
                lines.append(manager._format_job(run, compact=True))
            else:
                lines.append(
                    f"- `{run.id}` {run.status} {run.engine}: "
                    f"{truncate_output(run.subject or '', 80)}"
                )
        return "\n".join(lines)

    def resume_incomplete(self, limit: int = 20) -> List[str]:
        resumed: List[str] = []
        for manager in self._managers.values():
            if self._active_count() >= self.max_concurrent:
                break
            resumed.extend(manager.resume_incomplete(limit=limit))
        return resumed

    async def stop(self) -> None:
        for manager in self._managers.values():
            await manager.stop()
