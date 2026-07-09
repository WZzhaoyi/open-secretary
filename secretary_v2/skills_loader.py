"""Skills loader for secretary v2 - supports multiple skill formats."""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


# Skills directory
SKILLS_DIR = Path(__file__).parent / "skills"
# Global skill roots, aligned with the Agent Skills standard
# (`skills/**/SKILL.md`). Tool-specific skill directories are deliberately not
# hardcoded here; point `skills.paths` in config.yaml at any extra roots.
DEFAULT_GLOBAL_SKILL_DIRS = [
    Path("~/.agents/skills"),
    Path("~/.claude/skills"),
]

# Single regex used both to parse frontmatter and to strip it from the body, so
# discovery and content injection always agree on where the body starts.
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Return the markdown body, dropping a leading YAML frontmatter block."""
    match = _FRONTMATTER_RE.match(text)
    return text[match.end():] if match else text


class SkillMeta:
    """Skill metadata."""

    def __init__(
        self,
        name: str,
        description: str = "",
        triggers: List[str] = None,
        auto_load: bool = False,
        skill_type: str = "file",  # "file" or "directory"
        content_path: Optional[Path] = None,
        root_dir: Optional[Path] = None,
        source: str = "project",
        base_dir: Optional[Path] = None,
        resources: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.triggers = triggers or []
        self.auto_load = auto_load
        self.skill_type = skill_type
        self.content_path = content_path
        self.root_dir = root_dir
        self._source = source
        # base_dir is the skill's own folder (directory skills only); bundled
        # resource files are resolved relative to it. resources lists their
        # paths (posix, relative to base_dir) for on-demand progressive loading.
        self.base_dir = base_dir
        self.resources = resources or []

    @property
    def source(self) -> str:
        """Return a compact source label for LLM-facing discovery."""
        return self._source


class SkillsLoader:
    """Skills loader with trigger-based loading."""

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        skills_dirs: Optional[Iterable[Path]] = None,
        trigger_overrides: Optional[Dict[str, List[str]]] = None,
        auto_load: Optional[List[str]] = None,
        max_loaded: Optional[int] = None,
    ):
        if skills_dirs is not None:
            self.skills_dirs = [Path(p).expanduser() for p in skills_dirs]
        else:
            self.skills_dirs = [Path(skills_dir or SKILLS_DIR).expanduser()]
        # Keep the old public attribute for compatibility with tests/callers.
        self.skills_dir = self.skills_dirs[0]
        self.trigger_overrides = trigger_overrides or {}
        self.auto_load = auto_load or []
        self.max_loaded = max_loaded
        self._skills: Dict[str, SkillMeta] = {}
        self._load_skill_metadata()
        self._apply_config_overrides()

    def _load_skill_metadata(self):
        """Discover skills across all roots.

        Two formats are supported, matching the Agent Skills standard plus this
        project's single-file extension:

        - Directory skills: any-depth ``**/SKILL.md`` (the Agent Skills
          standard). Bundled resource files travel with the skill for
          progressive loading.
        - File skills: top-level ``*.md`` in a root (this project's own format).

        Earlier roots win on name collisions, so project skills override global
        ones. Names come from frontmatter, falling back to the folder/file name.
        """
        for index, skills_dir in enumerate(self.skills_dirs):
            if not skills_dir.exists():
                continue
            source = "project" if index == 0 else "global"

            for item in sorted(skills_dir.glob("*.md")):
                if item.name.startswith("."):
                    continue
                self._parse_markdown_skill(
                    item,
                    skill_type="file",
                    default_name=item.stem,
                    root_dir=skills_dir,
                    source=source,
                )

            for index_path in sorted(skills_dir.rglob("SKILL.md")):
                rel_parts = index_path.relative_to(skills_dir).parts
                if any(part.startswith(".") for part in rel_parts):
                    continue
                self._parse_markdown_skill(
                    index_path,
                    skill_type="directory",
                    default_name=index_path.parent.name,
                    root_dir=skills_dir,
                    source=source,
                )

    def _scan_resources(self, base_dir: Path, index_path: Path) -> List[str]:
        """List bundled files under a directory skill (excluding the index)."""
        resources: List[str] = []
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file() or path == index_path:
                continue
            rel = path.relative_to(base_dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            resources.append(rel.as_posix())
        return resources

    def _parse_markdown_skill(
        self,
        file_path: Path,
        skill_type: str = "file",
        default_name: Optional[str] = None,
        root_dir: Optional[Path] = None,
        source: str = "project",
    ):
        """Parse a skill markdown file (SKILL.md or single-file *.md)."""
        try:
            content = file_path.read_text(encoding="utf-8")
            frontmatter_match = _FRONTMATTER_RE.match(content)
            metadata = {}
            if frontmatter_match:
                metadata = yaml.safe_load(frontmatter_match.group(1)) or {}

            skill_name = metadata.get("name") or default_name or file_path.stem
            if skill_name in self._skills:
                # Earlier roots have higher precedence. By default this means
                # built-in/project skills override global user skills.
                return

            description = metadata.get("description", "")
            if not description and not frontmatter_match:
                description = content[:100] + "..." if len(content) > 100 else content

            base_dir = file_path.parent if skill_type == "directory" else None
            resources = self._scan_resources(base_dir, file_path) if base_dir else []

            self._skills[skill_name] = SkillMeta(
                name=skill_name,
                description=description,
                triggers=self._normalize_triggers(metadata),
                auto_load=metadata.get("auto_load", False),
                skill_type=skill_type,
                content_path=file_path,
                root_dir=root_dir,
                source=source,
                base_dir=base_dir,
                resources=resources,
            )
        except Exception as e:
            print(f"Warning: Failed to parse skill {file_path}: {e}")

    def _apply_config_overrides(self):
        """Apply config.yaml skill controls after all roots are loaded."""
        for skill_name, triggers in self.trigger_overrides.items():
            if skill_name not in self._skills:
                continue
            normalized = self._normalize_trigger_values(triggers)
            self._skills[skill_name].triggers = normalized

        for skill_name in self.auto_load:
            if skill_name in self._skills:
                self._skills[skill_name].auto_load = True

    def _normalize_triggers(self, metadata: dict) -> List[str]:
        """Return explicitly configured trigger terms only."""
        raw_triggers = metadata.get("triggers", metadata.get("trigger", []))
        return self._normalize_trigger_values(raw_triggers)

    def _normalize_trigger_values(self, raw_triggers) -> List[str]:
        """Normalize trigger values from skill frontmatter or config.yaml."""
        if isinstance(raw_triggers, str):
            raw_triggers = [raw_triggers]
        if not isinstance(raw_triggers, list):
            return []

        triggers = []
        for trigger in raw_triggers:
            if not isinstance(trigger, str):
                continue
            trigger = trigger.strip()
            if trigger:
                triggers.append(trigger)
        return triggers

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """Get a skill's markdown body (frontmatter stripped) by name."""
        skill_meta = self._skills.get(skill_name)
        if not skill_meta or not skill_meta.content_path:
            return None
        if not skill_meta.content_path.exists():
            return None
        return _strip_frontmatter(
            skill_meta.content_path.read_text(encoding="utf-8")
        )

    def get_skill_resources(self, skill_name: str) -> List[str]:
        """List a directory skill's bundled resource files (relative paths)."""
        skill_meta = self._skills.get(skill_name)
        return list(skill_meta.resources) if skill_meta else []

    def get_skill_resource(self, skill_name: str, relpath: str) -> Optional[str]:
        """Read one bundled resource file, confined to the skill's directory.

        Returns None if the skill has no bundle or the file is missing. Raises
        ValueError if ``relpath`` escapes the skill directory (path traversal).
        """
        skill_meta = self._skills.get(skill_name)
        if not skill_meta or skill_meta.skill_type != "directory" or not skill_meta.base_dir:
            return None
        base = skill_meta.base_dir.resolve()
        target = (base / relpath).resolve()
        if target != base and base not in target.parents:
            raise ValueError(
                f"resource path escapes skill directory: {relpath}"
            )
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def get_auto_loaded_skills(self) -> List[str]:
        """Get skills configured for every-run injection."""
        auto_loaded = [
            skill_name
            for skill_name, meta in self._skills.items()
            if meta.auto_load
        ]
        if self.max_loaded is not None:
            auto_loaded = auto_loaded[: self.max_loaded]
        return auto_loaded

    def get_triggered_skills(
        self,
        user_message: str,
        include_auto: bool = True,
    ) -> List[str]:
        """Get skills triggered by user message."""
        if self.max_loaded is not None and self.max_loaded <= 0:
            return []

        user_message_lower = user_message.lower()
        trigger_matched = []
        auto_loaded = []

        for skill_name, meta in self._skills.items():
            matched = False
            for trigger in meta.triggers:
                if trigger and trigger.lower() in user_message_lower:
                    trigger_matched.append(skill_name)
                    matched = True
                    break

            if include_auto and meta.auto_load and not matched:
                auto_loaded.append(skill_name)

        triggered = trigger_matched + auto_loaded
        if self.max_loaded is not None:
            triggered = triggered[: self.max_loaded]

        return triggered

    def get_all_skills(self) -> Dict[str, SkillMeta]:
        """Get all loaded skills metadata."""
        return self._skills.copy()

    def get_skill_index(self) -> str:
        """Return a compact skill index for discovery prompts."""
        lines = []
        for name, meta in sorted(self._skills.items()):
            description = " ".join((meta.description or "").split())
            if len(description) > 220:
                description = description[:217].rstrip() + "..."
            triggers = ", ".join(meta.triggers[:5]) if meta.triggers else "none"
            lines.append(
                f"- {name} [{meta.source}]: {description or 'No description'} "
                f"(triggers: {triggers})"
            )
        return "\n".join(lines)


# Global skills loader instance
_skills_loader: Optional[SkillsLoader] = None


def get_skills_loader() -> SkillsLoader:
    """Get global skills loader instance."""
    global _skills_loader
    if _skills_loader is None:
        from config import get_config

        cfg = get_config().skills
        skill_dirs = [SKILLS_DIR]
        if cfg.include_global:
            skill_dirs.extend(DEFAULT_GLOBAL_SKILL_DIRS)
        skill_dirs.extend(Path(p) for p in cfg.paths)
        _skills_loader = SkillsLoader(
            skills_dirs=skill_dirs,
            trigger_overrides=cfg.triggers,
            auto_load=cfg.auto_load,
            max_loaded=cfg.max_loaded,
        )
    return _skills_loader


def reset_skills_loader():
    """Reset global skills loader (for testing)."""
    global _skills_loader
    _skills_loader = None
