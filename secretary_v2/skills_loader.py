"""Skills loader for secretary v2 - supports multiple skill formats."""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


# Skills directory
SKILLS_DIR = Path(__file__).parent / "skills"
DEFAULT_GLOBAL_SKILL_DIRS = [
    Path("~/.agents/skills"),
    Path("~/.claude/skills"),
    Path("~/.config/opencode/skills"),
]


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
    ):
        self.name = name
        self.description = description
        self.triggers = triggers or []
        self.auto_load = auto_load
        self.skill_type = skill_type
        self.content_path = content_path
        self.root_dir = root_dir
        self._source = source

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
        """Load skill metadata from frontmatter."""
        for index, skills_dir in enumerate(self.skills_dirs):
            if not skills_dir.exists():
                continue
            source = "project" if index == 0 else "global"

            for item in skills_dir.iterdir():
                if item.name.startswith('.'):
                    continue

                if item.is_file() and item.suffix == ".md":
                    self._parse_markdown_skill(
                        item,
                        skill_type="file",
                        root_dir=skills_dir,
                        source=source,
                    )
                elif item.is_dir():
                    # Check for SKILL.md, index.md or README.md in directory
                    for index_file in ["SKILL.md", "index.md", "README.md"]:
                        index_path = item / index_file
                        if index_path.exists():
                            self._parse_markdown_skill(
                                index_path,
                                skill_type="directory",
                                name=item.name,
                                root_dir=skills_dir,
                                source=source,
                            )
                            break

    def _parse_markdown_skill(
        self,
        file_path: Path,
        skill_type: str = "file",
        name: Optional[str] = None,
        root_dir: Optional[Path] = None,
        source: str = "project",
    ):
        """Parse markdown skill file with YAML frontmatter."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract frontmatter
            frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
            if not frontmatter_match:
                # No frontmatter, create default metadata
                skill_name = name or file_path.stem
                if skill_name in self._skills:
                    return
                self._skills[skill_name] = SkillMeta(
                    name=skill_name,
                    description=content[:100] + "..." if len(content) > 100 else content,
                    skill_type=skill_type,
                    content_path=file_path,
                    root_dir=root_dir,
                    source=source,
                )
                return

            frontmatter_yaml = frontmatter_match.group(1)
            metadata = yaml.safe_load(frontmatter_yaml) or {}

            skill_name = name or metadata.get("name", file_path.stem)
            if skill_name in self._skills:
                # Earlier roots have higher precedence. By default this means
                # built-in/project skills override global user skills.
                return

            triggers = self._normalize_triggers(metadata)

            # Check for auto_load
            auto_load = metadata.get("auto_load", False)

            self._skills[skill_name] = SkillMeta(
                name=skill_name,
                description=metadata.get("description", ""),
                triggers=triggers,
                auto_load=auto_load,
                skill_type=skill_type,
                content_path=file_path,
                root_dir=root_dir,
                source=source,
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
        """Get skill content by name."""
        if skill_name not in self._skills:
            return None

        skill_meta = self._skills[skill_name]

        if skill_meta.content_path and skill_meta.content_path.exists():
            return skill_meta.content_path.read_text(encoding="utf-8")

        return None

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
