"""Test configuration for secretary v2."""

import pytest
import asyncio
from pathlib import Path
from typing import Generator

from config import Config, LLMConfig, ChannelConfig, DatabaseConfig, SkillsConfig, SearchConfig, HistoryConfig, SubagentConfig, reset_config
from memory import Database
from skills_loader import reset_skills_loader


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Create test configuration."""
    return Config(
        llm=LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        ),
        channels=ChannelConfig(enabled=True, default_outgoing="cli"),
        database=DatabaseConfig(path=":memory:"),
        skills=SkillsConfig(),
        search=SearchConfig(),
        history=HistoryConfig(),
        subagent=SubagentConfig(),
        schedules={},
        timezone="Asia/Shanghai",
    )


@pytest.fixture
def test_db(test_config):
    """Create test database."""
    db = Database(db_path=":memory:")
    yield db


@pytest.fixture
def skills_loader():
    """Create test skills loader."""
    reset_skills_loader()
    from skills_loader import SkillsLoader
    loader = SkillsLoader()
    yield loader
    reset_skills_loader()


@pytest.fixture
def mock_agent_response(monkeypatch):
    """Mock agent response for testing."""
    async def mock_run(*args, **kwargs):
        class MockResult:
            output = "Mock response"
        return MockResult()

    monkeypatch.setattr("runtime.agent.run", mock_run)
