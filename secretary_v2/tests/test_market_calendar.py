from datetime import date
from types import SimpleNamespace

from config import MarketCalendarConfig
from market_calendar import MarketCalendarService, _parse_json_prefix
from memory import Database


def test_runtime_policy_keeps_non_trading_days_informational_only():
    import runtime

    stable, _ = runtime._build_context_layers(
        runtime.SecretaryDeps(
            db=Database(db_path=":memory:"),
            current_time="2026-06-23T08:00:00+08:00",
        )
    )

    assert "A non-trading day does not imply `NO_ACTION`" in stable
    assert "休市 days may still need review reminders" in stable


def test_parse_json_prefix_ignores_longbridge_update_notice():
    payload = _parse_json_prefix(
        '{\n'
        '  "half_trading_days": [],\n'
        '  "trading_days": ["2026-06-23"]\n'
        '}\n\n'
        "New version 0.23.3 is available\n"
    )

    assert payload["trading_days"] == ["2026-06-23"]


def test_longbridge_calendar_fetches_window_and_uses_cache(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"half_trading_days":[],"trading_days":'
                '["2026-06-22","2026-06-23","2026-06-24"]}'
                "\nNew version 0.23.3 is available\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("market_calendar.subprocess.run", fake_run)
    service = MarketCalendarService(
        MarketCalendarConfig(
            lookbehind_days=1,
            lookahead_days=2,
            cache_ttl_minutes=60,
            command="longbridge",
        ),
        now=lambda: 1000.0,
    )

    first = service.day_status("CN", date(2026, 6, 23))
    second = service.day_status("CN", date(2026, 6, 24))

    assert first.is_trading_day
    assert second.is_trading_day
    assert first.source == "longbridge"
    assert second.source == "longbridge"
    assert len(calls) == 1
    assert calls[0][:3] == ["longbridge", "trading", "days"]
    assert "--start" in calls[0]
    assert "2026-06-22" in calls[0]
    assert "--end" in calls[0]
    assert "2026-06-25" in calls[0]


def test_longbridge_failure_falls_back_to_weekday(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="not authenticated",
        )

    monkeypatch.setattr("market_calendar.subprocess.run", fake_run)
    service = MarketCalendarService(
        MarketCalendarConfig(
            fallback_provider="weekday",
            lookbehind_days=0,
            lookahead_days=2,
            cache_ttl_minutes=60,
            command="longbridge",
        ),
        now=lambda: 1000.0,
    )

    weekday = service.day_status("US", date(2026, 6, 23))
    weekend = service.day_status("US", date(2026, 6, 27))

    assert weekday.is_trading_day
    assert weekday.source == "weekday_fallback"
    assert weekday.degraded
    assert "Longbridge exited 1" in (weekday.error or "")
    assert not weekend.is_trading_day
    assert weekend.source == "weekday_fallback"


def test_strategy_summary_reports_provider_fallback_and_cache():
    service = MarketCalendarService(
        MarketCalendarConfig(
            provider="longbridge",
            fallback_provider="pandas_market_calendars",
            markets=["CN", "HK", "US"],
            lookbehind_days=3,
            lookahead_days=14,
            cache_ttl_minutes=720,
            timeout_seconds=10,
        )
    )

    summary = service.strategy_summary()

    assert "primary=longbridge" in summary
    assert "fallback=pandas_market_calendars" in summary
    assert "last_resort=weekday" in summary
    assert "markets=CN,HK,US" in summary
    assert "window=-3d/+14d" in summary
    assert "cache_ttl=720m" in summary
