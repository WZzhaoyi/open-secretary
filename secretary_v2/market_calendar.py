"""Trading calendar service with Longbridge as the primary provider.

The service fetches a date window per market and keeps it in process memory.
That keeps scheduled tasks and tools from calling Longbridge for every single
date check while still avoiding hand-maintained holiday rules.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from importlib import metadata, util
from typing import Callable, Dict, Iterable, List, Optional, Set

from config import MarketCalendarConfig, get_config

logger = logging.getLogger(__name__)

_MARKET_ALIASES = {
    "A": "CN",
    "A股": "CN",
    "CN": "CN",
    "CHINA": "CN",
    "SH": "CN",
    "SZ": "CN",
    "HK": "HK",
    "HONGKONG": "HK",
    "US": "US",
    "USA": "US",
    "SG": "SG",
}
_PMC_CALENDAR_CANDIDATES = {
    "CN": ("SSE", "XSHG"),
    "HK": ("HKEX", "XHKG"),
    "US": ("NYSE", "XNYS"),
    "SG": ("XSES",),
}


def _optional_package_status(module_name: str, distribution_name: str) -> Dict[str, object]:
    available = util.find_spec(module_name) is not None
    version = None
    if available:
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": available, "version": version}


class MarketCalendarError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalendarRange:
    market: str
    start: date
    end: date
    trading_days: Set[date]
    half_trading_days: Set[date]
    source: str
    fetched_at: float
    degraded: bool = False
    error: Optional[str] = None

    def covers(self, start: date, end: date) -> bool:
        return self.start <= start and self.end >= end

    def is_fresh(self, now: float, ttl_seconds: int) -> bool:
        return now - self.fetched_at <= ttl_seconds

    def clipped(self, start: date, end: date) -> "CalendarRange":
        return CalendarRange(
            market=self.market,
            start=start,
            end=end,
            trading_days={d for d in self.trading_days if start <= d <= end},
            half_trading_days={d for d in self.half_trading_days if start <= d <= end},
            source=self.source,
            fetched_at=self.fetched_at,
            degraded=self.degraded,
            error=self.error,
        )


@dataclass(frozen=True)
class CalendarDay:
    market: str
    date: date
    is_trading_day: bool
    is_half_trading_day: bool
    source: str
    degraded: bool = False
    error: Optional[str] = None
    next_trading_day: Optional[date] = None


def normalize_market(market: str) -> str:
    normalized = (market or "").strip().upper().replace(".", "")
    if normalized not in _MARKET_ALIASES:
        raise ValueError(f"unsupported market: {market}")
    return _MARKET_ALIASES[normalized]


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _parse_json_prefix(text: str):
    """Parse the first JSON value from CLI output.

    Longbridge can append release notices after the JSON payload, so callers
    must not require stdout to be JSON and nothing else.
    """
    if not text:
        raise MarketCalendarError("empty Longbridge output")
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not starts:
        raise MarketCalendarError("Longbridge output did not contain JSON")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[min(starts) :])
    return payload


class MarketCalendarService:
    def __init__(
        self,
        config: Optional[MarketCalendarConfig] = None,
        *,
        now: Callable[[], float] = time.time,
    ):
        self.config = config or get_config().market_calendar
        self._now = now
        self._cache: Dict[str, CalendarRange] = {}

    def cache_stats(self) -> Dict[str, Dict[str, object]]:
        return {
            market: {
                "start": entry.start.isoformat(),
                "end": entry.end.isoformat(),
                "source": entry.source,
                "degraded": entry.degraded,
                "trading_days": len(entry.trading_days),
                "half_trading_days": len(entry.half_trading_days),
            }
            for market, entry in sorted(self._cache.items())
        }

    def strategy_status(self) -> Dict[str, object]:
        pmc = _optional_package_status(
            "pandas_market_calendars",
            "pandas-market-calendars",
        )
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "fallback_provider": self.config.fallback_provider,
            "last_resort_fallback": "weekday",
            "markets": list(self.config.markets),
            "window": {
                "lookbehind_days": self.config.lookbehind_days,
                "lookahead_days": self.config.lookahead_days,
            },
            "cache_ttl_minutes": self.config.cache_ttl_minutes,
            "timeout_seconds": self.config.timeout_seconds,
            "command": self.config.command,
            "optional_dependencies": {
                "pandas_market_calendars": pmc,
            },
        }

    def strategy_summary(self) -> str:
        status = self.strategy_status()
        pmc = status["optional_dependencies"]["pandas_market_calendars"]
        pmc_state = (
            f"available version={pmc['version']}"
            if pmc["available"]
            else "not installed"
        )
        return (
            "Market calendar strategy: "
            f"enabled={status['enabled']}; "
            f"primary={status['provider']} command={status['command']!r}; "
            f"fallback={status['fallback_provider']} ({pmc_state}); "
            f"last_resort={status['last_resort_fallback']}; "
            f"markets={','.join(status['markets'])}; "
            f"window=-{status['window']['lookbehind_days']}d/+{status['window']['lookahead_days']}d; "
            f"cache_ttl={status['cache_ttl_minutes']}m; "
            f"timeout={status['timeout_seconds']}s"
        )

    def get_range(self, market: str, start: date, end: date) -> CalendarRange:
        market = normalize_market(market)
        if end < start:
            raise ValueError("end date must be on or after start date")

        ttl_seconds = max(0, int(self.config.cache_ttl_minutes) * 60)
        now = self._now()
        cached = self._cache.get(market)
        if cached and cached.covers(start, end) and cached.is_fresh(now, ttl_seconds):
            return cached.clipped(start, end)

        fetch_start = start - timedelta(days=max(0, int(self.config.lookbehind_days)))
        fetch_end = end + timedelta(days=max(0, int(self.config.lookahead_days)))
        fetched = self._fetch_with_fallback(market, fetch_start, fetch_end, now)
        self._cache[market] = fetched
        return fetched.clipped(start, end)

    def day_status(self, market: str, target: date) -> CalendarDay:
        market = normalize_market(market)
        lookahead_end = target + timedelta(days=max(1, int(self.config.lookahead_days)))
        self.get_range(market, target, target)
        window = self._cache[market].clipped(target, lookahead_end)
        next_trading_day = next(
            (day for day in sorted(window.trading_days) if day > target),
            None,
        )
        return CalendarDay(
            market=market,
            date=target,
            is_trading_day=target in window.trading_days,
            is_half_trading_day=target in window.half_trading_days,
            source=window.source,
            degraded=window.degraded,
            error=window.error,
            next_trading_day=next_trading_day,
        )

    def _fetch_with_fallback(
        self,
        market: str,
        start: date,
        end: date,
        fetched_at: float,
    ) -> CalendarRange:
        if self.config.enabled and self.config.provider.lower() == "longbridge":
            try:
                return self._fetch_longbridge(market, start, end, fetched_at)
            except Exception as exc:
                logger.warning(
                    "Longbridge trading calendar failed for %s %s..%s: %s",
                    market,
                    start,
                    end,
                    exc,
                )
                fallback = self._fetch_fallback(market, start, end, fetched_at)
                return replace(fallback, degraded=True, error=str(exc))
        return self._fetch_fallback(market, start, end, fetched_at)

    def _fetch_longbridge(
        self,
        market: str,
        start: date,
        end: date,
        fetched_at: float,
    ) -> CalendarRange:
        command = [
            *shlex.split(self.config.command or "longbridge"),
            "trading",
            "days",
            market,
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
            "--format",
            "json",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, int(self.config.timeout_seconds)),
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise MarketCalendarError(
                f"Longbridge exited {completed.returncode}: {stderr[:240]}"
            )

        payload = _parse_json_prefix(completed.stdout)
        trading_days = {_parse_date(value) for value in payload.get("trading_days", [])}
        half_days = {
            _parse_date(value)
            for value in payload.get("half_trading_days", [])
        }
        return CalendarRange(
            market=market,
            start=start,
            end=end,
            trading_days=trading_days,
            half_trading_days=half_days,
            source="longbridge",
            fetched_at=fetched_at,
        )

    def _fetch_fallback(
        self,
        market: str,
        start: date,
        end: date,
        fetched_at: float,
    ) -> CalendarRange:
        if self.config.fallback_provider.lower() == "pandas_market_calendars":
            try:
                return self._fetch_pandas_market_calendars(
                    market,
                    start,
                    end,
                    fetched_at,
                )
            except Exception as exc:
                logger.warning("pandas_market_calendars fallback failed: %s", exc)

        trading_days = {day for day in _date_range(start, end) if day.weekday() < 5}
        return CalendarRange(
            market=market,
            start=start,
            end=end,
            trading_days=trading_days,
            half_trading_days=set(),
            source="weekday_fallback",
            fetched_at=fetched_at,
            degraded=True,
        )

    def _fetch_pandas_market_calendars(
        self,
        market: str,
        start: date,
        end: date,
        fetched_at: float,
    ) -> CalendarRange:
        import pandas_market_calendars as pmc

        calendar = None
        for name in _PMC_CALENDAR_CANDIDATES.get(market, (market,)):
            try:
                calendar = pmc.get_calendar(name)
                break
            except Exception:
                continue
        if calendar is None:
            raise MarketCalendarError(f"no pandas_market_calendars calendar for {market}")

        schedule = calendar.schedule(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        trading_days = {ts.date() for ts in schedule.index}
        half_days: Set[date] = set()
        try:
            half_days = {ts.date() for ts in calendar.early_closes(schedule).index}
        except Exception:
            half_days = set()
        return CalendarRange(
            market=market,
            start=start,
            end=end,
            trading_days=trading_days,
            half_trading_days=half_days,
            source="pandas_market_calendars",
            fetched_at=fetched_at,
            degraded=True,
        )


_market_calendar_service: Optional[MarketCalendarService] = None


def get_market_calendar_service() -> MarketCalendarService:
    global _market_calendar_service
    if _market_calendar_service is None:
        _market_calendar_service = MarketCalendarService()
    return _market_calendar_service


def reset_market_calendar_service() -> None:
    global _market_calendar_service
    _market_calendar_service = None
