"""One-off migration from legacy research_jobs to subagent_runs.

Only successful deep research jobs are preserved. Other legacy rows are
discarded when the old table is dropped.
"""

import argparse
import json
import sqlite3
from pathlib import Path


SUBAGENT_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS subagent_runs (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    agent_kind TEXT NOT NULL DEFAULT '',
    engine TEXT NOT NULL,
    input_json TEXT NOT NULL DEFAULT '{}',
    subject TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    origin_channel TEXT NOT NULL DEFAULT 'cli',
    user_id TEXT,
    stages_json TEXT DEFAULT '[]',
    artifact_path TEXT,
    result TEXT,
    error TEXT,
    created_at DATETIME,
    updated_at DATETIME,
    completed_at DATETIME
)
"""


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_subagent_runs_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SUBAGENT_RUNS_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_subagent_runs_agent_name "
        "ON subagent_runs (agent_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_subagent_runs_agent_kind "
        "ON subagent_runs (agent_kind)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_subagent_runs_status "
        "ON subagent_runs (status)"
    )


def migrate(db_path: str, drop_old_table: bool = True) -> int:
    """Migrate succeeded legacy research rows and return inserted row count."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_subagent_runs_schema(conn)
        if not _table_exists(conn, "research_jobs"):
            return 0

        rows = conn.execute(
            """
            SELECT id, engine, topic, status, origin_channel, user_id,
                   stages_json, artifact_path, result, error,
                   created_at, updated_at, completed_at
            FROM research_jobs
            WHERE status = 'succeeded'
            """
        ).fetchall()

        inserted = 0
        for row in rows:
            topic = row["topic"] or ""
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO subagent_runs (
                    id, agent_name, agent_kind, engine, input_json, subject,
                    status, origin_channel, user_id, stages_json,
                    artifact_path, result, error,
                    created_at, updated_at, completed_at
                )
                VALUES (
                    ?, 'deep_research', 'research', ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    row["id"],
                    row["engine"],
                    json.dumps({"topic": topic}, ensure_ascii=False),
                    topic,
                    row["status"],
                    row["origin_channel"] or "cli",
                    row["user_id"],
                    row["stages_json"] or "[]",
                    row["artifact_path"],
                    row["result"],
                    row["error"],
                    row["created_at"],
                    row["updated_at"],
                    row["completed_at"],
                ),
            )
            inserted += cursor.rowcount

        if drop_old_table:
            conn.execute("DROP TABLE research_jobs")
        conn.commit()
        return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate succeeded research_jobs rows into subagent_runs."
    )
    parser.add_argument("db_path", help="Path to secretary_v2 SQLite database")
    parser.add_argument(
        "--keep-old-table",
        action="store_true",
        help="Copy rows but do not drop research_jobs",
    )
    args = parser.parse_args()

    count = migrate(args.db_path, drop_old_table=not args.keep_old_table)
    print(f"Migrated {count} succeeded research_jobs row(s) into subagent_runs.")


if __name__ == "__main__":
    main()
