#!/usr/bin/env bash
# Safely archive and rebuild Secretary's SQLite database on a systemd host.
#
# Usage:
#   bash deploy/rebuild-database.sh --yes
#   bash deploy/rebuild-database.sh --yes --no-start
#   SECRETARY_VENV=/opt/secretary-venv bash deploy/rebuild-database.sh --yes
#
# The script retains only events with status=open and all scheduled_tasks rows.
# Every other table starts empty. The complete previous database is archived in
# secretary_v2/archive/ and is never overwritten.

set -Eeuo pipefail

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: bash deploy/rebuild-database.sh --yes [--no-start] [migration options]

Options:
  --yes          Required acknowledgement that historical SQLite data will be
                 removed from the live database and retained in an archive.
  --no-start     Leave secretary.service stopped after a successful rebuild.
  --database P   Override database.path from secretary_v2/config.yaml.
  --archive-dir P
                 Override the archive directory.
  -h, --help     Show this help.
EOF
}

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SEC_DIR="$REPO_DIR/secretary_v2"
APP_HOME="$HOME"
VENV_DIR="$APP_HOME/.venvs/secretary"
if [ -n "${SECRETARY_VENV:-}" ]; then
    VENV_DIR="$SECRETARY_VENV"
fi
PYTHON_BIN="$VENV_DIR/bin/python"
if [ -n "${SECRETARY_PYTHON:-}" ]; then
    PYTHON_BIN="$SECRETARY_PYTHON"
fi
LOG_FILE="$SEC_DIR/logs/secretary_v2.log"
CONFIRMED=0
NO_START=0
PYTHON_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes)
            CONFIRMED=1
            shift
            ;;
        --no-start)
            NO_START=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

[ "$CONFIRMED" -eq 1 ] || die "Refusing destructive rebuild without --yes"
[ "$(id -u)" -ne 0 ] || die "Run as the normal Secretary service user, not root"
[ -x "$PYTHON_BIN" ] || die "Runtime Python not found: $PYTHON_BIN"
[ -f "$SEC_DIR/main.py" ] || die "Secretary source directory not found: $SEC_DIR"
command -v systemctl >/dev/null 2>&1 || die "systemd is required"
sudo -v || die "sudo access is required to stop and start secretary.service"

LOAD_STATE="$(systemctl show secretary.service --property=LoadState --value)"
[ "$LOAD_STATE" = "loaded" ] || die "secretary.service is not loaded on this host"
if systemctl is-failed --quiet secretary.service; then
    warn "secretary.service is failed. Inspect the log before rebuilding:"
    tail -n 80 "$LOG_FILE" >&2 || true
    die "Fix the startup failure first; the rebuild script will not consume restart budget"
fi

SERVICE_WAS_ACTIVE=0
START_ATTEMPTED=0
if systemctl is-active --quiet secretary.service; then
    SERVICE_WAS_ACTIVE=1
elif [ "$(systemctl is-active secretary.service || true)" != "inactive" ]; then
    die "secretary.service is neither active nor cleanly inactive"
else
    warn "secretary.service was already inactive; it will remain inactive"
fi

restart_after_failure() {
    exit_code=$?
    if [ "$exit_code" -ne 0 ] && [ "$SERVICE_WAS_ACTIVE" -eq 1 ] \
        && [ "$NO_START" -eq 0 ] && [ "$START_ATTEMPTED" -eq 0 ]; then
        warn "Rebuild did not complete; restarting secretary.service with the unchanged or restored database"
        sudo systemctl start secretary.service || true
    fi
    exit "$exit_code"
}
trap restart_after_failure EXIT

if [ "$SERVICE_WAS_ACTIVE" -eq 1 ]; then
    log "Stopping secretary.service"
    sudo systemctl stop secretary.service
fi
if systemctl is-active --quiet secretary.service; then
    die "secretary.service did not stop"
fi

log "Archiving and rebuilding the SQLite database"
(cd "$SEC_DIR" && "$PYTHON_BIN" "$REPO_DIR/deploy/rebuild_database.py" "${PYTHON_ARGS[@]}")

if [ "$SERVICE_WAS_ACTIVE" -eq 0 ] || [ "$NO_START" -eq 1 ]; then
    warn "Database rebuild complete; secretary.service remains stopped"
    exit 0
fi

LOG_OFFSET=0
if [ -f "$LOG_FILE" ]; then
    LOG_OFFSET="$(stat -c '%s' "$LOG_FILE")"
fi

log "Starting secretary.service and waiting for startup self-test"
START_ATTEMPTED=1
sudo systemctl start secretary.service

SELF_TEST_PASSED=0
for _ in $(seq 1 45); do
    if ! systemctl is-active --quiet secretary.service; then
        sudo systemctl --no-pager status secretary.service || true
        die "secretary.service stopped before startup validation completed"
    fi
    if [ -f "$LOG_FILE" ]; then
        if tail -c "+$((LOG_OFFSET + 1))" "$LOG_FILE" | grep -q "Startup self-test: FAILED"; then
            die "Secretary startup self-test failed; inspect $LOG_FILE"
        fi
        if tail -c "+$((LOG_OFFSET + 1))" "$LOG_FILE" | grep -q "Startup self-test: PASSED"; then
            SELF_TEST_PASSED=1
            break
        fi
    fi
    sleep 1
done

[ "$SELF_TEST_PASSED" -eq 1 ] || die "Timed out waiting for Secretary startup self-test; inspect $LOG_FILE"
log "Database rebuild complete; secretary.service is active and startup self-test passed"
