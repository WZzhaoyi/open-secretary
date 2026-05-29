#!/bin/bash
# Secretary v2 management script: start | stop | restart | status | logs
#
# The background service runs with `--channel telegram`, which starts both
# Telegram and HTTP channels. The HTTP webhook listens on the stable default
# port 11269 for external calls and tests.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"
CHANNEL="telegram"
HTTP_PORT=11269                       # HTTPChannel default port
PATTERN="main.py --channel $CHANNEL"   # Unique process identifier
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/secretary_v2.log"
HEALTH_URL="http://127.0.0.1:$HTTP_PORT/health"
START_TIMEOUT=75                       # Includes the ~40s startup self-test

mkdir -p "$LOG_DIR"

# Current running PID. Empty means the service is not running.
get_pid() {
    pgrep -f "$PATTERN" | head -1
}

do_start() {
    local pid
    pid="$(get_pid)"
    if [ -n "$pid" ]; then
        echo "Already running (PID $pid); skipping start. Two polling instances would trigger Telegram 409."
        return 0
    fi

    # Rotate the previous log to avoid unbounded growth.
    if [ -f "$LOG_FILE" ]; then
        mv "$LOG_FILE" "$LOG_FILE.$(date +%Y%m%d-%H%M%S)"
    fi

    echo "Starting Secretary v2 (--channel $CHANNEL)..."
    cd "$SCRIPT_DIR" || exit 1
    nohup "$PYTHON" main.py --channel "$CHANNEL" > "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "Started PID ${new_pid}; waiting for startup self-test (up to ${START_TIMEOUT}s)..."

    local waited=0
    while [ "$waited" -lt "$START_TIMEOUT" ]; do
        if ! kill -0 "${new_pid}" 2>/dev/null; then
            echo "✗ Process exited early; startup failed. Last 20 log lines:"
            tail -20 "$LOG_FILE"
            return 1
        fi
        if grep -q "Starting .* channel" "$LOG_FILE" 2>/dev/null; then
            echo "✓ Started successfully (PID ${new_pid})"
            grep -E "self-test|Starting .* channel|Scheduler started" "$LOG_FILE" | tail -4
            return 0
        fi
        if grep -q "self-test FAILED\|self-test: FAILED" "$LOG_FILE" 2>/dev/null; then
            echo "✗ Startup self-test failed; the process will exit. Last 20 log lines:"
            tail -20 "$LOG_FILE"
            return 1
        fi
        sleep 3
        waited=$((waited + 3))
    done
    echo "⚠ No channel startup log within ${START_TIMEOUT}s; check manually: $0 status"
    return 1
}

do_stop() {
    local pid
    pid="$(get_pid)"
    if [ -z "$pid" ]; then
        echo "Not running."
        return 0
    fi
    echo "Stopping Secretary v2 (PID $pid)..."
    kill "$pid" 2>/dev/null
    local waited=0
    while [ "$waited" -lt 10 ]; do
        kill -0 "$pid" 2>/dev/null || { echo "Stopped."; return 0; }
        sleep 1
        waited=$((waited + 1))
    done
    echo "Still running 10s after SIGTERM; forcing kill -9..."
    kill -9 "$pid" 2>/dev/null
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "✗ Still running (PID $pid); manual intervention required."
        return 1
    fi
    echo "Stopped."
}

do_status() {
    local pid
    pid="$(get_pid)"
    if [ -z "$pid" ]; then
        echo "Status: not running"
        return 1
    fi
    echo "Status: running"
    ps -p "$pid" -o pid,etime,command | tail -n +1
    echo -n "Health check ($HEALTH_URL): "
    curl -s --max-time 5 "$HEALTH_URL" || echo "(no response)"
    echo
    echo "Recent logs:"
    if [ -f "$LOG_FILE" ]; then
        tail -3 "$LOG_FILE"
    else
        echo "($LOG_FILE does not exist; the service may have been started outside this script. Run restart to normalize it.)"
    fi
    return 0
}

case "${1:-}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop && sleep 2 && do_start ;;
    status)  do_status ;;
    logs)    tail -f "$LOG_FILE" ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
