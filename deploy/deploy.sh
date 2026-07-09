#!/usr/bin/env bash
# One-click VPS deploy for Secretary v2 (Ubuntu 22.04, 2 CPU / 2GB RAM / 40GB disk).
#
# What it does (idempotent — safe to re-run for upgrades):
#   1. Creates a 2G swapfile + vm.swappiness=10 (subagent CLI bursts need headroom)
#   2. Installs Python 3.12 from the deadsnakes PPA (Ubuntu 22.04 ships 3.10;
#      the codebase uses 3.11+ features such as tomllib)
#   3. Creates/updates secretary_v2/venv and installs requirements.txt
#   4. Copies config.yaml.example -> config.yaml if missing
#   5. Installs the OpenCode CLI (vibe-coding tool for remote maintenance)
#   6. Installs the systemd unit (bounded restarts), logrotate config, and the
#      daily stale-log cleanup timer
#   7. Starts the service only when config.yaml no longer contains placeholders
#
# Usage — as the non-root user that should own the service (sudo required):
#   bash deploy/deploy.sh
#   RUN_TESTS=1 bash deploy/deploy.sh    # also run pytest before starting
#
# The HTTP webhook binds 127.0.0.1:11269 (hard-coded in the app). Expose it
# through Cloudflare Tunnel + Zero Trust; do not open the port in the VPS
# security group.

set -euo pipefail

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEC_DIR="$REPO_DIR/secretary_v2"
TEMPLATE_DIR="$REPO_DIR/deploy/templates"
APP_USER="$(id -un)"
APP_GROUP="$(id -gn)"
APP_HOME="$HOME"
SWAP_FILE="/swapfile"
SWAP_SIZE="2G"

[ -f "$SEC_DIR/main.py" ] || die "Run from the repo checkout: bash deploy/deploy.sh"
command -v systemctl >/dev/null 2>&1 || die "systemd is required"
if [ "$(id -u)" -eq 0 ]; then
    die "Run as a normal user, not root. The agent shells out to coding CLIs and
    should not do that as root. Create one first:
      adduser --disabled-password --gecos '' secretary
      usermod -aG sudo secretary
      chown -R secretary:secretary $REPO_DIR
      su - secretary"
fi
sudo -v || die "sudo access is required"

# --- 1. Swap -----------------------------------------------------------------
if sudo swapon --show | grep -q .; then
    log "Swap already active, skipping"
else
    log "Creating ${SWAP_SIZE} swapfile at ${SWAP_FILE}"
    sudo fallocate -l "$SWAP_SIZE" "$SWAP_FILE"
    sudo chmod 600 "$SWAP_FILE"
    sudo mkswap "$SWAP_FILE"
    sudo swapon "$SWAP_FILE"
    grep -q "^$SWAP_FILE " /etc/fstab || \
        echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi
if [ ! -f /etc/sysctl.d/99-secretary.conf ]; then
    echo "vm.swappiness=10" | sudo tee /etc/sysctl.d/99-secretary.conf >/dev/null
    sudo sysctl -q -p /etc/sysctl.d/99-secretary.conf
fi

# --- 2. Python 3.12 ----------------------------------------------------------
if command -v python3.12 >/dev/null 2>&1; then
    log "python3.12 already installed, skipping"
else
    log "Installing Python 3.12 (deadsnakes PPA)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common curl git logrotate
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12 python3.12-venv
fi

# --- 3. venv + dependencies --------------------------------------------------
if [ ! -x "$SEC_DIR/venv/bin/python" ]; then
    log "Creating virtualenv at secretary_v2/venv"
    python3.12 -m venv "$SEC_DIR/venv"
fi
log "Installing Python dependencies"
"$SEC_DIR/venv/bin/pip" install --quiet --upgrade pip
"$SEC_DIR/venv/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"

# --- 4. Config ---------------------------------------------------------------
if [ ! -f "$SEC_DIR/config.yaml" ]; then
    log "Creating config.yaml from template — edit it before the service can run"
    cp "$SEC_DIR/config.yaml.example" "$SEC_DIR/config.yaml"
    chmod 600 "$SEC_DIR/config.yaml"
fi
CONFIG_READY=1
if grep -q "YOUR_LLM_API_KEY\|YOUR_TELEGRAM_BOT_TOKEN" "$SEC_DIR/config.yaml"; then
    CONFIG_READY=0
fi

# --- 5. OpenCode CLI ---------------------------------------------------------
if command -v opencode >/dev/null 2>&1 || [ -x "$APP_HOME/.opencode/bin/opencode" ]; then
    log "OpenCode already installed, skipping"
else
    log "Installing OpenCode CLI"
    curl -fsSL https://opencode.ai/install | bash
fi

# --- 6. systemd units + logrotate ---------------------------------------------
render() {
    sed -e "s|@APP_DIR@|$REPO_DIR|g" \
        -e "s|@APP_USER@|$APP_USER|g" \
        -e "s|@APP_GROUP@|$APP_GROUP|g" \
        -e "s|@APP_HOME@|$APP_HOME|g" \
        "$1" | sudo tee "$2" >/dev/null
}
log "Installing systemd units and logrotate config"
render "$TEMPLATE_DIR/secretary.service"          /etc/systemd/system/secretary.service
render "$TEMPLATE_DIR/secretary-logclean.service" /etc/systemd/system/secretary-logclean.service
render "$TEMPLATE_DIR/secretary-logclean.timer"   /etc/systemd/system/secretary-logclean.timer
render "$TEMPLATE_DIR/logrotate-secretary"        /etc/logrotate.d/secretary
sudo systemctl daemon-reload
sudo systemctl enable secretary.service secretary-logclean.timer >/dev/null
sudo systemctl start secretary-logclean.timer

# --- 7. Tests (optional) -----------------------------------------------------
if [ "${RUN_TESTS:-0}" = "1" ]; then
    log "Running test suite"
    (cd "$SEC_DIR" && ./venv/bin/python -m pytest tests -q)
fi

# --- 8. Start ----------------------------------------------------------------
if [ "$CONFIG_READY" = "1" ]; then
    log "Starting secretary.service (startup self-test takes ~40s)"
    sudo systemctl restart secretary.service
    sleep 5
    sudo systemctl --no-pager status secretary.service || true
else
    warn "config.yaml still contains placeholders — service NOT started.
    Edit $SEC_DIR/config.yaml then run: sudo systemctl start secretary"
fi

log "Done. Cheat sheet:"
cat <<EOF
  status      sudo systemctl status secretary
  logs        tail -f $SEC_DIR/logs/secretary_v2.log
  restart     sudo systemctl restart secretary
  after 5 failed starts in 10min systemd gives up; recover with:
              sudo systemctl reset-failed secretary && sudo systemctl start secretary
  webhook     bound to 127.0.0.1:11269 — point cloudflared at http://127.0.0.1:11269
              (tighten the VPS security group to SSH only; Cloudflare Tunnel is outbound)
  opencode    run inside tmux for remote vibe coding:
              cd $REPO_DIR && opencode
  NOTE        prefer systemctl over manage.sh on this host; manage.sh stop kills
              the process but systemd will restart it, which is confusing.
EOF
