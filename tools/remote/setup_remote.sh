#!/usr/bin/env bash
# One-time interactive setup for remote trainer management. Run it yourself in
# a terminal (it uses sudo and opens a browser login for tailscale):
#
#   ./tools/remote/setup_remote.sh
#
# What it does:
#   1. Installs tailscale and joins your tailnet with Tailscale SSH enabled.
#   2. Serves the dashboard + visualizer over tailnet-only HTTPS (no public ports).
#   3. Installs & enables the dashboard as a systemd --user service.
#   4. Enables lingering so user services survive logout/reboot.
#   5. Masks suspend/hibernate so the box doesn't sleep while you're away.
#   6. Sets the dashboard action PIN and runs a final health check.
#
# Safe to re-run; every step is idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

step "1/6 tailscale"
if ! command -v tailscale >/dev/null 2>&1; then
    echo "Installing tailscale (needs sudo)..."
    curl -fsSL https://tailscale.com/install.sh | sh
fi
if ! tailscale status >/dev/null 2>&1; then
    echo "Joining tailnet — a browser login link will be printed. --ssh enables"
    echo "Tailscale SSH so you can shell in from your phone/laptop with no sshd setup."
    sudo tailscale up --ssh
else
    echo "Already joined: $(tailscale status --self=true --peers=false 2>/dev/null | head -1)"
    # Make sure Tailscale SSH is on even if the node was already up.
    sudo tailscale set --ssh || true
fi

step "2/6 tailnet-only HTTPS for dashboard + visualizer"
# tailscale serve exposes localhost ports as HTTPS *inside the tailnet only*.
# 443   -> dashboard          https://<machine>.<tailnet>.ts.net
# 8443  -> visualizer page    https://<machine>.<tailnet>.ts.net:8443/visualizer/visualizer.html
# 10000 -> visualizer websocket (wss)
sudo tailscale serve --bg --https=443   http://127.0.0.1:8500
sudo tailscale serve --bg --https=8443  http://127.0.0.1:9275
sudo tailscale serve --bg --https=10000 http://127.0.0.1:9274
tailscale serve status

step "3/6 dashboard systemd --user service"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/pulsar-dashboard.service" <<EOF
[Unit]
Description=Pulsar remote trainer dashboard
After=network.target

[Service]
ExecStart=/usr/bin/python3 $REPO_ROOT/tools/remote/dashboard.py
WorkingDirectory=$REPO_ROOT
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now pulsar-dashboard.service
systemctl --user status pulsar-dashboard.service --no-pager | head -5

step "4/6 lingering (user services survive logout & start on boot)"
if loginctl show-user "$USER" 2>/dev/null | grep -q Linger=yes; then
    echo "linger already enabled"
else
    sudo loginctl enable-linger "$USER"
    echo "linger enabled"
fi

step "5/6 disable suspend/hibernate while unattended"
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
echo "masked (undo later with: sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target)"

step "6/6 PIN + health check"
if [ ! -s "$HOME/.config/pulsar-remote/pin.json" ]; then
    "$REPO_ROOT/tools/trainerctl" set-pin
else
    echo "PIN already configured (reset anytime with: tools/trainerctl set-pin)"
fi

"$REPO_ROOT/tools/trainerctl" doctor || true

HOSTNAME_TS="$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo '<machine>.<tailnet>.ts.net')"
cat <<EOF

Done. From any device logged into your tailnet:

  dashboard:   https://$HOSTNAME_TS
  visualizer:  https://$HOSTNAME_TS:8443/visualizer/visualizer.html   (after 'Viz on')
  ssh:         ssh $USER@$HOSTNAME_TS

Before you leave, read tools/remote/README.md and walk the pre-departure checklist.
EOF
