# Remote trainer management

Manage the GigaLearn trainer from anywhere: phone-friendly web dashboard,
SSH escape hatch, git-push-to-update, and a live 3D visualizer — all over
Tailscale (WireGuard). **Nothing is exposed to the public internet**; only
devices logged into your tailnet can reach the box.

```
your phone/laptop ──(WireGuard/tailnet)──▶ this box
   https://<box>.ts.net          → dashboard      (127.0.0.1:8500)
   https://<box>.ts.net:8443     → visualizer     (127.0.0.1:9275)
   wss://<box>.ts.net:10000      → viz websocket  (127.0.0.1:9274)
   ssh luca@<box>.ts.net         → Tailscale SSH
```

## Pieces

| Piece | What it is |
|---|---|
| `tools/trainerctl` | CLI for everything: status/start/stop/restart/logs/update/build/viz/doctor. Use over SSH. |
| `tools/remote/dashboard.py` | Web dashboard (stdlib-only, systemd user service `pulsar-dashboard`). |
| `tools/viz/RocketSimVisWeb/` | Submodule → [lucamignatti/RocketSimVisWeb](https://github.com/lucamignatti/RocketSimVisWeb) `pulsar-patches` (fork of chrisrca's three.js visualizer, patched for headless/localhost/remote use). Fresh clones: `git submodule update --init tools/viz/RocketSimVisWeb`. |
| `tools/remote/setup_remote.sh` | One-time interactive setup (tailscale, HTTPS serve, units, linger, PIN). |

The dashboard shells out to `trainerctl` for every action, so the web and SSH
paths run identical logic.

## Pre-departure checklist

1. `./tools/remote/setup_remote.sh` — installs/joins tailscale, serves the
   dashboard, enables the service, sets your PIN. Re-runnable.
2. Install the Tailscale app on your **phone and laptop**, log into the same
   tailnet.
3. **Push your branch** (`git push`). Remote updates pull from `origin`; local
   commits that were never pushed can't be recovered from the road. The update
   flow also refuses to run over uncommitted local edits — leave the tree clean.
4. Optional: put your wandb run URL in `~/.config/pulsar-remote/config.json`:
   `{"wandb_url": "https://wandb.ai/<you>/gigalearncpp/runs/<id>"}` — the
   dashboard links to it.
5. **Test from the real internet**: turn wifi OFF on your phone (cell data
   only), open the dashboard, stream the log, run `Check updates`, open the
   visualizer. Then `ssh luca@<box>.ts.net` once from the laptop.
6. `tools/trainerctl doctor` — everything should say OK.
7. Physical: BIOS "restore power on AC loss" if you can (power blips), and
   leave the machine on a wired connection if possible.

## Using it from the road

**Dashboard** (`https://<box>.ts.net`): trainer state, log freshness, steps /
checkpoint age, GPU util/VRAM/temp, disk, git sync state. Buttons:

- **Start / Stop / Restart** — wraps `run_trainer.sh`; checkpoints auto-resume,
  wandb run continues (run ID is stored in checkpoint json).
- **Check updates** — `git fetch` only; the "vs origin" card then shows how far
  behind you are.
- **Update** — fetch + **fast-forward-only** pull + rebuild (half the cores) +
  restart onto the new binary. The build runs **before** the trainer is
  touched: a broken build leaves the run untouched on the old binary. Watch the
  job output panel; a full rebuild takes a few minutes.
- **Viz on/off** — starts the visualizer web server plus a CPU-only
  `GGL_RENDER=1` viewer in `build-viz/` that hot-reloads the newest checkpoint
  every few seconds. Costs a little CPU, zero GPU. Turn it off when done.
  These are persistent enabled systemd user units: while "on" they restart on
  crash and come back after a reboot; "off" disables them until you turn them
  back on.
- Every action needs your **PIN** (asked once per page load). 5 wrong PINs =
  15 min lockout. All attempts are audit-logged with your tailscale identity.

**Metrics**: wandb is unaffected by any of this — losses/ratings/etc. live
there as always.

**Typical remote iteration loop**: edit on laptop → commit → `git push` →
dashboard **Update** → watch job output → confirm "running" + log fresh →
check wandb an hour later.

**SSH escape hatch** for anything the dashboard can't do:

```
ssh luca@<box>.ts.net
cd ~/Projects/pulsar2-3.0
tools/trainerctl status | logs 500 | follow | update | viz start | doctor ...
claude          # Claude Code is installed on the box for real surgery
```

## When things go wrong

| Symptom | Fix |
|---|---|
| Dashboard unreachable | `ssh` in → `systemctl --user restart pulsar-dashboard` → `journalctl --user -u pulsar-dashboard -n 50` |
| SSH also dead | Tailscale admin console (login.tailscale.com) shows if the box is offline: power/network outage → phone a friend / smart plug. Everything auto-resumes on boot (see below). |
| Box rebooted | Linger auto-starts the dashboard. Trainer does NOT auto-start (deliberate — a crash-looping run shouldn't fight you): press **Start**; it resumes from the last checkpoint. |
| Trainer crash-looping | `run_trainer.sh` gives up after 5 fast crashes. Log panel shows why. If a bad commit did it: revert/fix on laptop, push, **Update**. |
| Update says "not fast-forward-able" / "dirty" | Someone (you) left local state on the box. SSH in and resolve by hand — deliberately not automated. |
| Wrong-PIN lockout | Wait 15 min, or `ssh` in and `rm ~/.local/state/pulsar-remote/lockout.json`. |
| Forgot PIN | `ssh` in → `tools/trainerctl set-pin`. |
| Viz shows nothing | Both units must run: `tools/trainerctl viz status`. Page connects `wss://…:10000`; give it ~10 s after "Viz on" (viewer loads the checkpoint first). |

## Security notes

- Dashboard, viz, and websocket all bind `127.0.0.1` only; reachability comes
  exclusively from `tailscale serve`, which is **tailnet-only HTTPS** (this is
  not Funnel; nothing is public). Verify with `tailscale serve status`.
- Actions are a fixed allowlist of `trainerctl` invocations — no shell, no
  arguments from the network, one mutating job at a time.
- PIN is pbkdf2(600k)-hashed at rest, compared constant-time, lockout on
  failures — it's a second factor on top of tailnet membership in case a
  logged-in device is lost. Audit trail: `~/.local/state/pulsar-remote/audit.log`.
- Lost/stolen device? Revoke it at login.tailscale.com → Machines → …→ Remove.
  That kills all access instantly.
