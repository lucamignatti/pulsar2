#!/usr/bin/env python3
"""Pulsar remote trainer dashboard.

Stdlib-only web dashboard for managing the GigaLearn trainer from a phone or
laptop while away. Binds to 127.0.0.1 ONLY and is meant to be exposed through
`tailscale serve` (HTTPS, tailnet-only) — never expose this port directly.

Security model, in layers:
  1. Network: reachable only over the tailnet (WireGuard). No public port.
  2. Actions: every mutating action requires a PIN (pbkdf2-hashed at rest,
     constant-time compare, lockout after repeated failures).
  3. Surface: actions are a fixed allowlist of `trainerctl` argv invocations.
     No shell, no user-supplied arguments, single mutating job at a time.
  4. Audit: every action attempt is logged with the Tailscale identity that
     `tailscale serve` injects.

Run under systemd --user (see tools/remote/setup_remote.sh).
"""

import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import trainer_report

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINERCTL = REPO_ROOT / "tools" / "trainerctl"
LOG_DIR = Path(os.environ.get("RUN_TRAINER_LOG_DIR", REPO_ROOT / "run_logs"))
BUILD_DIR = Path(os.environ.get("TRAINERCTL_BUILD_DIR", REPO_ROOT / "build"))

CONF_DIR = Path.home() / ".config" / "pulsar-remote"
STATE_DIR = Path.home() / ".local" / "state" / "pulsar-remote"
PIN_FILE = CONF_DIR / "pin.json"
CONFIG_FILE = CONF_DIR / "config.json"
AUDIT_FILE = STATE_DIR / "audit.log"
LOCKOUT_FILE = STATE_DIR / "lockout.json"

BIND = ("127.0.0.1", int(os.environ.get("DASHBOARD_PORT", "8500")))

# Version handshake with dashboard.html. The HTML is read from DISK per request while
# this process keeps whatever code it loaded at start - so after a git pull the page
# can be newer than the service and API routes it expects may not exist (bit us live:
# /api/golden 404'd from a stale service and the card hung). Bump BOTH this constant
# and EXPECT_DASH_VERSION in dashboard.html whenever the API surface changes; the page
# shows a "restart the dashboard" banner on mismatch.
DASH_VERSION = "2026-07-25.1"

MAX_PIN_FAILURES = 5
LOCKOUT_WINDOW_SECS = 15 * 60

# action name -> (argv, description). Fixed allowlist; nothing user-supplied
# ever reaches an argv.
ACTIONS = {
    "start":         ([str(TRAINERCTL), "start"], "Start trainer"),
    "stop":          ([str(TRAINERCTL), "stop"], "Stop trainer"),
    "restart":       ([str(TRAINERCTL), "restart"], "Restart trainer"),
    "update":        ([str(TRAINERCTL), "update"], "Git pull + rebuild + restart"),
    "build":         ([str(TRAINERCTL), "build"], "Rebuild only"),
    "viz_start":     ([str(TRAINERCTL), "viz", "start"], "Start visualizer"),
    "viz_stop":      ([str(TRAINERCTL), "viz", "stop"], "Stop visualizer"),
    # Mode switch for the single viewer arena (restarts the render unit in place —
    # one viz at a time is guaranteed by the unit being a singleton)
    "viz_mode_1v1":  ([str(TRAINERCTL), "viz", "mode", "1v1"], "Viz mode 1v1"),
    "viz_mode_2v2":  ([str(TRAINERCTL), "viz", "mode", "2v2"], "Viz mode 2v2"),
    "viz_mode_3v3":  ([str(TRAINERCTL), "viz", "mode", "3v3"], "Viz mode 3v3"),
    "check_updates": (["git", "-C", str(REPO_ROOT), "fetch", "origin", "--prune"],
                      "Fetch origin (no merge)"),
    "dashboard_restart": (["systemctl", "--user", "restart", "pulsar-dashboard.service"],
                          "Restart this dashboard"),
}

# restore_golden is the one action with a parameter (which golden checkpoint). It keeps
# the allowlist discipline anyway: the target must match this strict pattern AND name an
# existing best_r* directory in CKPT_DIR - i.e. the argv argument is validated against an
# enumerated set the trainer itself maintains, never free-form user input.
GOLDEN_RE = re.compile(r"best_r\d+_\d+")

# The checkpoint directory is TRAINERCTL'S to decide, never ours. This used to be a
# second hardcoded default here, and it silently drifted a full run behind: trainerctl
# moved to checkpoints_resid while the dashboard still read checkpoints_5.0v3, so the
# Progress card reported 54.5B steps from a frozen lineage while the live run was at
# 417M, and the golden list offered entries that `trainerctl restore-golden` could not
# find. Ask the CLI, cache briefly, and SHOW the answer on the page so a future
# divergence is visible instead of silent.
_CKPT_CACHE = {"path": None, "at": 0.0}
_CKPT_TTL = 30.0


def ckpt_dir():
    now = time.time()
    if _CKPT_CACHE["path"] is None or now - _CKPT_CACHE["at"] > _CKPT_TTL:
        rc, out = run([str(TRAINERCTL), "ckpt-dir"])
        path = Path(out) if rc == 0 and out else BUILD_DIR / "checkpoints"
        _CKPT_CACHE.update(path=path, at=now)
    return _CKPT_CACHE["path"]


def golden_entries():
    entries = []
    try:
        for p in ckpt_dir().iterdir():
            if p.is_dir() and GOLDEN_RE.fullmatch(p.name):
                rating, _, ts = p.name[len("best_r"):].partition("_")
                entries.append({
                    "name": p.name,
                    "rating": int(rating),
                    "timesteps": int(ts),
                    "date": time.strftime("%Y-%m-%d %H:%M",
                                          time.localtime(p.stat().st_mtime)),
                })
    except Exception:
        pass
    entries.sort(key=lambda e: e["rating"], reverse=True)
    return entries


def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def run(argv, timeout=6):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return -1, str(e)


# ---------------------------------------------------------------- PIN + lockout

_auth_lock = threading.Lock()


def _load_lockout():
    try:
        return json.loads(LOCKOUT_FILE.read_text())
    except Exception:
        return {"failures": []}


def _save_lockout(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCKOUT_FILE.write_text(json.dumps(state))


def lockout_remaining():
    state = _load_lockout()
    now = time.time()
    recent = [t for t in state["failures"] if now - t < LOCKOUT_WINDOW_SECS]
    if len(recent) >= MAX_PIN_FAILURES:
        return int(LOCKOUT_WINDOW_SECS - (now - max(recent)))
    return 0


def verify_pin(pin):
    """Returns (ok, message). Records failures and enforces lockout."""
    with _auth_lock:
        remaining = lockout_remaining()
        if remaining > 0:
            return False, f"locked out for {remaining}s after repeated PIN failures"
        try:
            rec = json.loads(PIN_FILE.read_text())
        except Exception:
            return False, "no PIN configured — run: tools/trainerctl set-pin"
        calc = hashlib.pbkdf2_hmac(
            "sha256", pin.encode(), rec["salt"].encode(), rec["iterations"]
        ).hex()
        if hmac.compare_digest(calc, rec["hash"]):
            state = _load_lockout()
            if state["failures"]:
                _save_lockout({"failures": []})
            return True, "ok"
        state = _load_lockout()
        now = time.time()
        state["failures"] = [t for t in state["failures"] if now - t < LOCKOUT_WINDOW_SECS]
        state["failures"].append(now)
        _save_lockout(state)
        left = MAX_PIN_FAILURES - len(state["failures"])
        return False, f"wrong PIN ({max(left, 0)} attempts left before lockout)"


def audit(handler, action, ok, detail=""):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "ok": ok,
        "who": handler.headers.get("Tailscale-User-Login", "local"),
        "from": handler.client_address[0],
    }
    if detail:
        entry["detail"] = detail
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------------ jobs

class Job:
    def __init__(self, jid, action, argv):
        self.id = jid
        self.action = action
        self.argv = argv
        self.lines = []
        self.done = False
        self.rc = None
        self.started = time.time()
        self.lock = threading.Lock()

    def append(self, line):
        with self.lock:
            self.lines.append(line)

    def snapshot(self, since=0):
        with self.lock:
            return self.lines[since:], len(self.lines), self.done, self.rc


class JobRunner:
    def __init__(self):
        self.jobs = {}
        self.counter = 0
        self.lock = threading.Lock()
        self.active = None  # only one mutating job at a time

    def start(self, action, argv):
        with self.lock:
            if self.active and not self.jobs[self.active].done:
                return None, f"job '{self.jobs[self.active].action}' still running"
            self.counter += 1
            jid = str(self.counter)
            job = Job(jid, action, argv)
            self.jobs[jid] = job
            self.active = jid
            # keep memory bounded
            for old in list(self.jobs)[:-10]:
                del self.jobs[old]
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job, "started"

    def _run(self, job):
        try:
            proc = subprocess.Popen(
                job.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(REPO_ROOT),
            )
            for line in proc.stdout:
                job.append(line.rstrip("\n"))
            job.rc = proc.wait()
        except Exception as e:
            job.append(f"[dashboard] job failed to run: {e}")
            job.rc = -1
        job.append(f"[exit {job.rc}]")
        job.done = True


RUNNER = JobRunner()


# ---------------------------------------------------------------------- status

def systemctl_user(*args, timeout=5):
    return run(["systemctl", "--user", *args], timeout=timeout)


def trainer_status():
    st = {"running": False, "unit": None, "since": None, "pid": None}
    try:
        unit = (LOG_DIR / "trainer.unit").read_text().strip()
    except Exception:
        unit = ""
    if unit:
        rc, _ = systemctl_user("is-active", "--quiet", unit)
        if rc == 0:
            st["running"] = True
            st["unit"] = unit
            rc, out = systemctl_user("show", unit, "-p", "ActiveEnterTimestamp", "--value")
            if rc == 0 and out:
                st["since"] = out
    if not st["running"]:
        try:
            pid = int((LOG_DIR / "trainer.pid").read_text().strip())
            os.kill(pid, 0)
            st["running"] = True
            st["pid"] = pid
        except Exception:
            pass
    return st


def log_status():
    st = {"path": None, "mtime": None, "age_secs": None, "last_line": None}
    latest = LOG_DIR / "latest.log"
    try:
        path = latest.resolve()
        stat = path.stat()
        st["path"] = str(path)
        st["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        st["age_secs"] = int(time.time() - stat.st_mtime)
        with open(path, "rb") as f:
            f.seek(max(0, stat.st_size - 4096))
            tail = f.read().decode("utf-8", "replace").strip().splitlines()
            if tail:
                st["last_line"] = tail[-1][-300:]
    except Exception:
        pass
    return st


# Wrapper crash/restart lines accumulate across the life of one log file, so the
# scan is incremental (see trainer_report.RestartTracker) — logs reach hundreds of
# MB and a 15 s poll cannot re-read them.
RESTARTS = trainer_report.RestartTracker()


def metrics_status():
    """The trainer's newest report block, plus how many times the wrapper has had
    to relaunch it. Everything the console prints, without streaming the log —
    the point being that you can tell whether the run is LEARNING, not just
    whether the process is alive."""
    try:
        path = str((LOG_DIR / "latest.log").resolve())
    except Exception:
        return {"report": None, "restarts": None}
    return {"report": trainer_report.latest_report(path),
            "restarts": RESTARTS.update(path)}


def checkpoint_status():
    ckpts = ckpt_dir()
    st = {"steps": None, "mtime": None, "age_secs": None, "count": 0,
          "dir": ckpts.name}
    try:
        nums = sorted(
            int(p.name) for p in ckpts.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        st["count"] = len(nums)
        if nums:
            newest = ckpts / str(nums[-1])
            mtime = newest.stat().st_mtime
            st["steps"] = nums[-1]
            st["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            st["age_secs"] = int(time.time() - mtime)
    except Exception:
        pass
    return st


def git_status():
    st = {}
    rc, out = run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    st["branch"] = out if rc == 0 else "?"
    rc, out = run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%h %s"])
    st["head"] = out if rc == 0 else "?"
    rc, out = run(["git", "-C", str(REPO_ROOT), "rev-list", "--count",
                   f"HEAD..origin/{st['branch']}"])
    st["behind"] = int(out) if rc == 0 and out.isdigit() else None
    rc, out = run(["git", "-C", str(REPO_ROOT), "rev-list", "--count",
                   f"origin/{st['branch']}..HEAD"])
    st["ahead"] = int(out) if rc == 0 and out.isdigit() else None
    rc, out = run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                   "--untracked-files=no"])
    st["dirty"] = bool(out) if rc == 0 else None
    return st


def gpu_status():
    rc, out = run(["nvidia-smi",
                   "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                   "temperature.gpu,power.draw",
                   "--format=csv,noheader"])
    if rc != 0:
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 6:
        return {"raw": out}
    return {"name": parts[0], "util": parts[1], "mem_used": parts[2],
            "mem_total": parts[3], "temp": parts[4], "power": parts[5]}


def viz_status():
    st = {}
    for key, unit in (("web", "pulsar-viz-web.service"),
                      ("render", "pulsar-viz-render.service")):
        rc, _ = systemctl_user("is-active", "--quiet", unit)
        st[key] = rc == 0
    # Team size of the single viewer arena (trainerctl viz mode); default 1v1
    try:
        mode = (CONF_DIR / "viz_mode").read_text().strip()
    except Exception:
        mode = ""
    st["mode"] = int(mode) if mode in ("1", "2", "3") else 1
    return st


def full_status():
    du = shutil.disk_usage(REPO_ROOT)
    cfg = load_config()
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dash_version": DASH_VERSION,
        "host": os.uname().nodename,
        "trainer": trainer_status(),
        "log": log_status(),
        "metrics": metrics_status(),
        "checkpoint": checkpoint_status(),
        "git": git_status(),
        "gpu": gpu_status(),
        "viz": viz_status(),
        "disk": {"free_gb": round(du.free / 1e9, 1),
                 "total_gb": round(du.total / 1e9, 1)},
        "wandb_url": cfg.get("wandb_url"),
        "locked_out_secs": lockout_remaining(),
    }


# ------------------------------------------------------------------- handlers

HTML_PATH = Path(__file__).parent / "dashboard.html"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PulsarDash"

    def log_message(self, fmt, *args):  # quiet access log
        pass

    # -- helpers ---------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def sse(self, data):
        for line in data.splitlines() or [""]:
            self.wfile.write(f"data: {line}\n".encode("utf-8", "replace"))
        self.wfile.write(b"\n")
        self.wfile.flush()

    def read_body_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > 65536:
                return None
            return json.loads(self.rfile.read(n))
        except Exception:
            return None

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            try:
                body = HTML_PATH.read_bytes()
            except Exception:
                self.send_json({"error": "dashboard.html missing"}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'unsafe-inline'; "
                             "style-src 'unsafe-inline'; connect-src 'self'; "
                             "img-src 'self' data:")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self.send_json(full_status())
        elif path == "/api/logs":
            n = min(int(qs.get("n", ["300"])[0] or 300), 5000)
            rc, out = run([str(TRAINERCTL), "logs", str(n)], timeout=10)
            self.send_json({"ok": rc == 0, "lines": out.splitlines()})
        elif path == "/api/logs/stream":
            self.stream_log()
        elif path == "/api/checkpoints":
            rc, out = run([str(TRAINERCTL), "checkpoints", "15"], timeout=10)
            self.send_json({"ok": rc == 0, "lines": out.splitlines()})
        elif path == "/api/golden":
            self.send_json({"entries": golden_entries()})
        elif path == "/api/audit":
            lines = []
            try:
                lines = AUDIT_FILE.read_text().splitlines()[-50:]
            except Exception:
                pass
            self.send_json({"entries": [json.loads(l) for l in lines]})
        elif path.startswith("/api/job/"):
            m = re.fullmatch(r"/api/job/(\d+)(/stream)?", path)
            if not m:
                self.send_json({"error": "bad job path"}, 404)
                return
            job = RUNNER.jobs.get(m.group(1))
            if not job:
                self.send_json({"error": "no such job"}, 404)
                return
            if m.group(2):
                self.stream_job(job)
            else:
                lines, _, done, rc = job.snapshot()
                self.send_json({"id": job.id, "action": job.action,
                                "lines": lines, "done": done, "rc": rc})
        else:
            self.send_json({"error": "not found"}, 404)

    # -- POST --------------------------------------------------------------
    def do_POST(self):
        if self.path != "/api/action":
            self.send_json({"error": "not found"}, 404)
            return
        body = self.read_body_json()
        if not body:
            self.send_json({"error": "bad request"}, 400)
            return
        action = body.get("action", "")
        if action not in ACTIONS and action != "restore_golden":
            audit(self, action or "?", False, "unknown action")
            self.send_json({"error": "unknown action"}, 400)
            return
        ok, msg = verify_pin(str(body.get("pin", "")))
        if not ok:
            audit(self, action, False, msg)
            self.send_json({"error": msg}, 403)
            return
        if action == "restore_golden":
            # Parameterized action: the target must be an EXISTING golden dir whose
            # name matches the strict pattern (see GOLDEN_RE comment). Re-enumerate at
            # request time so the check races nothing.
            target = str(body.get("target", ""))
            if not GOLDEN_RE.fullmatch(target) \
                    or target not in {e["name"] for e in golden_entries()}:
                audit(self, action, False, f"invalid target {target!r}")
                self.send_json({"error": "unknown golden checkpoint"}, 400)
                return
            argv = [str(TRAINERCTL), "restore-golden", target]
            audit_detail = target
        else:
            argv, _ = ACTIONS[action]
            audit_detail = ""
        job, msg = RUNNER.start(action, argv)
        if job is None:
            audit(self, action, False, msg)
            self.send_json({"error": msg}, 409)
            return
        audit(self, action, True, audit_detail)
        self.send_json({"job": job.id})

    # -- streaming ----------------------------------------------------------
    def stream_job(self, job):
        self.start_sse()
        cursor = 0
        try:
            while True:
                lines, cursor_new, done, rc = job.snapshot(cursor)
                for line in lines:
                    self.sse(line)
                cursor = cursor_new
                if done:
                    self.sse(f"__DONE__ rc={rc}")
                    return
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_log(self):
        self.start_sse()
        try:
            f, path = self._open_latest_log()
            if f is None:
                self.sse("(no log file yet)")
                return
            last_beat = time.time()
            while True:
                chunk = f.readline()
                if chunk:
                    self.sse(chunk.rstrip("\n"))
                    last_beat = time.time()
                    continue
                # no new data: handle rotation (latest.log now points elsewhere
                # or the file shrank), then idle politely
                cur = self._resolve_latest()
                if cur != path or (cur and cur.stat().st_size < f.tell()):
                    f.close()
                    f, path = self._open_latest_log(from_start=True)
                    self.sse(f"--- log rotated -> {path} ---")
                    continue
                if time.time() - last_beat > 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_beat = time.time()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _resolve_latest(self):
        try:
            return (LOG_DIR / "latest.log").resolve()
        except Exception:
            return None

    def _open_latest_log(self, from_start=False):
        path = self._resolve_latest()
        if not path or not path.exists():
            return None, None
        f = open(path, "r", errors="replace")
        if not from_start:
            size = path.stat().st_size
            f.seek(max(0, size - 16384))
            if size > 16384:
                f.readline()  # drop partial first line
        return f, path


def main():
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(BIND, Handler)
    server.daemon_threads = True
    print(f"pulsar dashboard on http://{BIND[0]}:{BIND[1]} (repo {REPO_ROOT})")
    server.serve_forever()


if __name__ == "__main__":
    main()
