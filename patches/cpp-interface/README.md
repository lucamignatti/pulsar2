# cpp-interface local patches

`cpp-interface` is an **upstream third-party submodule** (RLBot v5 C++ interface). It carries
two local changes that exist in no upstream commit and are load-bearing for real-RL play.

They are kept here as a patch file *in addition to* being committed inside the submodule,
because a submodule-local commit lives on no remote: a fresh `git clone --recursive` of this
repo would resolve the gitlink to a commit nobody can fetch. The patch is the durable copy.

- **Base commit**: `8a7faf92d41e964fc3e1a4191a4bd21b433f212b` ("fix windows build", `master`)
- **Patch**: `0001-schema-bump-and-io_uring-buffer-cap.patch`

## What the two hunks do

1. **`CMakeLists.txt` — flatbuffers schema tag `70bcb31e` → `c2648b4`.**
   The stock tag is the June-2025 schema. RLBotServer v5.0.0-rc16 — the core built for the
   current Rocket League build (Bridge v0.9.2) — pins the June-2026 schema. The game's
   April-2026 update broke older cores: a v1.0.0-rc.3 / Bridge-v0.8.2 core drops the game link
   ~1s after map load. For real-RL play the game is the fixed component, so core, `pip rlbot`
   and this schema must all track it together.

2. **`library/Client.cpp` — `PREALLOCATED_BUFFERS` 32 → 4.**
   Each registered buffer is 128 KB of io_uring **pinned** memory charged against
   `RLIMIT_MEMLOCK` (8 MB hard cap on this box, shared per-user, un-raisable without root).
   At 32 that is 4 MB/bot, so only ONE bot could register before `Cannot allocate memory` —
   which broke 2v2/3v3, as every extra Pulsar2 process failed to connect. 4 buffers is
   512 KB/bot (~16 bots fit); a single 15 Hz bot with tiny packets needs nowhere near 32.

## Re-applying after a submodule reset

`git submodule update --init` resets the submodule to the gitlink and destroys uncommitted
work. To restore:

```bash
git -C cpp-interface apply ../patches/cpp-interface/0001-schema-bump-and-io_uring-buffer-cap.patch
```

Verify with `git -C cpp-interface diff --stat` — expect 2 files, 13 insertions, 2 deletions.
