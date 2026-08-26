"""Decide whether to trust the PACKET or the SIM MIRROR for the flip/jump flags.

ApplyMirror overwrites hasFlipped / hasJumped / hasDoubleJumped with the mirror's
reconstruction, even though the packet delivers all three unmasked. Those three feed
HasFlipOrJump() = isOnGround || (!hasFlipped && !hasDoubleJumped && atsj < MAX_DELAY),
which is BOTH an obs float and the gate on the action mask's jump rows - so a wrong
flag removes the dodge the policy is trying to use. The same pattern on isOnGround was
worth 4.66% of decisions and half the in-game accidental flips.

Needs a capture from the 2026-08-26 client or later (pkt_hf/pkt_hj/pkt_hdj present).

Reports, for each flag:
  1. disagreement rate, and its DIRECTION (mirror-set vs packet-set)
  2. the subset that actually changes HasFlipOrJump - the only ones that cost anything
  3. concentration near jump presses (where chains live)
and for the two unconsumed packet fields (dodge_elapsed / dodge_dir) how far the
mirror's reconstruction (flipTime / flipRelTorque) drifts from them.

A one-directional disagreement concentrated near presses is the signature that the
mirror is wrong (as it was for isOnGround). A symmetric, small, semantic-looking offset
argues the mirror is modelling training semantics and should be left alone - that was
the verdict for airTimeSinceJump, which sits an exact +3 ticks from the packet clock.

usage: flip_flag_eval.py <capture.jsonl> [more.jsonl ...]
"""
import json
import math
import sys

MAX_DELAY = 1.25  # RLConst::DOUBLEJUMP_MAX_DELAY


def load(path):
    out = []
    for line in open(path):
        if '"decision"' not in line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "decision" and "pkt_hf" in r:
            out.append(r)
    return out


def hasflip_or_jump(on_ground, hf, hdj, atsj):
    return bool(on_ground or (not hf and not hdj and atsj < MAX_DELAY))


def report(dec, name):
    n = len(dec)
    print(f"\n=== {name}  ({n} decisions with packet flags) ===")
    if not n:
        return
    for mir_k, pkt_k, label in (("hf", "pkt_hf", "hasFlipped"),
                                ("hj", "pkt_hj", "hasJumped"),
                                ("hdj", "pkt_hdj", "hasDoubleJumped")):
        dis = [d for d in dec if int(d[mir_k]) != int(d[pkt_k])]
        m1 = sum(1 for d in dis if int(d[mir_k]) == 1)  # mirror says set, packet says clear
        p1 = len(dis) - m1
        near = sum(1 for i, d in enumerate(dec)
                   if int(d[mir_k]) != int(d[pkt_k])
                   and any(dec[j]["act_tuple"][5] == 1 for j in range(max(0, i - 12), i + 1)))
        print(f"  {label:<16} disagree {len(dis):>6} ({100*len(dis)/n:5.2f}%)"
              f"  mirror-set/packet-clear {m1}  packet-set/mirror-clear {p1}"
              f"  within 12t of a press: {near}")

    # The only disagreements that cost anything: those that flip HasFlipOrJump.
    flips = 0
    mask_gain = mask_loss = 0
    for d in dec:
        atsj = d.get("atsj", 0.0)
        g = int(d["g"])
        a = hasflip_or_jump(g, int(d["hf"]), int(d["hdj"]), atsj)
        b = hasflip_or_jump(g, int(d["pkt_hf"]), int(d["pkt_hdj"]), atsj)
        if a != b:
            flips += 1
            if b and not a:
                mask_gain += 1   # packet would OFFER the dodge the mirror hid
            else:
                mask_loss += 1
    print(f"  --> HasFlipOrJump would change on {flips} ({100*flips/n:.2f}%): "
          f"packet offers a dodge the mirror hid {mask_gain}, hides one it offered {mask_loss}")

    # Unconsumed packet fields vs the mirror's reconstruction of them.
    de = [d for d in dec if d.get("pkt_de", 0) > 0]
    if de:
        err = sorted(abs(d["pkt_de"] - d.get("ft", 0.0)) for d in de)
        print(f"  dodge_elapsed vs mirror flipTime (n={len(de)}): "
              f"median {err[len(err)//2]*120:.2f} ticks, p90 {err[int(.9*len(err))]*120:.2f}")
    dd = [d for d in dec if d.get("pkt_dd") and any(d["pkt_dd"])]
    if dd:
        ang = []
        for d in dd:
            px, py = d["pkt_dd"]
            mx, my = d.get("mir_frt", [0, 0])
            if (px or py) and (mx or my):
                c = (px*mx + py*my) / (math.hypot(px, py) * math.hypot(mx, my))
                ang.append(math.degrees(math.acos(max(-1, min(1, c)))))
        if ang:
            ang.sort()
            print(f"  dodge_dir vs mirror flipRelTorque (n={len(ang)}): "
                  f"median {ang[len(ang)//2]:.1f} deg, p90 {ang[int(.9*len(ang))]:.1f} deg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        dec = load(p)
        if not dec:
            print(f"\n{p.split('/')[-1]}: no packet-flag rows "
                  f"(capture predates the 2026-08-26 client)")
            continue
        report(dec, p.split("/")[-1])
