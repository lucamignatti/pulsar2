"""Score the RLBot authoritative-air gate (GGL_NO_AIRSTATE_GATE) from real-game captures.

The gate (RLBotClient::ApplyMirror) forbids the sim mirror from reporting isOnGround=true
while the packet's AirState is Dodging / DoubleJumping / InAir - states that cannot
coexist with wheel contact. A false ground hands the policy the 24-action GROUND table
while the car is airborne, so a press meant as a hop becomes a dodge: the "accidental
flip" the operator sees in game but not in the viewer.

Reports, per capture:
  1. GATE FIRED  - consumed ground flag (g) true while packet says unambiguous air.
                   Must be 0 with the gate ON; is the pre-gate defect rate with it OFF.
  2. ACCIDENTAL FLIPS - jump-press edges taken while the policy believed it was grounded
                   (g==1) that produced a dodge within 8 ticks. This is the failure mode.
                   Reported over believed-ground presses, so it is a RATE, not a count
                   (denominators drift between sessions - see the trap list in CLAUDE.md).
  3. Context: air-state mix and press volume, to confirm the two arms are comparable.

usage: airstate_gate_eval.py <capture.jsonl> [more.jsonl ...]
Pair it with wavedash_meter.py on the same files for the success-rate view.
"""
import json
import sys
import collections

AS = {0: "OnGround", 1: "Jumping", 2: "DoubleJumping", 3: "Dodging", 4: "InAir"}
UNAMBIGUOUS_AIR = (2, 3, 4)  # Dodging / DoubleJumping / InAir


def load(path):
    out = []
    for line in open(path):
        if '"decision"' not in line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "decision" and "pkt_g" in r:
            out.append(r)
    return out


def score(dec):
    n = len(dec)
    fired = sum(1 for d in dec if d["g"] == 1 and d["as"] in UNAMBIGUOUS_AIR)

    # Accidental flips: a rising jump-press edge the policy took believing it was on the
    # ground, which turned into a dodge. Uses the AirState edge (engine truth) rather than
    # our own flip bookkeeping, so it stays independent of the flag under test.
    believed_ground_press = acc_flip = 0
    for i in range(1, n - 8):
        d, prev = dec[i], dec[i - 1]
        if d["act_tuple"][5] != 1 or prev["act_tuple"][5] != 0:
            continue
        if d["g"] != 1:
            continue  # policy believed it was airborne; not the failure mode
        believed_ground_press += 1
        for k in range(i, i + 8):
            if dec[k]["as"] == 3 and dec[max(k - 1, 0)]["as"] != 3:
                acc_flip += 1
                break
    mix = collections.Counter(AS.get(d["as"], d["as"]) for d in dec)
    return dict(n=n, fired=fired, presses=believed_ground_press, flips=acc_flip, mix=mix)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        dec = load(p)
        if not dec:
            print(f"{p.split('/')[-1]:<30} no decision rows with pkt_g (pre-2026-08-26 capture?)")
            continue
        r = score(dec)
        name = p.split("/")[-1]
        print(f"\n{name}  ({r['n']} decisions)")
        print(f"  gate fired (g=ground while packet=air): {r['fired']:>6}"
              f"  ({100 * r['fired'] / r['n']:.2f}% of decisions)"
              f"   <- must be 0 with the gate ON")
        if r["presses"]:
            print(f"  accidental flips: {r['flips']}/{r['presses']} believed-ground presses"
                  f"  ({100 * r['flips'] / r['presses']:.1f}%)")
        else:
            print("  accidental flips: no believed-ground jump presses in capture")
        print(f"  air-state mix: {dict(r['mix'])}")
