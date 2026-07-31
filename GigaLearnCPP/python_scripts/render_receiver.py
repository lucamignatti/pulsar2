import sys
import json
import traceback

import socket
import struct

# =======================
# Example implementation of render receiver, using RocketSimVis
# =======================

# Send to RocketSimVis
UDP_IP = "127.0.0.1"
UDP_PORT = 9273

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

def send_data_to_rsvis(j, gamemode, pulsar=None, actions=None):
    json_out = {}
    json_out["gamemode"] = gamemode
    json_out["ball_phys"] = j['ball']
    json_out["ball_phys"].pop('forward')
    json_out["ball_phys"].pop('right')
    json_out["ball_phys"].pop('up')
    json_out["cars"] = []
    for player in j['players']:
        json_out["cars"].append(player)
    json_out["boost_pad_states"] = j['boost_pads']
    # Viewer control-panel state (transport, rewind range). This payload is rebuilt
    # key by key, so anything not named here is silently dropped on the way out.
    if pulsar is not None:
        json_out["pulsar"] = pulsar
    # The controls each car was actually given this step. Forwarded so the page can
    # tell "the policy chose differently" apart from "the physics drifted" — the two
    # look identical from positions alone.
    if actions is not None:
        json_out["actions"] = actions

    sock.sendto(json.dumps(json_out).encode(), (UDP_IP, UDP_PORT))

def render_state(state_json_str):
    j = json.loads(state_json_str)
    try:
        if 'state' in j:
            send_data_to_rsvis(j['state'], j['gamemode'], j.get('pulsar'), j.get('actions'))
        else:
            send_data_to_rsvis(j)
    except Exception as err:
        print("Exception while sending data:")
        traceback.print_exc()
