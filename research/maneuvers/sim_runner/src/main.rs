// Runs research/maneuvers/maneuvers.txt through RocketSim and logs per-tick car state in
// the SAME format the real-game runner emits, so the two are directly comparable.
use rocketsim::{Arena, CarBodyConfig, CarControls, GameMode, Team};
use glam::{Vec3A, Mat3A};

/// Rotator -> rotation matrix, matching RLBotClient.cpp exactly:
///   Angle(yaw, pitch, roll).ToRotMat() == btMatrix3x3::setEulerYPR(yaw, -pitch, -roll)
///   setEulerYPR(y,p,r) == setEulerZYX(r, p, y)   (fixed-axis X then Y then Z)
fn rot_from_euler(pitch: f32, yaw: f32, roll: f32) -> Mat3A {
    let (ex, ey, ez) = (-roll, -pitch, yaw);
    let (ci, si) = (ex.cos(), ex.sin());
    let (cj, sj) = (ey.cos(), ey.sin());
    let (ch, sh) = (ez.cos(), ez.sin());
    let (cc, cs, sc, ss) = (ci * ch, ci * sh, si * ch, si * sh);
    // Bullet stores rows; glam's from_cols_array_2d takes columns, so transpose by
    // building each COLUMN from the corresponding entry of each row.
    let r0 = [cj * ch, sj * sc - cs, sj * cc + ss];
    let r1 = [cj * sh, sj * ss + cc, sj * cs - sc];
    let r2 = [-sj, cj * si, cj * ci];
    Mat3A::from_cols(
        Vec3A::new(r0[0], r1[0], r2[0]),
        Vec3A::new(r0[1], r1[1], r2[1]),
        Vec3A::new(r0[2], r1[2], r2[2]),
    )
}

fn selftest() {
    let close = |a: Vec3A, b: Vec3A, what: &str| {
        assert!((a - b).length() < 1e-4, "rotator convention wrong for {what}: got {a:?} want {b:?}");
    };
    let m = rot_from_euler(0.0, 0.0, 0.0);
    close(m.x_axis, Vec3A::new(1.0, 0.0, 0.0), "identity forward");
    close(m.z_axis, Vec3A::new(0.0, 0.0, 1.0), "identity up");
    let m = rot_from_euler(0.0, std::f32::consts::FRAC_PI_2, 0.0);
    close(m.x_axis, Vec3A::new(0.0, 1.0, 0.0), "yaw 90 forward");
    let m = rot_from_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    close(m.x_axis, Vec3A::new(0.0, 0.0, 1.0), "pitch 90 forward");
    let m = rot_from_euler(0.0, 0.0, std::f32::consts::PI);
    close(m.z_axis, Vec3A::new(0.0, 0.0, -1.0), "roll 180 up");
    eprintln!("rotator convention self-test: OK");
}

/// Shared with the real-game runner -- both venues must park the ball in the same place.
pub const BALL_PARK: [f32; 3] = [-3500.0, 4800.0, 93.0];

struct Act { at: i32, c: CarControls }
struct Seg { name: String, dur: i32, st: [f32; 13], rot: Option<Mat3A>, acts: Vec<Act> }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: sim_runner <meshdir> <script> [out.tsv]"); std::process::exit(2); }
    selftest();
    rocketsim::init(&a[1], true).unwrap();
    let txt = std::fs::read_to_string(&a[2]).unwrap();
    let mut segs: Vec<Seg> = vec![];
    for line in txt.lines() {
        let l = line.split('#').next().unwrap().trim();
        if l.is_empty() { continue }
        let p: Vec<&str> = l.split_whitespace().collect();
        match p[0] {
            "SEG" => segs.push(Seg{ name: p[1].into(), dur: p[2].parse().unwrap(), st: [0.0;13], rot: None, acts: vec![] }),
            "STATE" => { let s = segs.last_mut().unwrap();
                for i in 0..13 { s.st[i] = p[1+i].parse().unwrap(); } }
            "ACT" => { let s = segs.last_mut().unwrap();
                let f: Vec<f32> = p[2..10].iter().map(|x| x.parse().unwrap()).collect();
                s.acts.push(Act{ at: p[1].parse().unwrap(), c: CarControls{
                    throttle:f[0], steer:f[1], pitch:f[2], yaw:f[3], roll:f[4],
                    jump:f[5]>0.5, boost:f[6]>0.5, handbrake:f[7]>0.5 } }); }
            _ => {}
        }
    }
    // Optional 4th arg: a real capture TSV. The game's state set lands within a
    // speed-scaled tolerance (up to ~179uu for a 2200uu/s spawn), so the real car starts
    // somewhere near the scripted state while the sim starts exactly on it -- a constant
    // offset that is counted as physics error for the whole segment (no_jump_control
    // opened 18.2uu off and held ~16.4uu through its entire ground phase). Seeding from
    // the real t=0 row removes it and isolates dynamics. See SIM2REAL_AUDIT.md S26.
    let mut seed: std::collections::HashMap<String, ([f32; 13], Mat3A)> =
        std::collections::HashMap::new();
    if let Some(path) = a.get(4) {
        let rt = std::fs::read_to_string(path).expect("real capture");
        for line in rt.lines().skip(1) {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() < 19 || c[1] != "0" { continue }
            let g = |i: usize| c[i].parse::<f32>().unwrap_or(0.0);
            // Take the captured basis directly rather than round-tripping through Euler
            // angles: x_axis = forward, z_axis = up, y_axis = up x forward (see selftest).
            let fwd = Vec3A::new(g(8), g(9), g(10)).normalize_or_zero();
            let up = Vec3A::new(g(11), g(12), g(13)).normalize_or_zero();
            let right = up.cross(fwd);
            let rot = Mat3A::from_cols(fwd, right, up);
            seed.insert(c[0].to_string(),
                ([g(2), g(3), g(4), 0.0, 0.0, 0.0, g(5), g(6), g(7),
                  g(14), g(15), g(16), g(18)], rot));
        }
        eprintln!("seeded {} segments from {}", seed.len(), path);
    }
    for sg in segs.iter_mut() {
        if let Some((v, rot)) = seed.get(&sg.name) { sg.st = *v; sg.rot = Some(*rot); }
    }
    eprintln!("parsed {} segments", segs.len());
    let mut arena = Arena::new(GameMode::Soccar);
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let mut out = String::from("seg\ttick\tx\ty\tz\tvx\tvy\tvz\tfx\tfy\tfz\tux\tuy\tuz\tavx\tavy\tavz\tground\tboost\n");
    for s in &segs {
        // GROUND-PARK RESET, and it must be PHYSICAL, not a flag poke.
        //
        // This used to zero has_jumped/has_flipped/... directly. The real game cannot do
        // that: a state set CARRIES jump/flip state, so over there a segment inherited
        // whatever the previous one left. `stall` followed `speed_flip` and inherited a
        // spent flip (real car could not jump at all); `corner_flip_into` followed a
        // segment ending on the ground and took a +280.8 uu/s jump impulse in mid-air.
        // The sim's free reset hid that entirely and nearly justified a bogus has_jumped
        // patch to RocketSim. Park on the floor and let GROUND CONTACT clear the flags,
        // exactly as the real runner now does, then carry the result through the teleport.
        {
            let mut park = *arena.get_car_state(car);
            park.phys.pos = Vec3A::new(0.0, -4600.0, 17.0);
            park.phys.rot_mat = rot_from_euler(0.0, 1.5708, 0.0);
            park.phys.vel = Vec3A::ZERO;
            park.phys.ang_vel = Vec3A::ZERO;
            park.boost = 100.0;
            arena.set_car_state(car, park);
            arena.set_car_controls(car, CarControls::default());
            for _ in 0..12 { arena.step_tick(); }
        }
        // Preserve the flags the park produced; override only the physical state.
        let mut cs = *arena.get_car_state(car);
        cs.phys.pos = Vec3A::new(s.st[0], s.st[1], s.st[2]);
        cs.phys.rot_mat = s.rot.unwrap_or_else(|| rot_from_euler(s.st[3], s.st[4], s.st[5]));
        cs.phys.vel = Vec3A::new(s.st[6], s.st[7], s.st[8]);
        cs.phys.ang_vel = Vec3A::new(s.st[9], s.st[10], s.st[11]);
        cs.boost = s.st[12];
        cs.is_on_ground = s.st[2] < 30.0;
        arena.set_car_state(car, cs);
        // Park the ball in a far corner every segment. The default arena puts it at
        // kickoff (0,0,93) with radius 91, so its TOP is z~184 -- the drop segments were
        // landing on the ball, not the floor. The real-game runner parks it identically.
        {
            let mut bs = *arena.get_ball_state();
            bs.phys.pos = Vec3A::new(BALL_PARK[0], BALL_PARK[1], BALL_PARK[2]);
            bs.phys.vel = Vec3A::ZERO;
            bs.phys.ang_vel = Vec3A::ZERO;
            arena.set_ball_state(bs);
        }
        let mut cur = CarControls::default();
        for t in 0..s.dur {
            for act in &s.acts { if act.at == t { cur = act.c; } }
            // LOG BEFORE STEPPING. The real runner logs the packet it received and only
            // then sets controls, so its row for tick t is the state at the START of t.
            // This loop used to step first, so every sim row was one tick ahead -- at
            // supersonic that is 18.3 uu of pure offset counted as physics error on every
            // sample. Confirmed by scanning the alignment: seeded sim[t] vs real[t+1]
            // scored 1161.4 against 1451.0 at t+0. See SIM2REAL_AUDIT.md S26.
            let g = *arena.get_car_state(car);
            let g = &g;
            let (p, v, m, av) = (g.phys.pos, g.phys.vel, g.phys.rot_mat, g.phys.ang_vel);
            out.push_str(&format!(
                "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{}\t{:.2}\n",
                s.name, t, p.x,p.y,p.z, v.x,v.y,v.z,
                m.x_axis.x,m.x_axis.y,m.x_axis.z, m.z_axis.x,m.z_axis.y,m.z_axis.z,
                av.x,av.y,av.z, if g.is_on_ground {1} else {0}, g.boost));
            if std::env::var("DBG_SEG").map(|v| v == s.name).unwrap_or(false) {
                eprintln!("t={:3} z={:7.2} gnd={} hasJmp={} isJmp={} jt={:.3} atsj={:.3} hasFlip={} isFlip={} ft={:.3} jumpIn={}",
                    t, g.phys.pos.z, g.is_on_ground as u8, g.has_jumped as u8, g.is_jumping as u8,
                    g.jump_time, g.air_time_since_jump, g.has_flipped as u8, g.is_flipping as u8,
                    g.flip_time, cur.jump as u8);
            }
            arena.set_car_controls(car, cur);
            arena.step_tick();
        }
    }
    let path = a.get(3).cloned().unwrap_or_else(|| "sim_maneuvers.tsv".into());
    std::fs::write(&path, out).unwrap();
    eprintln!("wrote {path}");
}
