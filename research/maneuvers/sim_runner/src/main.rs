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
struct Seg { name: String, dur: i32, st: [f32; 13], acts: Vec<Act> }

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
            "SEG" => segs.push(Seg{ name: p[1].into(), dur: p[2].parse().unwrap(), st: [0.0;13], acts: vec![] }),
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
    eprintln!("parsed {} segments", segs.len());
    let mut arena = Arena::new(GameMode::Soccar);
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let mut out = String::from("seg\ttick\tx\ty\tz\tvx\tvy\tvz\tfx\tfy\tfz\tux\tuy\tuz\tavx\tavy\tavz\tground\tboost\n");
    for s in &segs {
        let mut cs = *arena.get_car_state(car);
        cs.phys.pos = Vec3A::new(s.st[0], s.st[1], s.st[2]);
        cs.phys.rot_mat = rot_from_euler(s.st[3], s.st[4], s.st[5]);
        cs.phys.vel = Vec3A::new(s.st[6], s.st[7], s.st[8]);
        cs.phys.ang_vel = Vec3A::new(s.st[9], s.st[10], s.st[11]);
        cs.boost = s.st[12];
        // clear carried-over flags so each segment is independent
        cs.is_on_ground = s.st[2] < 30.0;
        cs.has_jumped = false; cs.has_double_jumped = false; cs.has_flipped = false;
        cs.is_jumping = false; cs.is_flipping = false; cs.flip_time = 0.0; cs.jump_time = 0.0;
        cs.air_time_since_jump = 0.0; cs.flip_rel_torque = Vec3A::ZERO;
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
            arena.set_car_controls(car, cur);
            arena.step_tick();
            let g = arena.get_car_state(car);
            let (p, v, m, av) = (g.phys.pos, g.phys.vel, g.phys.rot_mat, g.phys.ang_vel);
            out.push_str(&format!(
                "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{}\t{:.2}\n",
                s.name, t, p.x,p.y,p.z, v.x,v.y,v.z,
                m.x_axis.x,m.x_axis.y,m.x_axis.z, m.z_axis.x,m.z_axis.y,m.z_axis.z,
                av.x,av.y,av.z, if g.is_on_ground {1} else {0}, g.boost));
        }
    }
    let path = a.get(3).cloned().unwrap_or_else(|| "sim_maneuvers.tsv".into());
    std::fs::write(&path, out).unwrap();
    eprintln!("wrote {path}");
}
