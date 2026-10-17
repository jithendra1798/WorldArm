"""Chapter 2, rung C: armfarm05 throws the ball, armfarm03 catches it in a bowl. One drill per piece.

Physics, no learning: the arm's own kinematics and the ball's flight say where a throw lands; three calibrated
constants (speed factor, release delay, where the arms stand) make it match. First prove the throw repeatable
at one setting, then calibrate, then catch. Start near (--distance 500: toward armfarm03) and work out.

  python toss.py probe                  no motion: joints, the arm's own kinematics, our model checked against it
  python toss.py plan --design          offline: search a wind-up pose and swing for the target (saved)
  python toss.py plan                   offline: the throw: joint speed, release point, margins, how forgiving
  python toss.py windup                 MOVES armfarm05 slowly to the wind-up pose
  python toss.py dry --repeat 3         MOVES: the swing with no ball; its real start delay, speed and timing
  python toss.py grip --teach           gripper only: the finger opening that just holds the ball
  python toss.py grip --repeat 10       gripper only: how long "open" takes to move the fingers
  python toss.py throw --repeat 5       MOVES: grip, swing, let go; the overhead camera finds where it landed
  python toss.py report                 offline: how repeatable the throws are, against the plan
  python toss.py calibrate              offline: speed factor, release delay, the arms' frames, from the throws
  python toss.py catch --teach          no motion: armfarm03 holding the bowl with the ball in it: how it holds it
  python toss.py catch [--dry]          MOVES BOTH: armfarm03 holds the bowl where the throw will land; the throw

Settings: config/toss.json; logs: runs/<stamp>-toss-<drill>/. Commands that move ask "type go" (--yes skips
it); Ctrl-C or any error stops the arm. Direct joint moves ignore Viam's obstacles: every move is checked
against the table, ceiling and walls in config/toss.json first, along its whole path.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from worldarm.toss.calibrate import PARAMS, fit_constants
from worldarm.toss.config import TossCfg
from worldarm.toss.kinematics import Chain, load_chain, solve_down
from worldarm.toss.landing import ArmsFrame, resting_ball
from worldarm.toss.profile import Trapezoid, fit_trapezoid
from worldarm.toss.sweep import KeepOut, motion_problem, pose_problem
from worldarm.toss.swing import (Forgiveness, ThrowPlan, choose_elevation, design_swing, end_pose, flight_mm,
                                 forgiveness, plan_throw, pose_at)
from worldarm.toss.trigger import Recorder, release

RUNS = Path("runs")
SETTLE_S = 0.5               # after a gripper move
MOVES = {"pick": "armfarm05 finds the ball with its camera and picks it up (slowly)",
         "windup": "armfarm05 to the wind-up pose (slowly)", "dry": "armfarm05 swings, no ball",
         "throw": "armfarm05 swings and throws the ball", "catch": "armfarm03 moves the bowl, then armfarm05 throws"}


def now() -> float:
    return asyncio.get_running_loop().time()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n")


def run_dir(kind: str, out: Path | None = None) -> Path:
    """A fresh log folder (never one another run this second already writes to)."""
    if out:
        out = Path(out)
    else:
        base = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-toss-{kind}"
        out, k = base, 2
        while out.exists():
            out, k = base.with_name(f"{base.name}-{k}"), k + 1
    out.mkdir(parents=True, exist_ok=True)
    return out


def chain_of(cfg: TossCfg) -> Chain:
    if not Path(cfg.kinematics).exists():
        raise SystemExit("No kinematics yet: python toss.py probe (reads the arm's own model)")
    return load_chain(cfg.kinematics)


# ---- the plan ---------------------------------------------------------------------------------------------

def throw_plan(cfg: TossCfg, chain: Chain, target=None, catch_z: float | None = None) -> tuple[ThrowPlan, Forgiveness | None]:
    """The throw for the target (thrower's frame, mm; default cfg.target_mm), checked along the whole swing."""
    spec = cfg.spec()
    if spec is None:
        raise SystemExit("No swing yet: python toss.py plan --design")
    by_heading = target is None and cfg.throw_mm is not None    # the ball's own path, not a point on the table
    target = np.asarray(cfg.target_mm if target is None else target, float)
    cz = cfg.catch_z_mm if catch_z is None else catch_z
    if cfg.accel_deg_s2 > cfg.max_accel_deg_s2:
        return ThrowPlan(False, f"acceleration {cfg.accel_deg_s2:.0f} deg/s^2 is over the cap {cfg.max_accel_deg_s2:.0f}"), None
    args = (cfg.max_speed_deg_s, cfg.release_delay_s, cfg.speed_factor)
    way = {"heading_deg": heading_of(cfg), "throw_mm": cfg.throw_mm} if by_heading else {}
    if cfg.elevation_deg is None:
        plan, f = choose_elevation(chain, spec, target, cz, *args, **way)
        if plan is None:                               # nothing works at any elevation: say why at 45
            plan, f = plan_throw(chain, spec, target, cz, 45.0, *args, **way), None
    else:
        plan = plan_throw(chain, spec, target, cz, cfg.elevation_deg, *args, **way)
        f = forgiveness(chain, spec, plan, cz, cfg.speed_factor) if plan.ok else None
    if plan.ok and path_near_base(plan) < cfg.clear_base_mm:
        plan.ok, plan.reason = False, (f"the ball would pass {path_near_base(plan):.0f} mm from armfarm05's own base "
                                       f"(needs {cfg.clear_base_mm:.0f})")
    if plan.ok and flight_mm(plan) < cfg.min_throw_mm:
        plan.ok, plan.reason = False, (f"the ball would only fly {flight_mm(plan):.0f} mm from the release (a drop, not a "
                                       f"throw; min_throw_mm {cfg.min_throw_mm:.0f}): aim further away")
    if plan.ok:
        why = motion_problem(chain, pose_at(spec, 0.0, plan.base_deg), end_pose(spec, plan.base_deg), spec.tool_mm,
                             KeepOut.of(cfg))
        if why:
            plan.ok, plan.reason = False, f"the swing isn't safe: {why}"
    return plan, f


def heading_of(cfg: TossCfg) -> float:
    """The ball's direction at release (deg, thrower's frame): as set, else toward armfarm03's base."""
    if cfg.heading_deg is not None:
        return cfg.heading_deg
    to = ArmsFrame.of(cfg.thrower_in_catcher).to_thrower((0.0, 0.0))
    return math.degrees(math.atan2(to[1], to[0]))


def path_near_base(plan: ThrowPlan) -> float:
    """How close the ball's path (release to landing, along the table) passes the thrower's base axis (mm)."""
    a, b = plan.release_mm[:2], plan.landing.xy * 1000.0
    ab = b - a
    t = float(np.clip(-(a @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab))


def flange_per_deg(chain: Chain, cfg: TossCfg, plan: ThrowPlan) -> float:
    """The flange's fastest speed along the swing (mm/s) per deg/s of swing speed: the arm's tool-speed cap
    ("Linear speed exceeds limit in ServoJ mode") is on it."""
    spec = cfg.spec()
    qd = np.zeros(chain.n)
    for j, s in zip(spec.joints, spec.signs):
        qd[j] = s
    return max(float(np.linalg.norm(chain.velocity(pose_at(spec, p, plan.base_deg), qd))) for p in
               np.linspace(0.0, spec.amplitude_deg, 41))


def fit_under_cap(cfg: TossCfg, chain: Chain, tcp_limit: float, say=print) -> None:
    """Shorten the throw (cfg.throw_mm) until the swing it needs keeps the flange under tcp_limit."""
    plan, _ = throw_plan(cfg, chain)
    if plan.ok and plan.speed_deg_s * flange_per_deg(chain, cfg, plan) <= tcp_limit:
        return
    lo, hi = 150.0, float(cfg.throw_mm or 600.0)
    for _ in range(25):
        cfg.throw_mm = 0.5 * (lo + hi)
        plan, _ = throw_plan(cfg, chain)
        ok = plan.ok and plan.speed_deg_s * flange_per_deg(chain, cfg, plan) <= tcp_limit
        lo, hi = (cfg.throw_mm, hi) if ok else (lo, cfg.throw_mm)
    cfg.throw_mm = lo
    say(f"held under the arm's speed cap ({tcp_limit:.0f} mm/s at the flange): the ball flies {lo:.0f} mm")


def describe(plan: ThrowPlan, f: Forgiveness | None, cfg: TossCfg, say=print) -> None:
    if not plan.ok:
        say(f"NO THROW: {plan.reason}")
        return
    v, land = plan.velocity_m_s, plan.landing
    el = math.degrees(math.atan2(v[2], float(np.hypot(v[0], v[1]))))
    arms = ArmsFrame.of(cfg.thrower_in_catcher)
    xy = land.xy * 1000.0
    say(f"aim: base joint at {plan.base_deg:.1f} deg; swing joints {[j + 1 for j in cfg.joints]} at "
        f"{plan.speed_deg_s:.1f} deg/s, accelerating at {cfg.accel_deg_s2:.0f} deg/s^2")
    say(f"release {plan.release_progress_deg:.1f} deg into the {cfg.amplitude_deg:.0f} deg swing, {plan.release_time_s:.3f} s "
        f"after it starts moving; full speed from {plan.cruise_deg[0]:.1f} to {plan.cruise_deg[1]:.1f} deg")
    say(f"gripper told to open at {plan.trigger_progress_deg:.1f} deg ({cfg.release_delay_s * 1000:.0f} ms before the "
        f"release point), by the {cfg.trigger} trigger")
    say(f"the ball leaves at ({plan.release_mm[0]:.0f}, {plan.release_mm[1]:.0f}, {plan.release_mm[2]:.0f}) mm at "
        f"{np.linalg.norm(v):.2f} m/s, {el:.1f} deg up, heading {math.degrees(math.atan2(v[1], v[0])):.0f} deg "
        f"(90 = armfarm05's +y); flies {land.t:.2f} s and {flight_mm(plan):.0f} mm, {land.apex_z * 1000:.0f} mm at the top")
    say(f"comes down at ({xy[0]:.0f}, {xy[1]:.0f}) mm in armfarm05's frame = ({arms.to_catcher(xy)[0]:.0f}, "
        f"{arms.to_catcher(xy)[1]:.0f}) mm in armfarm03's")
    if f is not None:
        say(f"how forgiving: released 10 ms late {f.late_10ms_mm:+.0f} mm; 1% faster {f.speed_1pct_mm:+.0f} mm; "
            f"heading 1 deg higher {f.elevation_1deg_mm:+.0f} mm")


def plan_cmd(cfg: TossCfg, chain: Chain, design: bool = False, target=None, catch_z=None, step_deg: float = 15.0,
             say=print) -> dict:
    """Offline. --design first searches wind-up poses for the swing and saves the best."""
    given = target
    target = np.asarray(cfg.target_mm if target is None else target, float)
    cz = cfg.catch_z_mm if catch_z is None else catch_z
    if design:
        rest = list(cfg.start_deg) if cfg.start_deg else [0.0] * chain.n
        say("searching wind-up poses (a minute or so)...")
        way = {"heading_deg": heading_of(cfg), "throw_mm": cfg.throw_mm} if cfg.throw_mm is not None else {}
        found = design_swing(chain, cfg.joints, cfg.amplitude_deg, cfg.accel_deg_s2, target, cz, KeepOut.of(cfg), rest,
                             tuple(cfg.tool_mm), cfg.max_speed_deg_s, step_deg=step_deg, say=say,
                             min_throw_mm=cfg.min_throw_mm, release_delay_s=cfg.release_delay_s, **way)
        found = [r for r in found if path_near_base(r[2]) >= cfg.clear_base_mm and r[2].release_mm[0] > -100.0]
        if not found:
            raise SystemExit("No safe swing reaches that target. Try a nearer target, a larger amplitude_deg, a higher "
                             "accel_deg_s2, or other joints.")
        for score, spec, p, f in found[:3]:
            say(f"  wind-up {[round(q) for q in spec.start_deg]}, joints turn {spec.signs[0]:+d}: {p.speed_deg_s:.0f} deg/s, "
                f"10 ms late {f.late_10ms_mm:+.0f} mm")
        best = found[0][1]
        cfg.start_deg, cfg.signs = [round(float(q), 2) for q in best.start_deg], list(best.signs)
        say(f"saved the first: {cfg.save()}")
    plan, f = throw_plan(cfg, chain, given, catch_z)
    describe(plan, f, cfg, say)
    return {"plan": plan.to_json(), "forgiveness": f.to_json() if f else None}


# ---- arm and gripper ---------------------------------------------------------------------------------------

async def joints_of(arm) -> list[float]:
    return [float(v) for v in (await arm.get_joint_positions()).values]


async def safe_stop(arm) -> None:
    try:
        await arm.stop()
    except Exception:
        pass


async def move_joints(arm, q) -> None:
    from viam.proto.component.arm import JointPositions
    await arm.move_to_joint_positions(JointPositions(values=[float(v) for v in q]))


async def go_slowly(arm, chain: Chain, cfg: TossCfg, q_to, say=print, keep: KeepOut | None = None,
                    where: str = "the next pose") -> None:
    """To q_to at the wind-up speed, if the whole way there is clear; if not, by way of straight up first."""
    keep = keep or KeepOut.of(cfg)
    q_now = await joints_of(arm)
    if float(np.max(np.abs(np.asarray(q_now) - np.asarray(q_to)))) < 0.2:
        return
    why = motion_problem(chain, q_now, q_to, tuple(cfg.tool_mm), keep)
    if why:
        M, _ = chain.frames(q_now)
        up = solve_down(chain, (M[0, 3], M[1, 3], 520.0), q_now, (0.0, 0.0, cfg.gripper_frame_mm))
        if up is None or motion_problem(chain, q_now, up, tuple(cfg.tool_mm), keep) or \
                motion_problem(chain, up, q_to, tuple(cfg.tool_mm), keep):
            raise SystemExit(f"Refused: the way to {where} isn't clear: {why}, and not by way of straight up either. "
                             "Move the arm nearer by hand (or in the Viam app) first.")
        say(f"  (to {where} by way of straight up)")
        await go_slowly(arm, chain, cfg, up, say, keep, "straight up")
        q_now = up
    try:
        await arm.do_command({"clear_error": True})        # a stop the controller made (e.g. self-collision) blocks moves
    except Exception:
        pass
    await arm.do_command({"set_speed": float(cfg.windup_speed_deg_s)})
    await move_joints(arm, q_to)


async def set_accel(arm, accel: float) -> bool:
    try:
        await arm.do_command({"set_acceleration": float(accel)})
        return True
    except Exception:
        return False


async def set_mm(gripper, cfg: TossCfg, mm: float) -> None:
    await gripper.do_command({"set": cfg.counts(mm)})


async def grip_ball(gripper, cfg: TossCfg) -> float | None:
    """Hold the ball barely: the taught opening if there is one, else close gently until the fingers stop on the
    ball (a low torque cap). Returns the finger opening it ended at."""
    if cfg.hold_mm is not None:
        await set_mm(gripper, cfg, cfg.hold_mm)
    else:
        await gripper.do_command({"grab_with_torque": {"position": 0, "speed": int(cfg.grip_speed),
                                                       "torque": int(cfg.grip_torque_pct)}})
    await asyncio.sleep(0.8)
    return await width_mm(gripper, cfg)


def holding(width: float | None, cfg: TossCfg) -> bool:
    """The fingers stopped on the ball short of where they were sent."""
    if width is None:
        return False
    return width > (cfg.hold_mm + 1.0 if cfg.hold_mm is not None else 0.4 * cfg.ball_diameter_mm)


def release_counts(cfg: TossCfg) -> float:
    """How far the fingers open to let go: just past the ball (less travel than fully open: the ball is free sooner)."""
    return cfg.counts(min(cfg.max_open_mm, cfg.pick_ball_height_mm + cfg.free_mm))


async def open_fast(gripper, cfg: TossCfg) -> None:
    """Let go: at the gripper's top speed, just past the ball."""
    try:
        await gripper.do_command({"set_gripper_speed": int(cfg.release_speed)})
    except Exception:
        pass
    await gripper.do_command({"set": float(release_counts(cfg))})


async def width_mm(gripper, cfg: TossCfg) -> float | None:
    try:
        return cfg.mm(float((await gripper.do_command({"get": True})).get("pos")))
    except Exception:
        return None


async def swing(arm, spec, base_deg: float, w: float, accel: float, clock=now, let_go=None, timeout_s: float = 10.0):
    """One swing from the wind-up pose to its end, the joints read throughout; let_go(t_cmd, recorder) runs
    alongside (the release). Returns (command time, recorder, what let_go returned)."""
    await arm.do_command({"set_speed": float(w)})
    await set_accel(arm, accel)
    rec = Recorder(lambda: joints_of(arm), clock)
    stop = asyncio.Event()
    reading = asyncio.ensure_future(rec.run(stop))
    await asyncio.sleep(0.05)                          # a reading or two at rest first
    t_cmd = clock()
    move = asyncio.ensure_future(move_joints(arm, end_pose(spec, base_deg)))
    rel = asyncio.ensure_future(let_go(t_cmd, rec)) if let_go else None
    try:
        await asyncio.wait_for(move, timeout_s)
        sent = None
        if rel is not None:
            done, _ = await asyncio.wait({rel}, timeout=1.0)   # a release that never fired keeps the ball:
            sent = rel.result() if rel in done else None       # never open it at the end of the swing
    finally:
        stop.set()
        for task in (reading, rel, move):
            if task is not None and not task.done():
                task.cancel()
    return t_cmd, rec, sent


# ---- drills: no motion ------------------------------------------------------------------------------------

async def probe(arm, gripper, cfg: TossCfg, out: Path | None = None, say=print) -> dict:
    """Read the arm's joints, pose and kinematics; check our model of the arm against the arm's own answer."""
    from worldarm.cell.geometry import unit
    q = await joints_of(arm)
    pose = await arm.get_end_position()
    kin = await arm.get_kinematics()
    fmt, data = kin[0], kin[1]
    path = Path(cfg.kinematics)
    if fmt == 2:                                        # URDF
        path = path.with_suffix(".urdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else bytes(data))
    cfg.kinematics = str(path)
    chain = load_chain(str(path))
    M, _ = chain.frames(q)
    pos_err = float(np.linalg.norm(M[:3, 3] - np.array([pose.x, pose.y, pose.z])))
    ov = unit((pose.o_x, pose.o_y, pose.o_z))
    dir_err = math.degrees(math.acos(max(-1.0, min(1.0, float(M[:3, 2] @ ov)))))
    replies = {}
    for cmd in ({"set_speed": float(cfg.windup_speed_deg_s)}, {"set_acceleration": float(cfg.accel_deg_s2)}):
        try:
            replies[next(iter(cmd))] = f"ok {await arm.do_command(cmd)}"
        except Exception as e:
            replies[next(iter(cmd))] = f"refused: {e}"
    res = {"joints_deg": q, "end_mm": [pose.x, pose.y, pose.z], "end_ov": [pose.o_x, pose.o_y, pose.o_z, pose.theta],
           "kinematics": str(path), "format": "urdf" if fmt == 2 else "sva", "joints": chain.n,
           "limits_deg": chain.limits_deg, "model_error_mm": round(pos_err, 2), "model_error_deg": round(dir_err, 2),
           "do_command": replies, "gripper_mm": await width_mm(gripper, cfg)}
    cfg.save()
    write_json(run_dir("probe", out) / "probe.json", res)
    say(f"{chain.n} joints at {[round(v, 1) for v in q]} deg; kinematics saved to {path}")
    say(f"our model of the arm vs. the arm: {pos_err:.1f} mm, {dir_err:.1f} deg apart at this pose")
    if pos_err > 2.0 or dir_err > 1.0:
        say("WARNING: the model disagrees with the arm. Don't plan throws until this is understood.")
    for k, v in replies.items():
        say(f"do_command {k}: {v}")
    if not replies.get("set_acceleration", "").startswith("ok"):
        say("the arm won't take an acceleration: set accel_deg_s2 in config/toss.json to what `dry` measures")
    say(f"gripper opening: {res['gripper_mm']} mm")
    return res


def report(cfg: TossCfg, runs: Path | None = None, say=print) -> dict:
    """How repeatable the throws are: per setting, the spread of the landings and how far off the plan."""
    throws = load_throws(runs)
    groups: dict[tuple, list] = {}
    for t in throws:
        if t.get("landing_catcher_mm") is not None:
            groups.setdefault((round(t["speed_deg_s"], 0), round(t["base_deg"], 0), t["trigger"]), []).append(t)
    res = {}
    if not groups:
        say("no measured throws yet: python toss.py throw --repeat 5")
    for (w, base, trig), ts in sorted(groups.items()):
        pts = np.array([t["landing_catcher_mm"] for t in ts], float)
        plan = np.array([t["planned_catcher_mm"] for t in ts], float)
        mean = pts.mean(axis=0)
        d = np.linalg.norm(pts - mean, axis=1)
        sd = pts.std(axis=0, ddof=1) if len(ts) > 1 else np.full(2, math.nan)
        r95 = float(np.percentile(d, 95)) if len(ts) >= 10 else float(2.45 * np.sqrt(np.nanmean(sd ** 2))) if len(ts) > 1 else math.nan
        late = [t["sent_progress_deg"] - t["trigger_deg"] for t in ts if t.get("sent_progress_deg") is not None]
        late_ms = [1000.0 * x / w for x in late] if w else []
        key = f"{w:.0f} deg/s, base {base:.0f}, {trig}"
        res[key] = {"n": len(ts), "mean_mm": mean.round(1).tolist(), "sd_mm": np.round(sd, 1).tolist(),
                    "radius95_mm": round(r95, 1), "off_plan_mm": (mean - plan.mean(axis=0)).round(1).tolist(),
                    "release_ms_sd": round(statistics.stdev(late_ms), 1) if len(late_ms) > 1 else None}
        say(f"{key}: {len(ts)} throws; spread sd ({sd[0]:.0f}, {sd[1]:.0f}) mm, 95% within {r95:.0f} mm; "
            f"{np.linalg.norm(mean - plan.mean(axis=0)):.0f} mm off the plan")
        if res[key]["release_ms_sd"] is not None:
            say(f"  the open command went out within {res[key]['release_ms_sd']:.0f} ms (sd) of its moment")
    return res


def load_throws(runs: Path | None = None) -> list[dict]:
    throws = []
    for f in sorted(Path(runs or RUNS).glob("*-toss-throw*/throws.json")):
        throws += json.loads(f.read_text())
    return throws


def calibrate(cfg: TossCfg, chain: Chain, runs: Path | None = None, fit=PARAMS, save: bool = True, say=print) -> dict:
    """Fit speed factor, release delay and the arms' frames to the measured landings; save them."""
    throws = load_throws(runs)
    prior = {"speed_factor": cfg.speed_factor, "release_delay_s": cfg.release_delay_s, **cfg.thrower_in_catcher}
    cal = fit_constants(chain, throws, prior, tuple(fit))
    if cal.n == 0:
        raise SystemExit("No measured throws yet: python toss.py throw --repeat 5")
    for k in PARAMS:
        sd = cal.sd.get(k)
        say(f"  {k:<16} {cal.values[k]:9.4f}" + (f"  +- {sd:.4f}" if sd is not None else "  (held)"))
    say(f"from {cal.n} throws: the landings now match the model within {cal.rms_mm:.0f} mm (rms)")
    if cal.note:
        say(f"note: {cal.note}")
    if save:
        cfg.speed_factor = round(cal.values["speed_factor"], 4)
        cfg.release_delay_s = round(cal.values["release_delay_s"], 4)
        cfg.thrower_in_catcher = {k: round(cal.values[k], 2) for k in ("x_mm", "y_mm", "yaw_deg")}
        say(f"saved: {cfg.save()}")
    return {"values": cal.values, "sd": cal.sd, "rms_mm": cal.rms_mm, "n": cal.n, "residuals_mm": cal.residuals_mm,
            "note": cal.note}


# ---- drills: the gripper -----------------------------------------------------------------------------------

async def grip(gripper, cfg: TossCfg, teach: bool = False, repeat: int = 0, out: Path | None = None, say=print,
               ask=input, clock=now) -> dict:
    """--teach: close on the ball a millimetre at a time until it just holds. --repeat: how long the open command
    takes to move the fingers (2 mm apart; an upper bound, a reading's round trip included)."""
    res: dict = {}
    if teach:
        await set_mm(gripper, cfg, cfg.ball_diameter_mm + 15.0)
        ask("Hold the ball between the fingers, then press Enter ")
        mm = cfg.ball_diameter_mm + 2.0
        while True:
            await set_mm(gripper, cfg, mm)
            await asyncio.sleep(SETTLE_S)
            if ask(f"fingers at {mm:.0f} mm: let go of the ball. Does it stay? [y = it holds, Enter = 1 mm tighter] "
                   ).strip().lower() == "y":
                break
            mm -= 1.0
            if mm < cfg.ball_diameter_mm - 12.0:
                raise SystemExit("12 mm tighter than the ball and it still doesn't hold: check ball_diameter_mm")
        cfg.hold_mm = mm
        say(f"holds at {mm:.0f} mm; saved: {cfg.save()}")
        res["hold_mm"] = mm
    if repeat:
        lat, acks = [], []
        close_pos = cfg.counts(cfg.pick_ball_height_mm - cfg.squeeze_mm)   # as the pick closes on the ball
        free_at = cfg.pick_ball_height_mm + 3.0              # the fingers are clear of the ball here
        for _ in range(repeat):
            await gripper.do_command({"set": float(close_pos)})  # as the pick closes on the ball
            await asyncio.sleep(SETTLE_S)
            t_send = clock()
            opening = asyncio.ensure_future(open_fast(gripper, cfg))
            while clock() - t_send < 2.0:
                w = await width_mm(gripper, cfg)
                if w is not None and w >= free_at:
                    lat.append(clock() - t_send)
                    break
            await opening
            acks.append(clock() - t_send)
        if not lat:
            raise SystemExit("The fingers never moved after the open command (or the gripper can't be read)")
        res.update({"open_latency_s": lat, "median_s": statistics.median(lat),
                    "sd_s": statistics.stdev(lat) if len(lat) > 1 else 0.0, "command_returned_s": acks})
        say(f"open -> fingers clear of the ball: median {res['median_s'] * 1000:.0f} ms, sd {res['sd_s'] * 1000:.0f} ms "
            f"(over {len(lat)}); the command itself returned after {statistics.median(acks) * 1000:.0f} ms")
        cfg.jaw_s = cfg.release_delay_s = round(res["median_s"], 3)   # both throws lead the release by it
        say(f"the throw now sends the open command {cfg.jaw_s * 1000:.0f} ms early; saved: {cfg.save()}")
        await set_mm(gripper, cfg, cfg.ball_diameter_mm + 15.0)
    write_json(run_dir("grip", out) / "grip.json", res)
    return res


# ---- drills: moving ----------------------------------------------------------------------------------------

async def windup(arm, chain: Chain, cfg: TossCfg, target=None, say=print) -> list[float]:
    plan, _ = throw_plan(cfg, chain, target)
    if not plan.ok:
        raise SystemExit(f"No throw planned: {plan.reason}")
    q = pose_at(cfg.spec(), 0.0, plan.base_deg)
    try:
        await go_slowly(arm, chain, cfg, q, say)
    finally:
        await safe_stop(arm)
    say(f"at the wind-up pose {[round(float(v), 1) for v in q]}")
    return list(q)


async def dry(arm, chain: Chain, cfg: TossCfg, repeat: int = 3, scale: float = 1.0, target=None, out: Path | None = None,
              say=print, clock=now) -> dict:
    """The swing with no ball, `repeat` times: when it really starts after the command, how fast it really
    cruises, how hard it accelerates, and how much that varies (the timed trigger's jitter)."""
    plan, _ = throw_plan(cfg, chain, target)
    if not plan.ok:
        raise SystemExit(f"No throw planned: {plan.reason}")
    spec = cfg.spec()
    w, a = plan.speed_deg_s * scale, spec.accel_deg_s2
    nominal = Trapezoid(spec.amplitude_deg, w, a)
    out = run_dir("dry", out)
    swings = []
    try:
        for i in range(repeat):
            await go_slowly(arm, chain, cfg, pose_at(spec, 0.0, plan.base_deg), say)
            t_cmd, rec, _ = await swing(arm, spec, plan.base_deg, w, a, clock, timeout_s=nominal.duration + 5.0)
            t, p = rec.progress(spec)
            write_json(out / f"swing{i + 1}.json", {"t": (t - t_cmd).tolist(), "progress": p.tolist()})
            prof, sd, rms = fit_trapezoid(t - t_cmd, p, spec.amplitude_deg, w, a)
            rtt = [r[2] for r in rec.readings]
            s = {"start_latency_s": prof.t0, "w_deg_s": prof.w, "accel_deg_s2": prof.a, "fit_rms_deg": rms,
                 "to_trigger_s": prof.time_at(plan.trigger_progress_deg), "readings": len(rtt),
                 "reading_ms": 1000.0 * statistics.median(rtt) if rtt else None, "errors": rec.errors,
                 "t": (t - t_cmd).tolist(), "progress": p.tolist()}
            swings.append(s)
            say(f"swing {i + 1}: started {prof.t0 * 1000:.0f} ms after the command, cruised at {prof.w:.1f} deg/s "
                f"(asked {w:.1f}), accelerated at {prof.a:.0f} deg/s^2; {len(rtt)} readings, "
                f"{s['reading_ms'] or 0:.0f} ms each")
    finally:
        await safe_stop(arm)
    lat = [s["start_latency_s"] for s in swings]
    trig = [s["to_trigger_s"] for s in swings]
    res = {"plan": plan.to_json(), "scale": scale, "swings": swings,
           "start_latency_s": statistics.mean(lat), "start_latency_sd_s": statistics.stdev(lat) if len(lat) > 1 else None,
           "trigger_time_sd_s": statistics.stdev(trig) if len(trig) > 1 else None,
           "speed_ratio": statistics.mean(s["w_deg_s"] for s in swings) / w,
           "accel_deg_s2": statistics.mean(s["accel_deg_s2"] for s in swings)}
    cfg.start_latency_s = round(res["start_latency_s"], 4)
    say(f"start delay {res['start_latency_s'] * 1000:.0f} ms" + (f" +- {res['start_latency_sd_s'] * 1000:.0f} ms (sd)"
        if res["start_latency_sd_s"] is not None else "") + f"; it cruised at {res['speed_ratio']:.3f} x the speed asked")
    if abs(res["accel_deg_s2"] - a) > 0.15 * a:
        say(f"it accelerated at {res['accel_deg_s2']:.0f}, not {a:.0f} deg/s^2: set accel_deg_s2 to what it does and re-plan")
    if res["trigger_time_sd_s"] is not None:
        rms = statistics.median([s["reading_ms"] for s in swings if s["reading_ms"]] or [0.0])
        say(f"timed trigger would vary {res['trigger_time_sd_s'] * 1000:.0f} ms (sd); joint readings come every "
            f"{rms:.0f} ms for the predicted one")
    say(f"saved the start delay: {cfg.save()}")
    write_json(out / "dry.json", res)
    return res


# ---- picking the ball up: Waj's armfarm05 method (throw/real_throw_armfarm05.py) on our kinematics -----------

CAM_IN_FLANGE = np.array([83.0, -14.0, 18.0])            # the wrist camera's frame in the flange frame (machine config)


def decode_depth(data: bytes) -> np.ndarray:
    """Viam image/vnd.viam.dep -> (H, W) depth in mm."""
    w, h = int.from_bytes(data[8:16], "big"), int.from_bytes(data[16:24], "big")
    return np.frombuffer(data[24:], dtype=">u2").reshape(h, w).astype(np.float32)


def detect_balls(bgr: np.ndarray, depth: np.ndarray, intr, cam_pos, cam_R, cfg: TossCfg) -> list[dict]:
    """Dark round foam blobs on the table, armfarm05 world mm, darkest first (Waj's WorldArm method, OpenCV)."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    mask = ((sat < 110) & (val < 100)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    found = []
    for i in range(1, n):
        x0, y0, w, h, area = (int(v) for v in stats[i])
        if not 400 <= area <= 25000 or max(h, w) / max(1, min(h, w)) > 1.8 or area / (h * w) < 0.5:
            continue
        if x0 <= 1 or y0 <= 1 or x0 + w >= W - 1 or y0 + h >= H - 1:
            continue                                       # cut off by the image edge (a fingertip shows that way)
        region = lab[y0:y0 + h, x0:x0 + w] == i
        vals = val[y0:y0 + h, x0:x0 + w][region].astype(np.float32)
        if np.mean(vals < 70) < 0.3:
            continue
        ys, xs = np.nonzero(region)
        weight = 255.0 - vals                              # aim at the dark foam, not highlights
        u = x0 + float((weight * xs).sum() / weight.sum())
        v = y0 + float((weight * ys).sum() / weight.sum())
        d = depth[y0:y0 + h, x0:x0 + w][region]
        d = d[d > 0]
        if d.size < 5:
            continue
        ray = cam_R @ np.array([(u - intr.center_x_px) / intr.focal_x_px, (v - intr.center_y_px) / intr.focal_y_px, 1.0])
        if ray[2] > -0.1:
            continue
        dist = float(np.median(d))
        r_px = intr.focal_x_px * (cfg.pick_ball_height_mm / 2.0) / max(dist, 1.0)   # a ball's radius at that distance
        if not 0.3 <= area / (math.pi * r_px * r_px) <= 3.0:
            continue                                       # too small or too big to be the ball (a speck, a seam)
        top = cam_pos + ray * dist
        height = top[2] - cfg.pick_table_z_mm
        if not -15 <= height <= 130:
            continue
        if height < 8:                                     # soft foam reads nearly flat in depth: assume the size
            height = cfg.pick_ball_height_mm
            cz = cfg.pick_table_z_mm + height / 2.0
        else:                                              # the depth saw its top: the centre is a radius below,
            cz = top[2] - cfg.pick_ball_height_mm / 2.0    # whatever the table's exact height
        c = cam_pos + ray * ((cz - cam_pos[2]) / ray[2])
        found.append({"x": float(c[0]), "y": float(c[1]), "cz": float(cz), "height": float(height), "u": round(u),
                      "v": round(v), "area": area, "darkness": float(np.median(vals))})
    return sorted(found, key=lambda b: b["darkness"])


async def wrist_look(machine, camera: str, cfg: TossCfg, chain: Chain, q, save: Path | None = None) -> list[dict]:
    """Balls the wrist camera sees from joints q (colour + aligned depth); the picture is saved with them marked."""
    import cv2
    from viam.components.camera import Camera
    from worldarm.cell.geometry import rot_from_ov
    from worldarm.perception import decode_color
    cam = Camera.from_robot(machine, camera)
    images, _ = await cam.get_images()
    by = {im.name: im for im in images}
    if "color" not in by or "depth" not in by:
        raise SystemExit("armfarm05's wrist camera must return 'color' and 'depth'")
    bgr, depth = decode_color(by["color"].data), decode_depth(by["depth"].data)
    intr = (await cam.get_properties()).intrinsic_parameters
    M, _ = chain.frames(q)                                 # camera pose from our model (matches the arm to 0.0 mm)
    cam_pos = (M @ np.r_[CAM_IN_FLANGE, 1.0])[:3]
    cam_R = M[:3, :3] @ rot_from_ov(-0.030391, -0.003538, 0.999532, -97.731173)
    balls = detect_balls(bgr, depth, intr, cam_pos, cam_R, cfg)
    if save is not None:
        for b in balls:
            cv2.circle(bgr, (b["u"], b["v"]), 25, (0, 0, 255), 3)
        cv2.imwrite(str(save), bgr)
    return balls


def reachable(x: float, y: float) -> bool:
    return 150 <= math.hypot(x, y) <= 720 and x <= 640 and y >= -400


async def line_to(arm, chain: Chain, cfg: TossCfg, gripper_xyz, keep: KeepOut, label: str, step_mm: float = 20.0) -> None:
    """The gripper frame straight to gripper_xyz, pointing down: IK every step_mm, each pose checked, then moved
    through slowly (a straight line near the ball; a joint move would curve)."""
    from viam.proto.component.arm import JointPositions
    tool = (0.0, 0.0, cfg.gripper_frame_mm)
    q = np.asarray(await joints_of(arm), float)
    M, _ = chain.frames(q)
    p0, p1 = (M @ np.r_[tool, 1.0])[:3], np.asarray(gripper_xyz, float)
    if M[2, 2] > -0.99:                                    # not pointing down yet: turn it down from where it is
        q1 = solve_down(chain, p1, q, tool)
        if q1 is None:
            raise SystemExit(f"{label}: no way to point the gripper down at {np.round(p1).tolist()}")
        await go_slowly(arm, chain, cfg, q1, keep=keep)
        return
    path = []
    for i in range(1, max(1, int(math.ceil(np.linalg.norm(p1 - p0) / step_mm))) + 1):
        pt = p0 + (p1 - p0) * i / max(1, int(math.ceil(np.linalg.norm(p1 - p0) / step_mm)))
        qi = solve_down(chain, pt, q, tool)
        if qi is None or float(np.max(np.abs(qi - q))) > 12.0:
            raise SystemExit(f"{label}: the straight path breaks near {np.round(pt).tolist()}")
        why = pose_problem(chain, qi, tuple(cfg.tool_mm), keep)
        if why:
            raise SystemExit(f"{label}: {why} near {np.round(pt).tolist()}")
        path.append(qi)
        q = qi
    await arm.do_command({"set_speed": float(cfg.windup_speed_deg_s)})
    positions = [JointPositions(values=[float(v) for v in qi]) for qi in path]
    try:
        await arm.move_through_joint_positions(positions)  # one smooth move along the line (Waj's run_path)
    except (NotImplementedError, AttributeError):
        for jp in positions:
            await arm.move_to_joint_positions(jp)


async def pick_ball(arm, gripper, machine, chain: Chain, cfg: TossCfg, camera: str = "cam", say=print,
                    seen: list | None = None, grab: bool = True) -> bool:
    """Find the ball from the scan poses, look again from straight above it, go down and close on it (67% of the
    travel), lift. True if the fingers stopped on the ball."""
    reaching = dataclasses.replace(KeepOut.of(cfg), reach_mm=180.0,     # reaching down near the base folds it more,
                                   table_z_mm=cfg.pick_table_z_mm)       # and the real table top is what matters here
    near_table = dataclasses.replace(reaching, clearance_mm=cfg.pick_clearance_mm)
    tool = (0.0, 0.0, cfg.gripper_frame_mm)
    out = run_dir("pick")
    close_pos = cfg.counts(cfg.pick_ball_height_mm - cfg.squeeze_mm)   # barely under the ball's size
    try:
        await gripper.do_command({"set": float(cfg.full_open_counts)})
        est = None
        say("1. looking for the ball")
        for i, pose in enumerate(cfg.pick_scan):
            q = solve_down(chain, pose, await joints_of(arm), tool)
            if q is None:
                continue
            await go_slowly(arm, chain, cfg, q, say, keep=reaching, where=f"scan pose {i + 1}")
            balls = [b for b in await wrist_look(machine, camera, cfg, chain, await joints_of(arm), out / f"scan{i + 1}.jpg")
                     if reachable(b["x"], b["y"])]
            say(f"scan {i + 1}: {len(balls)} ball(s)")
            if balls:
                est = balls[0]
                break
        if est is None:
            say(f"no ball seen from any scan pose (pictures in {out})")
            return False
        q = await joints_of(arm)
        M, _ = chain.frames(q)
        off = M[:3, :3] @ (CAM_IN_FLANGE - np.array([0.0, 0.0, cfg.gripper_frame_mm]))   # camera minus gripper frame
        above = np.array([est["x"] - off[0], est["y"] - off[1], cfg.pick_look_z_mm])
        if reachable(above[0], above[1]):
            try:                                           # a closer look: a refinement, skipped if it can't be safe
                await line_to(arm, chain, cfg, above, reaching, "look above the ball")
                near = [b for b in await wrist_look(machine, camera, cfg, chain, await joints_of(arm), out / "look.jpg")
                        if math.hypot(b["x"] - est["x"], b["y"] - est["y"]) < 80]
                if near:
                    est = min(near, key=lambda b: math.hypot(b["x"] - est["x"], b["y"] - est["y"]))
            except SystemExit as e:
                say(f"no close look ({e}); using the first fix")
        say(f"2. the ball is at ({est['x']:.0f}, {est['y']:.0f}) mm, {est['height']:.0f} mm tall")
        if seen is not None:
            seen.append((est["x"], est["y"]))               # where it lay: the last throw's landing
        if not grab:
            return True
        cz = est.get("cz", cfg.pick_table_z_mm + est["height"] / 2.0)
        gz = max(cz - cfg.grasp_lower_mm + cfg.grasp_raise_mm + cfg.tcp_to_tip_mm,
                 cfg.pick_table_z_mm + cfg.min_tcp_above_table_mm)
        hover = max(cfg.pick_hover_mm, est["height"] + 40.0)
        say("3. going over it, then straight down around it")
        await line_to(arm, chain, cfg, (est["x"], est["y"], gz + hover), reaching, "above the ball")
        await line_to(arm, chain, cfg, (est["x"], est["y"], gz), near_table, "down to the ball")
        say(f"4. closing the fingers to {cfg.pick_ball_height_mm - cfg.squeeze_mm:.0f} mm on it (barely holding)")
        await gripper.do_command({"set": float(close_pos)})  # barely: just under the ball's size, no hard press
        await asyncio.sleep(0.6)
        pos = (await gripper.do_command({"get": True})).get("pos")
        width = None if pos is None else cfg.mm(float(pos))
        say("5. lifting it, and checking the table")
        await line_to(arm, chain, cfg, (est["x"], est["y"], max(gz + 150.0, 320.0)), near_table, "lift")
        left = [b for b in await wrist_look(machine, camera, cfg, chain, await joints_of(arm), out / "after-lift.jpg")
                if math.hypot(b["x"] - est["x"], b["y"] - est["y"]) < 40.0]
        held = not left                                     # the camera decides: is the ball still lying there?
    finally:
        await safe_stop(arm)
    say(f"fingers at {width if width is None else round(width)} mm; "
        + ("the ball is gone from the table: holding it" if held else "the ball is still on the table: NOT picked"))
    if not held:
        await gripper.do_command({"set": float(cfg.full_open_counts)})
    return held


def ask_landing(ask, planned_mm: np.ndarray, release_xy_mm: np.ndarray, arms: ArmsFrame) -> np.ndarray | None:
    """The landing measured by hand from the target mark: cm past (+) or short (-), cm left (+) or right (-)."""
    ans = ask("Where did it land vs. the target mark? cm past(+)/short(-), cm left(+)/right(-), e.g. 4,-2 "
              "(Enter: not seen): ").strip()
    if not ans:
        return None
    along, left = (float(v) * 10.0 for v in ans.replace(" ", "").split(","))
    u = planned_mm - release_xy_mm
    u = u / (np.linalg.norm(u) or 1.0)
    return arms.to_catcher(planned_mm + along * u + left * np.array([-u[1], u[0]]))


async def throw(arm, gripper, overhead, table, cfg: TossCfg, chain: Chain, repeat: int = 1, manual: bool = False,
                target=None, catch_z=None, out: Path | None = None, say=print, ask=input, clock=now,
                radius_px=(8.0, 30.0), table_px=None, picker=None) -> list[dict]:
    """Wind up, grip the ball barely, swing, let go at the trigger; then where it came down (the overhead
    camera, or measured by hand with --manual). Every throw is logged as it happens."""
    plan, f = throw_plan(cfg, chain, target, catch_z)
    if not plan.ok:
        raise SystemExit(f"No throw: {plan.reason}")
    if cfg.trigger == "timed" and cfg.start_latency_s is None:
        raise SystemExit("The timed trigger needs the start delay: python toss.py dry")
    spec, arms = cfg.spec(), ArmsFrame.of(cfg.thrower_in_catcher)
    cz = cfg.catch_z_mm if catch_z is None else catch_z
    planned = plan.landing.xy * 1000.0
    prof = Trapezoid(spec.amplitude_deg, plan.speed_deg_s, spec.accel_deg_s2)
    out = run_dir("throw", out)
    say(f"target mark: ({planned[0]:.0f}, {planned[1]:.0f}) mm in armfarm05's frame (x ahead of its base, y to its left)")
    log: list[dict] = []
    seen: list = []

    def landed() -> None:                                # the ball the scan just found is where the last throw landed
        if picker is not None and seen and log and log[-1].get("landing_catcher_mm") is None and "sent" in log[-1]:
            log[-1]["landing_catcher_mm"] = arms.to_catcher(np.array(seen[-1], float))
            log[-1]["landing_how"] = "wrist camera, before the next pick"
            miss = np.array(seen[-1], float) - planned
            say(f"  throw {log[-1]['i']} landed {float(np.linalg.norm(miss)):.0f} mm from the plan "
                f"({miss[0]:+.0f}, {miss[1]:+.0f}) mm")
            write_json(out / "throws.json", log)
    try:
        for i in range(repeat):
            if picker is not None:                           # the arm picks the ball up itself
                if not await picker(seen):
                    say("no ball in the gripper: stopping")
                    break
                landed()
                await go_slowly(arm, chain, cfg, pose_at(spec, 0.0, plan.base_deg), say,   # from the pick: opening up
                                keep=dataclasses.replace(KeepOut.of(cfg), reach_mm=300.0), where="the wind-up pose")
                if ask(f"throw {i + 1}/{repeat}: everyone clear of the throw? Enter: throw (s: stop) ").strip().lower() == "s":
                    break
            else:
                await go_slowly(arm, chain, cfg, pose_at(spec, 0.0, plan.base_deg), say)
                await set_mm(gripper, cfg, cfg.ball_diameter_mm + 15.0)
                if ask(f"throw {i + 1}/{repeat}: hold the ball in the fingers, press Enter (s: stop) ").strip().lower() == "s":
                    break
                if not holding(await grip_ball(gripper, cfg), cfg):
                    say("the fingers closed on nothing: put the ball between them")
                    continue
                if ask("let go of the ball; everyone clear of the throw? Enter: throw (s: stop) ").strip().lower() == "s":
                    break

            async def let_go(t_cmd, rec):
                return await release(lambda: open_fast(gripper, cfg), spec, prof, plan.trigger_progress_deg,
                                     cfg.trigger, t_cmd, cfg.start_latency_s, rec, clock, prof.duration + 3.0)

            t_cmd, rec, sent = await swing(arm, spec, plan.base_deg, plan.speed_deg_s, spec.accel_deg_s2, clock, let_go,
                                           timeout_s=prof.duration + 5.0)
            t, p = rec.progress(spec)
            fit, _, rms = fit_trapezoid(t - t_cmd, p, spec.amplitude_deg, plan.speed_deg_s, spec.accel_deg_s2)
            if sent is None or sent.at is None:
                why = sent.reason if sent is not None else "no answer in time"
                say(f"throw {i + 1}: the release never fired ({why}); the ball should still be in the gripper")
                log.append({"i": i + 1, "time": time.strftime("%H:%M:%S"), "released": False, "why": why,
                            "readings": {"t": (t - t_cmd).tolist(), "progress": p.tolist()}})
                write_json(out / "throws.json", log)
                continue
            sent_p = fit.progress(sent.at - t_cmd)
            if manual or overhead is None:
                landing = ask_landing(ask, planned, plan.release_mm[:2], arms)
                how = "by hand"
            else:
                landing, how = await resting_ball(overhead.frame, table, radius_px, table_px)
            entry = {"i": i + 1, "time": time.strftime("%H:%M:%S"), "spec": {**spec.__dict__, "tool_mm": list(spec.tool_mm)},
                     "base_deg": plan.base_deg, "speed_deg_s": plan.speed_deg_s, "catch_z_mm": cz,
                     "trigger": cfg.trigger, "trigger_deg": plan.trigger_progress_deg,
                     "release_deg": plan.release_progress_deg, "release_delay_s": cfg.release_delay_s,
                     "speed_factor": cfg.speed_factor, "sent": sent.to_json() if sent else None,
                     "sent_progress_deg": sent_p, "swing_start_s": fit.t0, "swing_w_deg_s": fit.w,
                     "swing_accel_deg_s2": fit.a, "swing_fit_rms_deg": rms,
                     "readings": {"t": (t - t_cmd).tolist(), "progress": p.tolist()},
                     "planned_thrower_mm": planned, "planned_catcher_mm": arms.to_catcher(planned),
                     "landing_catcher_mm": None if landing is None else np.asarray(landing, float),
                     "landing_how": how}
            log.append(entry)
            write_json(out / "throws.json", log)
            if landing is None:
                say(f"throw {i + 1}: landing not measured ({how})")
            else:
                miss = arms.to_thrower(landing) - planned
                u = planned - plan.release_mm[:2]
                u = u / (np.linalg.norm(u) or 1.0)
                say(f"throw {i + 1}: {float(miss @ u):+.0f} mm past the plan, {float(miss @ np.array([-u[1], u[0]])):+.0f} mm "
                    f"left; let go at {sent_p:.1f} deg (meant {plan.trigger_progress_deg:.1f})")
        if picker is not None and log and "sent" in log[-1]:
            await picker(seen, grab=False)                   # one last look: where the last throw landed
            landed()
    finally:
        await safe_stop(arm)
    summarize(log, say)
    return log


def summarize(log: list[dict], say=print) -> None:
    """How repeatable the throws were: spread of the landings and how far off the plan, as measured."""
    pts = np.array([t["landing_catcher_mm"] for t in log if t.get("landing_catcher_mm") is not None], float)
    if len(pts) == 0:
        return
    plan = np.array([t["planned_catcher_mm"] for t in log if t.get("landing_catcher_mm") is not None], float)
    off = pts.mean(axis=0) - plan.mean(axis=0)
    if len(pts) > 1:
        sd = pts.std(axis=0, ddof=1)
        say(f"{len(pts)} landings: spread sd ({sd[0]:.0f}, {sd[1]:.0f}) mm; on average {np.linalg.norm(off):.0f} mm "
            f"off the plan ({off[0]:+.0f}, {off[1]:+.0f})")
    else:
        say(f"1 landing: {np.linalg.norm(off):.0f} mm off the plan")


async def throw_taught(arm, gripper, cfg: TossCfg, chain: Chain, connect, picker=None, repeat: int = 1,
                       tcp_limit: float = 1000.0, release_shift_s: float = 0.0, jaw_s: float | None = None, out: Path | None = None,
                       say=print, ask=None, flick: bool = False) -> list[dict]:
    """Waj's throw (worldarm/toss/taught.py): pick (optional), the taught start pose, then one smooth streamed
    spline start -> second -> last under the arm's tool-speed cap, the gripper opened near peak ball speed from its
    own connection. connect(): a fresh connection to armfarm05 for the release thread."""
    from worldarm.toss import taught
    from worldarm.toss.physics import Release, fly
    keep = dataclasses.replace(KeepOut.of(cfg), reach_mm=300.0, table_z_mm=cfg.pick_table_z_mm, clearance_mm=60.0)
    out = run_dir("throw-taught", out)
    log: list[dict] = []
    try:
        for i in range(repeat):
            if picker is not None:
                if not await picker():
                    say("no ball in the gripper: stopping")
                    break
            await go_slowly(arm, chain, cfg, taught.POSES["start"], say, keep=keep)
            plan = (taught.plan_flick(chain, await joints_of(arm), tcp_limit=tcp_limit) if flick else
                    taught.plan(chain, await joints_of(arm), tcp_limit=tcp_limit, release_shift_s=release_shift_s))
            bad = next(((t, why) for t, q in zip(plan["times"][::3], plan["q"][::3])
                        if (why := pose_problem(chain, q, tuple(cfg.tool_mm), keep))), None)
            if bad:
                raise SystemExit(f"Refused: the throw's path isn't clear at {bad[0]:.2f} s: {bad[1]}")
            land = fly(Release(plan["ball_pos"] / 1000.0, plan["ball_vel"]), (cfg.pick_table_z_mm + 22.0) / 1000.0)
            planned = None if land is None else land.xy * 1000.0
            say(f"throw {i + 1}: {plan['duration']:.2f} s swing, ball leaves at {plan['ball_speed']:.2f} m/s from "
                f"({plan['ball_pos'][0]:.0f}, {plan['ball_pos'][1]:.0f}, {plan['ball_pos'][2]:.0f}) mm"
                + (f"; should land near ({planned[0]:.0f}, {planned[1]:.0f}) mm" if planned is not None else ""))
            try:
                await gripper.do_command({"set_gripper_speed": int(cfg.release_speed)})
            except Exception:
                pass
            lead = cfg.jaw_s if jaw_s is None else jaw_s
            releaser = taught.Releaser(connect, "gripper", plan["release_t"] - lead, release_counts(cfg), cfg.release_speed)
            releaser.start()
            if not await asyncio.to_thread(releaser.ready.wait, 20.0) or "error" in releaser.result:
                releaser.cancel()
                raise SystemExit(f"Release thread could not connect ({releaser.result.get('error', 'timeout')}); no throw")
            res = await taught.stream(arm, plan, releaser)
            rel = res["release"]
            say("RELEASE NOT SENT: the gripper is still closed" if rel is None else
                f"released at {rel['sent_s']:.3f} s (planned {rel['planned_release_s']:.3f} s), mid-motion: {rel['mid_motion']}")
            landing = ask_landing(ask, planned, plan["ball_pos"][:2], ArmsFrame.of(cfg.thrower_in_catcher)) \
                if ask is not None and planned is not None else None             # only when measuring by hand
            log.append({"i": i + 1, "time": time.strftime("%H:%M:%S"), "style": "taught", "tcp_limit": tcp_limit,
                        "release_shift_s": release_shift_s, "plan": {k: v for k, v in plan.items() if k not in ("times", "q")},
                        "stream": res, "planned_thrower_mm": planned, "landing_catcher_mm": landing})
            write_json(out / "throws.json", log)
    finally:
        await safe_stop(arm)
    return log


async def hold_bowl_at(arm, xy, cfg: TossCfg, say=print) -> None:
    """armfarm03 brings the bowl over xy (its frame, mm), held as taught: up, over, down, slowly."""
    from viam.proto.common import Pose
    h = cfg.bowl_hold
    if h is None:
        raise SystemExit("Teach how armfarm03 holds the bowl first: python toss.py catch --teach")
    gx, gy = float(xy[0]) + h["dx"], float(xy[1]) + h["dy"]
    reach = math.hypot(gx, gy)
    if not 200.0 <= reach <= 700.0 or h["z"] < 100.0:
        raise SystemExit(f"Refused: the bowl would have to be at ({gx:.0f}, {gy:.0f}, {h['z']:.0f}) mm, out of "
                         "armfarm03's comfortable reach (200-700 mm out, 100 mm up). Move the arms or the target.")
    await arm.do_command({"set_speed": float(cfg.catch_speed_deg_s)})
    cur = await arm.get_end_position()
    up = max(cur.z, h["z"]) + 80.0
    held = dict(o_x=h["o_x"], o_y=h["o_y"], o_z=h["o_z"], theta=h["theta"])
    await arm.move_to_position(Pose(x=cur.x, y=cur.y, z=up, o_x=cur.o_x, o_y=cur.o_y, o_z=cur.o_z, theta=cur.theta))
    await arm.move_to_position(Pose(x=gx, y=gy, z=up, **held))
    await arm.move_to_position(Pose(x=gx, y=gy, z=h["z"], **held))
    say(f"bowl over ({float(xy[0]):.0f}, {float(xy[1]):.0f}) mm")


async def teach_bowl(arm, overhead, table, cfg: TossCfg, say=print, ask=input, radius_px=(8.0, 30.0), table_px=None) -> dict:
    """With armfarm03 holding the bowl and the ball lying in it: where the bowl's middle is relative to the flange."""
    pose = await arm.get_end_position()
    ask("armfarm03 holding the bowl, the ball lying in it: press Enter ")
    xy, why = await resting_ball(overhead.frame, table, radius_px, table_px)
    if xy is None:
        raise SystemExit(f"The camera can't find the ball in the bowl: {why}")
    cfg.bowl_hold = {"dx": pose.x - float(xy[0]), "dy": pose.y - float(xy[1]), "z": pose.z, "o_x": pose.o_x,
                     "o_y": pose.o_y, "o_z": pose.o_z, "theta": pose.theta}
    say(f"the bowl's middle is {math.hypot(cfg.bowl_hold['dx'], cfg.bowl_hold['dy']):.0f} mm from the flange; saved: "
        f"{cfg.save()}")
    return cfg.bowl_hold


async def catch(catcher_arm, arm, gripper, overhead, table, cfg: TossCfg, chain: Chain, dry_run: bool = False,
                manual: bool = False, target=None, catch_z=None, out: Path | None = None, say=print, ask=input, clock=now,
                radius_px=(8.0, 30.0), table_px=None) -> dict:
    """The thrower's settings say where the ball will come down; armfarm03 puts the bowl there; then the throw."""
    plan, _ = throw_plan(cfg, chain, target, catch_z)
    if not plan.ok:
        raise SystemExit(f"No throw: {plan.reason}")
    arms = ArmsFrame.of(cfg.thrower_in_catcher)
    where = arms.to_catcher(plan.landing.xy * 1000.0)
    say(f"the ball should come down at ({where[0]:.0f}, {where[1]:.0f}) mm in armfarm03's frame")
    try:
        await hold_bowl_at(catcher_arm, where, cfg, say)
    finally:
        await safe_stop(catcher_arm)
    res = {"bowl_at_mm": where}
    if dry_run:
        return res
    log = await throw(arm, gripper, overhead, table, cfg, chain, 1, manual, target, catch_z, out, say, ask, clock,
                      radius_px, table_px)
    if log and log[-1]["landing_catcher_mm"] is not None:
        off = float(np.linalg.norm(np.asarray(log[-1]["landing_catcher_mm"]) - where))
        res.update({"off_mm": off, "caught": off <= cfg.bowl_diameter_mm / 2.0})
        say(f"{'CAUGHT' if res['caught'] else 'missed'}: the ball rests {off:.0f} mm from the bowl's middle")
    return res


# ---- the command line --------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def aim(s):
        s.add_argument("--distance", type=float, help="the ball flies this many mm from the release, toward armfarm03")
        s.add_argument("--target", help="target x,y in armfarm05's frame (mm)")
        s.add_argument("--catch-z", type=float, help="height the ball comes down to (mm, armfarm05's frame)")
    sub.add_parser("probe")
    s = sub.add_parser("plan")
    aim(s)
    s.add_argument("--design", action="store_true", help="search a wind-up pose and swing first (saved)")
    s.add_argument("--step", type=float, default=15.0, help="design: grid step (deg)")
    for name in ("windup", "dry", "throw", "catch"):
        s = sub.add_parser(name)
        aim(s)
        s.add_argument("--yes", action="store_true", help="skip the 'type go' question")
        if name == "dry":
            s.add_argument("--repeat", type=int, default=3)
            s.add_argument("--scale", type=float, default=1.0, help="swing at this share of the planned speed")
        if name == "throw":
            s.add_argument("--repeat", type=int, default=1)
        if name == "throw":
            s.add_argument("--hand", action="store_true", help="hand the ball to the gripper instead of it picking it up")
            s.add_argument("--squeeze", type=float, help="grip: close this many mm under the ball's size (default 2)")
            s.add_argument("--measure", action="store_true", help="type where each throw landed")
            s.add_argument("--tcp-limit", type=float, default=1100.0, help="the arm's tool speed cap (mm/s)")
            s.add_argument("--release-shift", type=float, default=0.0, help="release this long after peak speed (s)")
            s.add_argument("--flick", action="store_true", help="Waj's wrist snap: the ball faster than the arm's cap")
            s.add_argument("--taught", action="store_true", help="Waj's taught throw instead of our swing")
        if name in ("throw", "catch"):
            s.add_argument("--manual", action="store_true", help="measure the landing by hand, not with the camera")
            s.add_argument("--pick", action="store_true", help="the arm finds the ball on the table and picks it up")
        if name == "catch":
            s.add_argument("--teach", action="store_true", help="no motion: learn how armfarm03 holds the bowl")
            s.add_argument("--dry", action="store_true", help="move the bowl, don't throw")
    s = sub.add_parser("pick")
    s.add_argument("--yes", action="store_true", help="skip the 'type go' question")
    s.add_argument("--squeeze", type=float, help="grip: close this many mm under the ball's size (default 2)")
    s = sub.add_parser("grip")
    s.add_argument("--teach", action="store_true")
    s.add_argument("--repeat", type=int, default=0)
    sub.add_parser("report")
    s = sub.add_parser("calibrate")
    s.add_argument("--fit", default=",".join(PARAMS), help=f"which to fit, of {','.join(PARAMS)}")
    return p


def target_of(cfg: TossCfg, args) -> np.ndarray | None:
    if getattr(args, "target", None):
        return np.array([float(v) for v in args.target.split(",")], float)
    if getattr(args, "distance", None):
        cfg.throw_mm = float(args.distance)                    # by heading: the ball's path, toward armfarm03
    return None


async def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    cfg = TossCfg.load()
    target, catch_z = target_of(cfg, args), getattr(args, "catch_z", None)
    if args.cmd == "plan":
        plan_cmd(cfg, chain_of(cfg), args.design, target, catch_z, args.step)
        return
    if args.cmd == "report":
        report(cfg)
        return
    if args.cmd == "calibrate":
        calibrate(cfg, chain_of(cfg), fit=tuple(args.fit.split(",")))
        return
    from viam.components.arm import Arm
    from viam.components.gripper import Gripper

    from worldarm.ball.config import BallCfg
    from worldarm.ball.live import Overhead
    from worldarm.toss import machines

    chain = chain_of(cfg) if args.cmd != "probe" else None
    teaching = args.cmd == "catch" and args.teach
    if args.cmd in MOVES and not teaching and not args.yes:
        speed = ""
        if args.cmd in ("dry", "catch") or (args.cmd == "throw" and not (args.taught or args.flick)):
            plan, _ = throw_plan(cfg, chain, target, catch_z)
            if not plan.ok:
                raise SystemExit(f"No throw: {plan.reason}")
            speed = f" at up to {plan.speed_deg_s * getattr(args, 'scale', 1.0):.0f} deg/s"
        if input(f"This MOVES {MOVES[args.cmd]}{speed}. Everyone clear of both arms and the throw path, someone at the "
                 "E-stop? Type 'go': ").strip() != "go":
            raise SystemExit("Cancelled.")
    needs_catcher = args.cmd == "catch"
    thrower = catcher = None
    try:
        if not teaching:
            thrower, tv = await machines.connect(cfg.thrower_env, cfg.thrower)
            arm = Arm.from_robot(thrower, machines.component(tv, "VIAM_ARM", "arm"))
            gripper = Gripper.from_robot(thrower, machines.component(tv, "VIAM_GRIPPER", "gripper"))
        overhead = table = table_px = None
        bcfg = BallCfg.load(cfg.catcher)
        if needs_catcher:
            catcher, cv = await machines.connect(cfg.catcher_env, cfg.catcher)
            overhead, table, table_px = Overhead(catcher, cfg.camera), bcfg.table(), bcfg.table_px
            if table is None:
                raise SystemExit("No table map for the overhead camera: python ball.py calibrate (or use --manual)")
        if args.cmd == "probe":
            await probe(arm, gripper, cfg)
        elif args.cmd == "windup":
            await windup(arm, chain, cfg, target)
        elif args.cmd == "dry":
            await dry(arm, chain, cfg, args.repeat, args.scale, target)
        elif args.cmd == "grip":
            await grip(gripper, cfg, args.teach, args.repeat)
        elif args.cmd == "pick":                                # find it, pick it, go to the throwing position
            if args.squeeze is not None:
                cfg.squeeze_mm = args.squeeze
            if await pick_ball(arm, gripper, thrower, chain, cfg, machines.component(tv, "VIAM_CAMERA", "cam")):
                fit_under_cap(cfg, chain, 1100.0, say=lambda *_: None)
                plan, _ = throw_plan(cfg, chain)
                if not plan.ok:
                    raise SystemExit(f"No throw planned, so no throwing position: {plan.reason}")
                print("6. moving to the throwing position")
                try:
                    await go_slowly(arm, chain, cfg, pose_at(cfg.spec(), 0.0, plan.base_deg),
                                    keep=dataclasses.replace(KeepOut.of(cfg), reach_mm=300.0), where="the throwing position")
                finally:
                    await safe_stop(arm)
                print("ready: in the throwing position, holding the ball (the next run opens the gripper first: "
                      "take the ball when it does)")
        elif args.cmd == "throw" and (args.taught or args.flick):   # Waj's taught throw, or his wrist flick
            cam = machines.component(tv, "VIAM_CAMERA", "cam")
            picker = None if args.hand else (lambda: pick_ball(arm, gripper, thrower, chain, cfg, cam))
            if args.squeeze is not None:
                cfg.squeeze_mm = args.squeeze

            async def fresh():
                m, _ = await machines.connect(cfg.thrower_env, cfg.thrower)
                return m
            await throw_taught(arm, gripper, cfg, chain, fresh, picker, args.repeat, args.tcp_limit, args.release_shift,
                               ask=input if args.measure else None, flick=args.flick)
        elif args.cmd == "throw":                               # the default: find it, pick it, our swing
            cam = machines.component(tv, "VIAM_CAMERA", "cam")
            picker = None if args.hand else (lambda seen=None, grab=True: pick_ball(arm, gripper, thrower, chain, cfg, cam,
                                                                                  seen=seen, grab=grab))
            if args.squeeze is not None:
                cfg.squeeze_mm = args.squeeze
            fit_under_cap(cfg, chain, args.tcp_limit)
            await throw(arm, gripper, overhead, table, cfg, chain, args.repeat, True, target, catch_z,
                        radius_px=tuple(bcfg.ball_radius_px), table_px=table_px, picker=picker,
                        ask=input if args.measure else (lambda *_: ""))
        elif args.cmd == "catch":
            catcher_arm = Arm.from_robot(catcher, machines.component(cv, "VIAM_ARM", "arm"))
            if args.teach:
                await teach_bowl(catcher_arm, overhead, table, cfg, radius_px=tuple(bcfg.ball_radius_px), table_px=table_px)
            else:
                await catch(catcher_arm, arm, gripper, overhead, table, cfg, chain, args.dry, args.manual, target,
                            catch_z, radius_px=tuple(bcfg.ball_radius_px), table_px=table_px)
    finally:
        for m in (thrower, catcher):
            if m is not None:
                await m.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
