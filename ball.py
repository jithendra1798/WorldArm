"""Chapter 2, one drill per piece: the goalie that gets to where a rolled ball WILL cross its lane.

Run in this order the first time (settings land in config/ball_armfarm03.json, logs in runs/):

  python ball.py check                 no motion: overhead camera speed, ball detection, table outline
  python ball.py lane --teach a        no motion: the arm's pose now is one end of the defended lane
  python ball.py lane --teach b        ...and the other end (paddle in the gripper, low over the table)
  python ball.py calibrate             MOVES: the arm sets the ball down on a 3x3 grid; camera <-> table map
  python ball.py lane-test             MOVES: how fast the paddle moves along the lane, and its latency
  python ball.py record --rolls 15     no motion: people roll the ball; the tracks are saved
  python ball.py fit                   learn how fast the ball slows on this table, from the recorded rolls
  python ball.py goalie --rolls 20 --mode predict    MOVES: defend the lane (the demo)
  python ball.py goalie --rolls 20 --mode chase      MOVES: same, following the ball instead (H5 baseline)
  python ball.py report                predict vs. chase, with 95% intervals; the learned slowdown (H4)
  python ball.py sim                   offline: intercept rates for these settings, simulated

Commands that move ask "type go" first (--yes skips it); Ctrl-C or any error stops the arm. fit, report and
sim never connect to the robot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm

from worldarm.ball.config import BallCfg
from worldarm.ball.detect import find_ball
from worldarm.ball.intercept import Lane
from worldarm.ball.live import (LaneArm, Overhead, defend, look_for_lookalikes, now, paddle_zone_px, setup,
                                 sightings, wait_for_ready)
from worldarm.ball.holdout import holdout_errors
from worldarm.ball.model import fit_deceleration
from worldarm.ball.sim import SimCfg, intercept_rate
from worldarm.ball.table import fit_table_map, leave_one_out_mm
from worldarm.metrics import rate as wilson_rate

RUNS = Path("runs")
DEFAULT_LANE = Lane(np.array([450.0, -250.0]), np.array([450.0, 250.0]), 0.0, 0.0)   # sim/report before teaching


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def table_outline(img: np.ndarray) -> np.ndarray | None:
    """The white table in the overhead picture: the largest bright, unsaturated region, slightly shrunk."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    otsu, _ = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # adapts to the room light
    white = ((hsv[:, :, 2] > max(100.0, otsu)) & (hsv[:, :, 1] < 45)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # cables, the ball: filled in
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))    # cut thin links (arm, edges)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == big)
    box = cv2.boxPoints(cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32)))  # a table is a rectangle
    c = box.mean(axis=0)
    return c + (box - c) * 0.97


# ---- no motion -------------------------------------------------------------------------------------------

async def check(cam, cfg: BallCfg, frames: int = 40, out: Path | None = Path("captures"), say=print) -> dict:
    """The overhead camera's speed, whether it sees the ball, and the table's outline (saved as the search area)."""
    t0, seen, img, blank = now(), 0, None, 0
    table_px = np.array(cfg.table_px) if cfg.table_px else None
    for _ in range(frames):
        _, im = await cam.frame()
        if im is None:
            blank += 1
            continue
        img = im
        seen += find_ball(img, cfg.ball_radius_px, table_px) is not None
    fps = frames / max(now() - t0, 1e-6)
    say(f"overhead camera: {fps:.1f} fps, ball seen in {seen}/{frames} frames")
    if img is None:
        raise SystemExit(f"No color image from '{cfg.camera}' in {frames} tries.")
    if blank:
        say(f"  {blank}/{frames} frames came back without a color image")
    outline = table_outline(img)
    if outline is not None:
        cfg.table_px = np.round(outline, 1).tolist()
        say(f"table outline found ({len(outline)} corners) and saved: {cfg.save()}")
    s = find_ball(img, cfg.ball_radius_px, outline)
    if s:
        tm = cfg.table()
        where = f" -> table ({tm.to_world((s.u, s.v))[0]:.0f}, {tm.to_world((s.u, s.v))[1]:.0f})" if tm else ""
        say(f"ball at pixel ({s.u:.0f}, {s.v:.0f}), radius {s.radius_px:.0f} px{where}")
    elif not seen:
        say("  no ball seen: if the ball is out of view, good - nothing else on the table looks like it")
    if out is not None:
        vis = img.copy()
        if outline is not None:
            cv2.polylines(vis, [outline.astype(np.int32)], True, (0, 200, 0), 3)
        if s:
            cv2.circle(vis, (int(s.u), int(s.v)), int(s.radius_px) + 4, (0, 0, 255), 3)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "ball_check.jpg"), vis)
        say(f"picture: {out / 'ball_check.jpg'}")
    return {"fps": fps, "seen": seen, "frames": frames, "table_px": cfg.table_px,
            "ball_px": None if s is None else (s.u, s.v)}


async def teach(arm, cfg: BallCfg, end: str, say=print) -> dict:
    """The arm's flange pose now is one end of the lane (the paddle in the gripper, low over the table). The
    lane keeps the higher of the two ends' heights, so the paddle clears the table at both."""
    p = await arm.get_end_position()
    lane = dict(cfg.lane or {})
    lane[end] = [p.x, p.y]
    lane[f"z_{end}"], lane[f"theta_{end}"] = p.z, p.theta
    lane.update(theta=p.theta, orientation=[p.o_x, p.o_y, p.o_z], home=[0.0, 0.0])
    lane["z"] = max(lane[k] for k in ("z_a", "z_b") if k in lane)
    cfg.lane = lane
    say(f"lane end {end}: ({p.x:.0f}, {p.y:.0f}) at flange z {p.z:.0f}, wrist {p.theta:.0f}")
    if "a" in lane and "b" in lane:
        length = float(np.hypot(*(np.array(lane["b"]) - np.array(lane["a"]))))
        say(f"lane: {length:.0f} mm long")
        if length < 150:
            say("  that's short: the two ends should be well apart")
        if "z_a" in lane and "z_b" in lane and abs(lane["z_a"] - lane["z_b"]) > 5.0:
            say(f"  the ends were taught {abs(lane['z_a'] - lane['z_b']):.0f} mm apart in height: the paddle "
                f"moves at the higher, flange z {lane['z']:.0f}")
        if "theta_a" in lane and "theta_b" in lane:
            turn = abs((lane["theta_a"] - lane["theta_b"] + 180.0) % 360.0 - 180.0)
            if turn > 5.0:
                say(f"  the wrist differs by {turn:.0f} deg between the ends: the paddle keeps end {end}'s")
    say(f"saved: {cfg.save()}")
    return lane


async def track_roll(stream, wait_s: float = 30.0, max_s: float = 8.0) -> list:
    """One roll's sightings [(t, x, y)], from the ball at rest (ready) until it has rolled and stopped, is lost
    for a second, or max_s after it started; empty-handed after wait_s if nobody rolls it."""
    track, t_ready, t_roll, t_seen = [], None, None, None
    async for t, xy, _ in stream:
        t_ready = t if t_ready is None else t_ready
        if xy is not None:
            track.append((round(float(t), 4), round(float(xy[0]), 1), round(float(xy[1]), 1)))
            t_seen = t
            if t_roll is None and math.dist(track[-1][1:], track[0][1:]) > 30.0:
                t_roll = t
        if t_roll is None:
            if t - t_ready > wait_s:
                break
            continue
        k = len(track) - 6
        stopped = k >= 0 and t - track[k][0] < 0.5 and math.dist(track[-1][1:], track[k][1:]) < 8.0
        if stopped or t - t_seen > 1.0 or t - t_roll > max_s:
            break
    return track


async def record(cam, cfg: BallCfg, rolls: int, out: Path | None = None, say=print,
                 paddle_s: float | None = None) -> list:
    """People roll the ball (the arm doesn't move); each roll's sightings are saved, to learn the slowdown from.
    paddle_s: where the paddle rests on the lane, if it does (the camera then ignores its surroundings)."""
    table, lane = cfg.table(), cfg.get_lane()
    if table is None or lane is None:
        raise SystemExit("Calibrate and teach the lane first.")
    out = out or RUNS / f"{stamp()}-ball-record"
    out.mkdir(parents=True, exist_ok=True)
    zone = paddle_zone_px(table, lane, (paddle_s, paddle_s), cfg) if paddle_s is not None else []
    stream = sightings(cam, table, cfg, avoid=zone, lane=lane)
    tracks = []
    for i in range(1, rolls + 1):
        say(f"roll {i}/{rolls}")
        await wait_for_ready(stream, lane, say=say)
        tracks.append(await track_roll(stream))
        (out / "rolls.json").write_text(json.dumps(tracks))    # after every roll: nothing lost to a Ctrl-C
        say(f"  {len(tracks[-1])} sightings")
    say(f"saved {len(tracks)} rolls: {out / 'rolls.json'}   next: python ball.py fit")
    return tracks


def rolling_part(track, min_points: int = 8) -> tuple[np.ndarray, np.ndarray] | None:
    """The free roll in one recorded track: from its fastest moment (the push just over) until it stops, turns
    back (it hit something) or is lost for a while. None if too little of it is left."""
    a = np.asarray(track, float)
    if len(a) < max(min_points, 6):
        return None
    t, xy = a[:, 0], a[:, 1:3]
    speed = np.zeros(len(t))                       # over 4 sightings (~0.17 s): steadier than frame to frame
    speed[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=1) / np.maximum(t[4:] - t[:-4], 1e-6)
    i0 = int(np.argmax(speed))
    u = xy[min(i0 + 2, len(t) - 1)] - xy[max(i0 - 2, 0)]
    u = u / (np.linalg.norm(u) or 1.0)
    s = (xy - xy[i0]) @ u
    end = i0
    for j in range(i0 + 1, len(t)):
        if t[j] - t[j - 1] > 0.25 or speed[j] < 30.0 or s[j] < s[i0:j].max() - 15.0:
            break
        end = j
    end = i0 + int(np.argmax(s[i0:end + 1]))        # not past its furthest point (where it turned back)
    return (t[i0:end + 1], xy[i0:end + 1]) if end + 1 - i0 >= min_points else None


def load_rolls(runs: Path | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """The free-rolling part of every roll recorded with `record` (goalie runs are left out: there the paddle
    hides the ball and turns it back)."""
    tracks = []
    for f in sorted(Path(runs or RUNS).glob("*-ball-record/rolls.json")):
        for tr in json.loads(f.read_text()):
            part = rolling_part(tr)
            if part is not None:
                tracks.append(part)
    return tracks


def fit(cfg: BallCfg, runs: Path | None = None, say=print) -> dict:
    """Learn how fast the ball slows on this table from the recorded rolls, save it, and show how well it
    predicts rolls it hasn't seen (H4)."""
    tracks = load_rolls(runs)
    a, spread, n = fit_deceleration(tracks)
    if n == 0:
        raise SystemExit("No usable recorded rolls yet: python ball.py record --rolls 15")
    say(f"the ball slows at {a:.0f} mm/s^2 (spread {spread:.0f}) on this table, from {n} rolls")
    cfg.decel_mm_s2, cfg.decel_spread = round(a, 1), round(spread, 1)
    say(f"saved: {cfg.save()}")
    return {"decel": a, "spread": spread, "rolls": n, "held_out": h4_on_real_rolls(tracks, say)}


def h4_on_real_rolls(tracks, say=print) -> dict | None:
    """H4 on our own recorded rolls: each one predicted with the slowdown learned from the others (holdout.py)."""
    if len(tracks) < 3:
        say("  H4 on real rolls needs at least 3 recorded rolls")
        return None
    h = holdout_errors(tracks).summary()
    say(f"  H4: each of the {h['rolls']} rolls predicted with the slowdown learned from the other rolls")
    for horizon, r in h["arrival"].items():
        lr, cv, closer = r["learned"], r["constant"], r["learned_closer"]
        if lr["n"] and cv["n"]:
            say(f"  {horizon:.1f} s before it got there: arrival-time error {_ms(lr)} learned vs {_ms(cv)} "
                f"constant-velocity; learned closer on {closer['closer']} of {closer['n']} rolls")
    st = h["stop"]
    if st["n"]:
        say(f"  where it came to rest, after watching it 0.5 s: {st['median']:.0f} mm off (median of {st['n']} "
            "rolls; constant velocity can't say)")
    return h


def _ms(m: dict) -> str:
    """A median error in ms, and how many predictions failed outright (said it stops short; it arrived)."""
    if m["median"] is None:
        return "-"
    med = "inf" if math.isinf(m["median"]) else f"{m['median'] * 1000:.0f} ms"
    return med + (f" ({m['failed']} of {m['n']} wrongly said it stops short)" if m["failed"] else "")


def report(cfg: BallCfg, runs: Path | None = None, say=print) -> dict:
    """H5 from the goalie runs (blocked / (blocked + goal), with 95% intervals; the operator's calls where they
    were confirmed, and how often the camera's call agreed) and H4 from the recorded rolls."""
    by_mode: dict[str, list] = {}
    checked: dict[str, list] = {}                     # (camera's call, operator's call), runs with --confirm
    for f in sorted(Path(runs or RUNS).glob("*-ball-goalie-*/goalie.json")):
        d = json.loads(f.read_text())
        by_mode.setdefault(d["mode"], []).extend(r["outcome"] for r in d["rolls"])
        checked.setdefault(d["mode"], []).extend((r.get("camera"), r["human"]) for r in d["rolls"] if "human" in r)
    res: dict = {"H5": {}}
    say("H5  going to the predicted point vs. chasing the ball:")
    if not by_mode:
        say("  no goalie runs yet")
    for mode, outs in sorted(by_mode.items()):
        b, g = outs.count("blocked"), outs.count("goal")
        res["H5"][mode] = {"blocked": b, "goal": g, "short": outs.count("short"), "not_judged": outs.count(None)}
        say(f"  {mode:<8} blocked {wilson_rate(b, b + g)}   ({outs.count('short')} stopped short, "
            f"{outs.count(None)} not judged)")
    for mode, pairs in sorted((m, ps) for m, ps in checked.items() if ps):
        k = sum(c == h for c, h in pairs)
        res.setdefault("camera_agrees", {})[mode] = (k, len(pairs))
        say(f"  {mode:<8} the camera's call matched the operator's on {wilson_rate(k, len(pairs))}")
    tracks = load_rolls(runs)
    a, spread, n = fit_deceleration(tracks)
    res["H4"] = {"decel": a, "spread": spread, "rolls": n}
    say(f"H4  learned slowdown {a:.0f} mm/s^2 (spread {spread:.0f}) from {n} recorded rolls")
    if n:
        res["H4"]["held_out"] = h4_on_real_rolls(tracks, say)
    return res


def sim(cfg: BallCfg, n: int = 300, say=print, realistic: bool = False) -> dict:
    """Intercept rates for these settings (lane, arm timing, paddle, slowdown), simulated. realistic: the
    harsher profile (the paddle speeds up and slows down, part of the table out of view, jittery timestamps,
    uneven rolls; sim.SimCfg.realistic), with the measured roll-to-roll spread if `fit` has run."""
    lane = cfg.get_lane() or DEFAULT_LANE
    arm = cfg.arm_model()
    decel = cfg.decel_mm_s2 or 120.0
    if realistic:
        spread = cfg.decel_spread / decel if cfg.decel_spread and math.isfinite(cfg.decel_spread) else None
        simcfg = SimCfg.realistic(true_decel=decel, **({"decel_spread": spread} if spread else {}))
    else:
        simcfg = SimCfg(true_decel=decel)
    say(f"SIMULATED ({'realistic' if realistic else 'ideal'} profile): lane {lane.length:.0f} mm, paddle half-width "
        f"{cfg.paddle_half_width_mm:.0f} mm, arm {arm}, slowdown {simcfg.true_decel:.0f} mm/s^2")
    res = {}
    for mode in ("predict", "chase"):
        r = intercept_rate(lane, mode, simcfg.true_decel, arm, simcfg, n=n, half_width_mm=cfg.paddle_half_width_mm)
        res[mode] = r
        say(f"  {mode:<8} {wilson_rate(r['defended'], r['reached'])}   first command {r['median_lead_s'] or 0:.2f} s "
            "before the ball arrives")
    return res


# ---- moving ----------------------------------------------------------------------------------------------

def _holding_ball(width_mm: float | None, r: float) -> bool:
    return width_mm is not None and abs(width_mm - 2 * r) <= 12.0


async def _ball_pixel(cam, cfg: BallCfg, table_px, frames: int = 8) -> tuple[np.ndarray | None, str]:
    """Where the ball sits in the overhead image (median over a few frames), or None and why not."""
    seen = []
    for _ in range(frames):
        _, img = await cam.frame()
        s = find_ball(img, cfg.ball_radius_px, table_px) if img is not None else None
        if s:
            seen.append((s.u, s.v))
    if len(seen) < frames // 2:
        return None, f"the camera saw the ball in {len(seen)}/{frames} frames"
    a = np.array(seen)
    uv = np.median(a, axis=0)
    close = np.linalg.norm(a - uv, axis=1) <= 2.5
    if close.sum() < frames // 2:
        return None, "it was still moving, or something else was seen too"
    return np.median(a[close], axis=0), ""


async def calibrate(rig, cam, cfg: BallCfg, ask=input, say=print) -> dict:
    """The arm sets the ball down on a grid around the lane and the overhead camera sees it there: the
    camera <-> table map. Grid points outside the work area or out of reach are left out; a point the camera
    doesn't see clearly is skipped. Every move goes through the rig (planned, checked); any error stops the arm."""
    from worldarm.act.watch import ContactWatch
    from worldarm.cell.geometry import down_pose, to_pose
    from worldarm.cell.rig import MotionRefused
    lane = cfg.get_lane()
    if lane is None:
        raise SystemExit("Teach the lane first (ball.py lane --teach a / --teach b): the grid is laid out around it.")
    cell = rig.cfg
    g, w, r = cell.gripper, cell.workcell, cfg.ball_radius_mm
    grip_z = w.table_z_mm + r + g.pad_center_below_frame_mm          # pads at the ball's middle
    carry_z = w.table_z_mm + 2 * r + 60 + g.tip_below_frame_mm

    def unreachable(p) -> str | None:
        try:
            for z in (carry_z, grip_z):
                rig.check(down_pose(p[0], p[1], z, 0.0), "calibration point")
        except MotionRefused as e:
            return str(e)
        return "out of the arm's reach" if float(np.hypot(*p)) > w.reach_mm - 20.0 else None

    pts, left_out = [], []
    for far in (240.0, 180.0, 120.0):                 # 3 rows out from the lane: as deep as the work area allows
        grid = [lane.point(s) - lane.normal * d for d in (0.0, far / 2, far)
                for s in np.linspace(0.1 * lane.length, 0.9 * lane.length, 3)]
        why = [unreachable(p) for p in grid]
        if why.count(None) > len(pts):
            pts, left_out = [p for p, y in zip(grid, why) if y is None], [(p, y) for p, y in zip(grid, why) if y]
        if len(pts) == len(grid):
            break
    for p, y in left_out:
        say(f"  grid point ({p[0]:.0f}, {p[1]:.0f}) left out: {y}")
    if len(pts) < 6:
        raise SystemExit(f"Only {len(pts)} of the 9 grid points around the lane are in the work area; need 6+.")
    park = to_pose(cell.view_pose) if cell.view_pose else None
    table_px = np.array(cfg.table_px) if cfg.table_px else None
    pixels, world, holding = [], [], False
    try:
        ask("The gripper will open: if it's holding the paddle, take it out now. Press Enter. ")
        await rig.open_mm(min(g.max_open_mm, 2 * r + 25))
        ask("Hold the ball between the open fingers, then press Enter (the gripper will close on it). ")
        await rig.grab(g.block_torque_pct)
        width = await rig.width_mm()
        if not _holding_ball(width, r):
            raise SystemExit(f"Not holding the ball (fingers at {width} mm, expected ~{2 * r:.0f}). Check ball_radius_mm.")
        holding = True
        for i, p in enumerate(pts, 1):
            await rig.set_speed(cell.speeds.travel_deg_s)
            await rig.move(down_pose(p[0], p[1], carry_z, 0.0), label=f"calibration point {i}")
            await rig.set_speed(cell.speeds.slow_deg_s)
            await rig.move_watched(down_pose(p[0], p[1], grip_z + 3.0, 0.0), ContactWatch(cell.contact, carry_z),
                                   label="set the ball down")
            await rig.open_mm(2 * r + 20)
            holding = False
            await rig.move(down_pose(p[0], p[1], carry_z, 0.0), linear=True, label="up from the ball")
            if park is not None:
                await rig.set_speed(cell.speeds.travel_deg_s)
                await rig.move(park, label="out of the camera's way")
            uv, why = await _ball_pixel(cam, cfg, table_px)
            if uv is not None and any(np.linalg.norm(uv - q) < 20.0 for q in pixels):
                uv, why = None, "it was seen where an earlier point was: is something else being taken for the ball?"
            if uv is not None:
                pixels.append(uv)
                world.append(p)
                say(f"  point {i}: table ({p[0]:.0f}, {p[1]:.0f}) seen at pixel ({uv[0]:.1f}, {uv[1]:.1f})")
            else:
                say(f"  point {i}: skipped ({why})")
            await rig.set_speed(cell.speeds.travel_deg_s)
            await rig.move(down_pose(p[0], p[1], carry_z, 0.0), label="back above the ball")
            await rig.set_speed(cell.speeds.slow_deg_s)
            hit = await rig.move_watched(down_pose(p[0], p[1], grip_z, 0.0), ContactWatch(cell.contact, carry_z),
                                         label="down to the ball")
            if not hit.contact:
                await rig.grab(g.block_torque_pct)
                holding = _holding_ball(await rig.width_mm(), r)
            if not holding:
                say(f"  the ball isn't where it was put down at point {i}: the grid stops here")
                await rig.open_mm(2 * r + 20)
                await rig.move(down_pose(p[0], p[1], carry_z, 0.0), linear=True, label="up, without the ball")
                break
            await rig.move(down_pose(p[0], p[1], carry_z, 0.0), linear=True, label="lift the ball")
        if holding:                                   # above the last point, ball in hand: leave it there
            await rig.move_watched(down_pose(p[0], p[1], grip_z + 3.0, 0.0), ContactWatch(cell.contact, carry_z),
                                   label="put the ball down")
            await rig.open_mm(2 * r + 20)
            holding = False
            await rig.move(down_pose(p[0], p[1], carry_z, 0.0), linear=True, label="up")
        if len(pixels) < 6:
            raise SystemExit(f"Only {len(pixels)} points seen; need 6+. Is the camera's view of the lane area clear?")
        tm, loo = fit_table_map(pixels, world), leave_one_out_mm(pixels, world)
        worst = int(np.argmax(loo))
        if len(pixels) >= 7 and loo[worst] > 15.0 and loo[worst] > 3.0 * statistics.median(loo):
            say(f"  point at table ({world[worst][0]:.0f}, {world[worst][1]:.0f}) disagrees with the rest by "
                f"{loo[worst]:.0f} mm: left out")
            del pixels[worst], world[worst]
            tm, loo = fit_table_map(pixels, world), leave_one_out_mm(pixels, world)
        say(f"table map from {len(pixels)} points: fit {tm.rms_mm:.1f} mm rms, leave-one-out "
            f"median {statistics.median(loo):.1f} mm, worst {max(loo):.1f} mm")
        if tm.rms_mm > 10.0:
            say("  WARNING: over 10 mm: ball positions will be off by about that much")
        cfg.table_map = tm.to_json()
        say(f"saved: {cfg.save()}")
        mid = lane.point(lane.length / 2)
        await rig.set_speed(cell.speeds.travel_deg_s)
        await rig.move(down_pose(mid[0], mid[1], carry_z, 0.0), label="above the middle of the lane")
        if ask("Hand the paddle between the open fingers and press Enter (the gripper closes on it), "
               "or type s to skip. ").strip().lower() != "s":
            await rig.grab(g.block_torque_pct)
    except BaseException:
        await rig.stop()
        raise
    return {"table_map": tm, "points": len(pixels), "world": world, "leave_one_out_mm": loo}


async def _retarget(la: LaneArm, arm, lane: Lane) -> float | None:
    """A re-aim, as the goalie makes them: the paddle heading for the far end gets a new target (back to 0)
    once it's on its way. Seconds from that command until it's moving back (stop, clear, restart, turn),
    or None if it never turned back."""
    def along(p) -> float:
        return lane.along((p.x, p.y))

    await la.go(lane.length)
    t_go = now()
    while True:
        s = along(await arm.get_end_position())
        if s >= 30.0 and now() - t_go >= 0.3:
            break
        if la.idle or now() - t_go > 3.0:
            return None                                    # it never got going
        await asyncio.sleep(0.01)
    t0, best, back_at = now(), s, None
    await la.go(0.0)
    while not la.idle:
        s = along(await arm.get_end_position())
        best = max(best, s)
        if back_at is None and s < best - 2.0:
            back_at = now()
        await asyncio.sleep(0.01)
    await la.settle()
    if back_at is None or abs(along(await arm.get_end_position())) > 5.0:
        return None
    return back_at - t0


async def lane_test(arm, cfg: BallCfg, times: int = 2, speed: float | None = None, cam=None, say=print) -> dict:
    """Time the paddle along the lane, end to end and back: command -> it starts moving (latency), how fast it
    goes, how far it strays from the lane on the way (a direct arm move needn't follow a straight line), and
    how long a new target mid-move takes to turn it back (what every re-aim costs). With the overhead camera
    and the ball off the table, also looks at both ends for anything outside the paddle zone that passes
    for the ball (the camera sees the gripper and wrist further out the higher they are)."""
    lane = cfg.get_lane()
    if lane is None:
        raise SystemExit("Teach the lane first.")
    la = LaneArm(arm, lane, speed or cfg.lane_speed_deg_s, cfg.max_lane_speed_deg_s)
    problem = await la.off_lane()
    if problem:
        raise SystemExit(problem)
    table = cfg.table() if cam is not None else None
    lat, spd, stray, high, lookalikes, turn = [], [], 0.0, 0.0, [], None
    try:
        await la.start()
        await la.approach(0.0, say=say)
        for k, target in enumerate([lane.length, 0.0] * times):
            if table is not None and k < 2:               # at end a, then at end b
                lookalikes += [(la.target, xy) for xy in
                               await look_for_lookalikes(cam, table, lane, cfg, (la.target, la.target))]
            p0 = await arm.get_end_position()
            t0 = now()
            await la.go(target)
            start = None
            while not la.idle:
                p = await arm.get_end_position()
                if start is None and math.hypot(p.x - p0.x, p.y - p0.y) > 2.0:
                    start = now()
                stray = max(stray, abs(float((np.array([p.x, p.y]) - lane.a) @ lane.normal)))
                high = max(high, abs(float(p.z) - lane.z))
                await asyncio.sleep(0.01)
            t_end = now()
            await la.settle()
            if start is None:
                say(f"  to {target:.0f} mm: it didn't move")
                continue
            lat.append(start - t0)
            spd.append(lane.length / max(t_end - start, 1e-3))
            say(f"  {lane.length:.0f} mm: started after {lat[-1]:.2f} s, {spd[-1]:.0f} mm/s")
        turn = await _retarget(la, arm, lane)
    finally:
        await la.stop()
    if not lat:
        raise SystemExit("The paddle never moved: nothing measured.")
    say(f"  on the way it strayed up to {stray:.0f} mm from the lane line and {high:.0f} mm from its height")
    if stray > 20.0 or high > 10.0:
        say("  WARNING: that's the path the goalie's moves take too: check the paddle clears the table")
    if turn is None:
        say("  WARNING: a new target mid-move didn't turn the paddle back: the goalie's re-aims won't work")
    else:
        say(f"  a new target mid-move turned it back after {turn:.2f} s (stop + restart: each re-aim costs this)")
    for s_at, xy in lookalikes:
        say(f"  WARNING: with the paddle at {s_at:.0f} mm, something at table ({xy[0]:.0f}, {xy[1]:.0f}) passes for "
            "the ball: widen zone_front_mm / zone_back_mm / zone_side_mm, or take it away")
    cfg.arm = {"latency_s": round(statistics.median(lat), 3), "speed_mm_s": round(statistics.median(spd), 1)}
    cfg.arm_speed_deg_s, cfg.arm_retarget_s = la.speed, None if turn is None else round(turn, 3)
    say(f"lane arm: {cfg.arm}, re-aim {cfg.arm_retarget_s} s, at {la.speed:.0f} deg/s   saved: {cfg.save()}")
    return {"latency_s": statistics.median(lat), "speed_mm_s": statistics.median(spd), "stray_mm": stray,
            "height_mm": high, "retarget_s": turn, "lookalikes": lookalikes}


CALLS = {"b": "blocked", "g": "goal", "s": "short", "x": None}


def operator_call(camera: str | None, ask=input) -> str | None:
    """The operator's call on a roll: Enter keeps the camera's; b / g / s say blocked / goal / short; x says
    it doesn't count."""
    while True:
        a = ask(f"  camera: {camera or 'not judged'}. Enter to agree, or b / g / s (blocked / goal / short), "
                "x (doesn't count): ").strip().lower()
        if a == "":
            return camera
        if a in CALLS:
            return CALLS[a]


async def goalie(arm, cam, cfg: BallCfg, mode: str, rolls: int, speed: float | None = None,
                 out: Path | None = None, say=print, ask=None) -> dict:
    """Defend the lane: for each roll, wait for the ball to sit still at the far end, then track it and move
    the paddle (predict: to where it will cross; chase: to where it is). Every roll is logged (goalie.json).
    With `ask` (ball.py goalie --confirm), the operator confirms or corrects each call: both are logged,
    the operator's counts, and the report gives the camera's agreement with it."""
    table, lane, _, keeper = setup(cfg, mode)
    la = LaneArm(arm, lane, speed or cfg.lane_speed_deg_s, cfg.max_lane_speed_deg_s)
    problem = await la.off_lane()
    if problem:
        raise SystemExit(problem)
    if cfg.arm_speed_deg_s is not None and abs(cfg.arm_speed_deg_s - la.speed) > 0.5:
        say(f"Note: the paddle's timing was measured at {cfg.arm_speed_deg_s:.0f} deg/s, this runs at {la.speed:.0f}: "
            "run lane-test at this speed for the goalie to know when it gets there.")
    out = out or RUNS / f"{stamp()}-ball-goalie-{mode}"
    out.mkdir(parents=True, exist_ok=True)
    avoid: list = []
    stream = sightings(cam, table, cfg, avoid=avoid, lane=lane)
    logs = []
    try:
        await la.start()
        await la.approach(lane.length / 2, say=say)
        for i in range(1, rolls + 1):
            say(f"roll {i}/{rolls} ({mode})")
            avoid[:] = paddle_zone_px(table, lane, la.span(), cfg)
            await wait_for_ready(stream, lane, say=say)
            log = await defend(stream, lane, keeper, la, cfg, say=say, avoid=avoid, table=table)
            await la.settle()
            entry = {**log.to_json(), "camera": log.outcome}
            if ask is not None:
                entry["human"] = entry["outcome"] = operator_call(log.outcome, ask)
            logs.append(entry)
            (out / "goalie.json").write_text(json.dumps({"mode": mode, "decel": cfg.decel_mm_s2, "arm": cfg.arm,
                                                         "speed_deg_s": la.speed, "rolls": logs}, indent=1))
            await la.go(lane.length / 2)
        await la.settle()
    finally:
        await la.stop()
    blocked = sum(lg["outcome"] == "blocked" for lg in logs)
    counted = sum(lg["outcome"] in ("blocked", "goal") for lg in logs)
    say(f"\n{mode}: blocked {wilson_rate(blocked, counted)}   log: {out}")
    return {"mode": mode, "blocked": blocked, "counted": counted, "out": out, "rolls": logs}


# ---- command line ----------------------------------------------------------------------------------------

def _env(name: str) -> str:
    from worldarm import connection   # loads the env file: robot commands only
    return connection.env(name)


def _lane_speed(cfg: BallCfg, args) -> float:
    return min(getattr(args, "speed", None) or cfg.lane_speed_deg_s, cfg.max_lane_speed_deg_s)


async def cmd_check(machine, cfg: BallCfg, args) -> None:
    await check(Overhead(machine, cfg.camera), cfg)


async def cmd_lane(machine, cfg: BallCfg, args) -> None:
    await teach(Arm.from_robot(machine, _env("VIAM_ARM")), cfg, args.teach)


async def cmd_calibrate(machine, cfg: BallCfg, args) -> None:
    from worldarm.cell.config import CellConfig
    from worldarm.cell.rig import Rig
    rig = Rig(machine, CellConfig.load(cfg.machine), _env("VIAM_ARM"), _env("VIAM_GRIPPER"), _env("VIAM_CAMERA"))
    await calibrate(rig, Overhead(machine, cfg.camera), cfg)


async def cmd_lane_test(machine, cfg: BallCfg, args) -> None:
    cam = Overhead(machine, cfg.camera) if cfg.table() is not None else None
    await lane_test(Arm.from_robot(machine, _env("VIAM_ARM")), cfg, args.times, _lane_speed(cfg, args), cam)


async def cmd_record(machine, cfg: BallCfg, args) -> None:
    lane, paddle_s = cfg.get_lane(), None
    if lane is not None:                               # where the paddle rests (read only: nothing moves)
        p = await Arm.from_robot(machine, _env("VIAM_ARM")).get_end_position()
        s = lane.clamp(lane.along((p.x, p.y)))
        if np.linalg.norm(np.array([p.x, p.y]) - lane.point(s)) < 150.0:
            paddle_s = s
    await record(Overhead(machine, cfg.camera), cfg, args.rolls, paddle_s=paddle_s)


async def cmd_goalie(machine, cfg: BallCfg, args) -> None:
    await goalie(Arm.from_robot(machine, _env("VIAM_ARM")), Overhead(machine, cfg.camera), cfg, args.mode,
                 args.rolls, _lane_speed(cfg, args), ask=input if args.confirm else None)


ROBOT = {"check": cmd_check, "lane": cmd_lane, "calibrate": cmd_calibrate, "lane-test": cmd_lane_test,
         "record": cmd_record, "goalie": cmd_goalie}
MOVES = {"calibrate": "it carries the ball over a grid around the lane",
         "lane-test": "the paddle goes end to end along the lane (keep the ball off the table)",
         "goalie": "the paddle defends the lane"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="armfarm03")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    lane = sub.add_parser("lane")
    lane.add_argument("--teach", choices=["a", "b"], required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--yes", action="store_true", help="skip the 'type go' question")
    lt = sub.add_parser("lane-test")
    lt.add_argument("--times", type=int, default=2)
    lt.add_argument("--speed", type=float, help="deg/s along the lane (capped by max_lane_speed_deg_s)")
    lt.add_argument("--yes", action="store_true", help="skip the 'type go' question")
    rec = sub.add_parser("record")
    rec.add_argument("--rolls", type=int, default=15)
    sub.add_parser("fit")
    g = sub.add_parser("goalie")
    g.add_argument("--mode", choices=["predict", "chase"], default="predict")
    g.add_argument("--rolls", type=int, default=10)
    g.add_argument("--speed", type=float, help="deg/s along the lane (capped by max_lane_speed_deg_s)")
    g.add_argument("--yes", action="store_true", help="skip the 'type go' question")
    g.add_argument("--confirm", action="store_true", help="after each roll, confirm or correct the camera's call")
    sub.add_parser("report")
    s = sub.add_parser("sim")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--realistic", action="store_true", help="the harsher, lifelike profile (sim.SimCfg.realistic)")
    return p


async def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    cfg = BallCfg.load(args.machine)
    if args.cmd == "fit":
        fit(cfg)
        return
    if args.cmd == "report":
        report(cfg)
        return
    if args.cmd == "sim":
        sim(cfg, args.n, realistic=args.realistic)
        return
    from worldarm import connection   # loads the env file: robot commands only
    address = connection.env("VIAM_MACHINE_ADDRESS").split(".")[0]
    if cfg.machine.replace("0", "") not in address.replace("0", ""):
        raise SystemExit(f"config is for {cfg.machine} but the env file points at {address}")
    if args.cmd in MOVES and not args.yes:
        speed = f" at {_lane_speed(cfg, args):.0f} deg/s" if args.cmd != "calibrate" else ""
        if input(f"This MOVES the arm{speed}: {MOVES[args.cmd]}. Everyone clear of the arm, someone at the "
                 "E-stop? Type 'go': ").strip() != "go":
            raise SystemExit("Cancelled.")
    machine = await connection.connect()
    try:
        await ROBOT[args.cmd](machine, cfg, args)
    finally:
        await machine.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
