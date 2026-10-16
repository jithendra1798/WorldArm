# Deploy catch-thrown-object as Viam Control Code (armfarm05)

Soft-lob intercept only. People clear of the workcell. Hand on e-stop.

**Optional safety fragment** (corridor keepouts around the catch volume — import only for catch demos; do not leave it on the pick machine permanently):

https://app.viam.com/fragment/f72256c5-0856-4702-854d-c2a54975ec0e/configure/builder?org=a5994076-daba-41a6-9915-cbd310ff7d38

(`catch-thrown-object-safety` in Waj's Org — walls + ceiling keepouts matching the default catch AABB.)

## 0. Permission note (hackathons)

Creating Control Code (or `viam module create`) on **hackathons** requires org permission `write_private_registry_item`. Without it, Create module fails with PermissionDenied even when logged in. Ask a hackathons org admin to grant that, then continue.

Already on **armfarm05** (wired for this arm): `arm`, `cam`, `gripper`, workcell obstacles, `red-object-detector`, `red-object-segmenter`.

## 1. Create Control Code

1. [app.viam.com](https://app.viam.com) → **hackathons** → **Arm Farm** → **armfarm05**
2. **Configure** → **+** → **Control code**
3. **Viam-hosted**, language **Python**
4. Name: `catch-thrown-object`
5. **Create module**

## 2. Paste the logic

In the generated `src/models/generic_service.py` (name may vary):

- **Keep** the generated class name and `MODEL = ...` line exactly as Viam created them.
- Copy helpers + methods from `src/models/catcher.py` in this folder (`_req_str`, `_num`, `_clamp_down`, class body).
- Also add `src/models/workcell_obstacles.py` (or paste `TABLE_TOP_Z_MM` + `workcell_world_state` into the same file).
- **Save & Deploy**. Wait for the build. Check **Logs**.

## 3. Service attributes (safe defaults)

```json
{
  "arm": "arm",
  "gripper": "gripper",
  "camera": "cam",
  "detector": "red-object-detector",
  "segmenter": "red-object-segmenter",
  "motion": "builtin",
  "detect_label": "red",
  "enable_catch": false,
  "arm_speed_degs_per_sec": 10,
  "max_object_speed_mm_s": 700,
  "max_session_sec": 6,
  "min_tcp_above_table_mm": 70,
  "close_radius_mm": 80,
  "lead_time_sec": 0.25,
  "catch_x_min_mm": 150,
  "catch_x_max_mm": 450,
  "catch_y_min_mm": 250,
  "catch_y_max_mm": 550,
  "catch_z_min_mm": 50,
  "catch_z_max_mm": 280,
  "ready_x_mm": 300,
  "ready_y_mm": 400,
  "ready_z_mm": 180,
  "grasp_ox": 0,
  "grasp_oy": 0,
  "grasp_oz": -1,
  "grasp_theta_deg": 0,
  "gripper_speed": 800,
  "gripper_torque_pct": 40
}
```

Change detector/segmenter/label for other colors (`blue-object-detector`, label `blue`, etc.).

## 4. CONTROL tab DoCommands

**Preview (no motion):**
```json
{"action": "preview"}
```

**Ready pose** (only after `"enable_catch": true`):
```json
{"action": "arm_ready"}
```

**Catch attempt** (enable_catch true + ack):
```json
{
  "action": "catch",
  "safety_ack": "I_UNDERSTAND_CATCH_RISKS"
}
```

**Abort:**
```json
{"action": "abort"}
```

## Safety behavior baked in

| Guard | Behavior |
|-------|----------|
| `enable_catch` | Default false; no motion without it |
| `safety_ack` | Exact phrase required for `catch` |
| Arm speed | Hard-capped at 15 deg/s |
| Object speed | Refuses tracks faster than ~900 mm/s |
| Session | Hard max 8 s then stop + open |
| Catch volume | TCP clamped to small AABB above table |
| Obstacles | Table / walls / ceiling in every Move |
| Miss / fail | Stop arm, open gripper, retreat to ready |
