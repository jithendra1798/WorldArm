# pick-red-cube (Viam generic service module)

Runs on the robot. Reuses armfarm05 resources you already configured:

| Resource | Role |
| --- | --- |
| `cam` | RealSense RGB-D |
| `red-cube-detector` | builtin `color_detector` |
| `red-cube-segmenter` | `viam:detections-to-segments` |
| `builtin` | motion (static obstacles from table/walls/ceiling) |
| `arm` / `gripper` | xArm850 + UFactory gripper |

## Fastest path: Viam inline Control Code (workshop Phase 6)

1. Open **armfarm05** → **Configure** → **+** → **Control code** → **Viam-hosted** → Python.
2. Name it e.g. `pick-red-cube`.
3. **Do not change** the auto-generated `MODEL` or class name.
4. Replace the stub `validate_config` / `new` / `do_command` bodies with the logic from `src/models/picker.py` (same methods), adapting the class name to the generated one and keeping the generated `MODEL`.
5. **Save & Deploy**, wait for the cloud build.
6. Add a generic service of that model named `pick-red-cube` with the attributes JSON below.
7. On **Control**, open the service **DoCommand** card.

### Preview (no motion)

```json
{"action": "preview"}
```

### Pick (only after `enable_pick: true` in config)

```json
{"action": "pick"}
```

## Attributes (service config)

```json
{
  "arm": "arm",
  "gripper": "gripper",
  "camera": "cam",
  "detector": "red-cube-detector",
  "segmenter": "red-cube-segmenter",
  "motion": "builtin",
  "detect_label": "red-cube",
  "enable_pick": false,
  "approach_clearance_mm": 100,
  "lift_mm": 100,
  "approach_speed_degs_per_sec": 8,
  "descent_step_mm": 25,
  "grasp_tcp_z_offset_mm": -6,
  "min_tcp_above_table_mm": 15,
  "grasp_ox": 0,
  "grasp_oy": 0,
  "grasp_oz": -1,
  "grasp_theta_deg": 0,
  "gripper_speed": 800,
  "gripper_torque_pct": 40,
  "table_top_z_mm": -23
}
```

Leave `enable_pick` **false** until preview looks right, then set it to `true` and send `{"action":"pick"}`.

Arm speed: module calls `arm.do_command({"set_speed": ...})`. Persistent default is also `speed_degs_per_sec` on the arm resource in Viam (currently 60). Sleeps do not slow motion.

## Local / registry module (optional)

From this directory, with [Viam CLI](https://docs.viam.com/cli/) logged in:

```bash
chmod +x setup.sh run.sh build.sh
./setup.sh
# Hot-reload onto armfarm05 main part (id from machine Configure / metadata):
viam module reload \
  --part-id cf85a512-0675-455f-b52e-cdafd12c5c62 \
  --model-name hackathons:pick-red-cube:picker \
  --resource-name pick-red-cube
```

Then add the service attributes as above on the machine.

## Safety

- Default DoCommand action is `preview` (no motion).
- Physical pick requires **both** `enable_pick: true` in config **and** `{"action":"pick"}`.
- Failed grasp retreats; does not lift.
- Does not release the cube after lift.
