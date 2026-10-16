# Moving Blocks — Viam Generic Service

This module transfers arbitrary depth-separated objects from the occupied one of
two blue-tape zones to the empty zone. It uses the configured RealSense RGB-D
camera, the machine frame system through Viam Motion, and Motion's collision-aware
planner with the armfarm workcell obstacles.

## Safety behavior

- Creating the service and `preview` do not command any motion.
- `move` requires both `enable_motion: true` in service configuration and the
  explicit DoCommand `{"action":"move","confirm":true}`.
- A confirmed move first checks that the gripper is empty, then returns the arm to
  the configured overview pose, captures the scene, validates the two zones and
  every object target, and only then begins transfers.
- Motion refuses ambiguous zones, unsafe depth/table estimates, merged oversized
  foreground regions, targets outside the workspace, and grasps below the table
  safety floor.
- Destination approach/release preserves the camera-derived downward surface
  orientation (including its yaw). The prior script replaced that with a fixed
  yaw and Viam's planner returned `zero IK solutions` at the far drop zone.

## Machine service configuration

Add a Generic service using model `hackathons:moving-blocks:transfer` and set:

```json
{
  "arm": "arm",
  "camera": "cam",
  "gripper": "gripper",
  "motion": "builtin",
  "enable_motion": false,
  "view_pose": {
    "x": 381.0091571560523,
    "y": 611.9512703615794,
    "z": 616.1069475769257,
    "o_x": 0.07032923618761304,
    "o_y": 0.010971569348844546,
    "o_z": -0.9974634946720057,
    "theta": 8.627752805905791
  },
  "approach_clearance_mm": 100,
  "lift_mm": 100,
  "approach_speed_degs_per_sec": 8,
  "descent_step_mm": 25,
  "grasp_tcp_z_offset_mm": 65,
  "min_tcp_above_table_mm": 55,
  "table_keep_out_mm": 0,
  "grasp_ox": 0,
  "grasp_oy": 0,
  "grasp_oz": -1,
  "grasp_theta_deg": 0,
  "gripper_speed": 800,
  "gripper_torque_pct": 40
}
```

Use the service's Test panel to call `DoCommand` with `{"action":"preview"}`
and inspect its returned positions. Movement stays disabled until the operator
deliberately enables it in config and sends the confirmed `move` command.

## Local package

`./setup.sh` installs the Python dependencies in a module-local virtual
environment. `./build.sh` creates `dist/archive.tar.gz`. The module does not
connect to a robot from its entrypoint and does not move during startup.
