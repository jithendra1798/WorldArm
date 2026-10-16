# Viam machine configuration for WorldArm

Add these to **each** machine in the Viam app (CONFIGURE tab). The names here are the ones our code expects. The component names `arm`, `gripper` and `cam` are already configured on the Arm Farm machines.

## 1. Block detection: shape-finder → 3D objects

Add the registry module **`devrel:shape-finder`**, then add two vision services:

```json
{
  "name": "shape-detector",
  "api": "rdk:service:vision",
  "model": "devrel:shape-finder:detector",
  "attributes": { "camera_name": "cam" }
}
```

```json
{
  "name": "vision-segment",
  "api": "rdk:service:vision",
  "model": "viam:vision:detections-to-segments",
  "attributes": {
    "detector_name": "shape-detector",
    "camera_name": "cam",
    "mean_k": 5,
    "sigma": 1.25
  }
}
```

**To check it:** put a block under the camera from `table-view`, then open the Test panel for `shape-detector` and `vision-segment`. You should see labels like `square-blue` or `rectangle-yellow`.

**Limits:**
- shape-finder only finds **four-sided** shapes (squares and rectangles, from contour approximation).
- Its color ranges are fixed: red 0–10 and 170–179, yellow 26–34, green 35–85, blue 86–130 (OpenCV hue).

Cylinders, triangles and the round red knobs won't be detected. The perception track handles those.

## 2. Saved poses: `erh:vmodutils:arm-position-saver`

The `erh:vmodutils` module is already installed, because the obstacles use it. Add one switch component per pose, each with `"attributes": { "arm": "arm" }`:

| Switch name | Pose |
|---|---|
| `home-pose` | Safe pose, clear of everything |
| `table-view` | Camera pointing straight down, about 38 cm above the block area |
| `organizer-view` | Camera facing the drawer fronts from about 30 cm away |
| `knob-1-pre`, `knob-2-pre`, `knob-3-pre` | Gripper horizontal and **open**, about 5 cm in front of knob k (1 = top) |
| `drop-1`, `drop-2`, `drop-3` | Gripper above the exposed part of drawer k **when it's open** |

On Arm B, with bins instead of drawers, only `home-pose`, `table-view` and `drop-1..3` are needed (one drop above each bin).

**To save a pose:**
1. Move the arm there (manual mode, see below).
2. On the switch's card in CONTROL, click **update config** (position 1). That writes the current joints into the switch's config; doing it again overwrites them.
3. **go to** (position 2) replays the pose. From code, that's `set_position(2)`. Going to an unsaved switch does nothing.

**Caution:** the switch's `motion` attribute defaults to empty. In that case "go to" may be a direct joint move that **ignores obstacles**. Set `"motion": "builtin"` if you want it to plan around them, and test every new pose with someone on the E-stop.

## 3. Recorded drawer moves: `devrel:arm-recorder:recorder`

Add the registry module **`devrel:arm-recorder`**, then this sensor component. It's arm only, because the xArm gripper doesn't support `get_position`:

```json
{
  "name": "drawer-recorder",
  "api": "rdk:component:sensor",
  "model": "devrel:arm-recorder:recorder",
  "attributes": {
    "arm": "arm",
    "frequency_hz": 10,
    "max_velocity_rads_per_sec": 0.5,
    "max_acceleration_rads_per_sec": 0.5,
    "playback_interpolation_steps": 10
  }
}
```

**Recordings to make, for each drawer k = 1, 2, 3.** Send these DoCommands from the recorder's card in CONTROL while you guide the arm by hand:

| Session | Start | Move by hand | End |
|---|---|---|---|
| `grasp-k` | At `knob-k-pre`, gripper open | Straight forward until the fingers surround the knob's **stem** | Fingers around the stem (don't grab yet) |
| `pull-k` | Fingers closed on the stem (`teach.py` → `g`) | Straight back, 50–60% of the drawer's depth | Drawer open; then `o` to release |
| `push-k` | Gripper closed, touching the open drawer's front | Straight forward until the drawer is flush | Drawer closed |

```json
{"command": "start_recording", "session": "pull-1"}
{"command": "stop_recording"}
{"command": "list_sessions"}
{"command": "play", "session": "pull-1"}
```

**Notes:**
- Recordings replay **joint positions**, so they only work while the organizer stays exactly where it was taped. Making drawers work from any organizer position is the drawers track's job.
- The first move of every playback goes to the recording's start pose. Test playback with the drawer already open, and with someone on the E-stop.

## Manual mode: guiding the arm by hand

1. **Someone stands at the E-stop.**
2. **Support the arm with one hand**, then click CONTROL → arm card → **Enter manual mode**. The arm goes weightless; if the payload setting is off, it can drift.
3. Guide it by the wrist. Move slowly near the organizer.
4. Work the gripper from the laptop with `python teach.py` (`o` open, `g` grab, `h` holding?, `p` print pose, `c NAME` capture), or with the gripper card's Open and Grab buttons.
5. **Hold the arm still**, then click **Exit manual mode**. It holds its position again.

**Picking a block by hand:** guide the gripper above the block with the fingers lined up against two parallel faces, lower to mid-height, `g`, then `h` should say `holding: True`, then lift.

**Pulling a drawer by hand:** gripper horizontal, fingers around the knob stem behind the head, `g`, then pull straight back. Watch that the organizer doesn't slide.
