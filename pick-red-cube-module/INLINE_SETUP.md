# Deploy as Viam Control Code (recommended for the hackathon)

This matches [Phase 6: Inline module](https://docs.viam.com/tutorials/pick-and-place/inline-module/).

## 1. Create the inline module on armfarm05

1. Open https://app.viam.com → **hackathons** → **Arm Farm** → **armfarm05**
2. **Configure** → **+** → **Control code**
3. Choose **Viam-hosted**, language **Python**
4. Name: `pick-red-cube`
5. Click **Create module**

## 2. Paste the logic

In the generated `src/models/generic_service.py` (name may vary):

- **Keep** the generated `class ...` name and `MODEL = ...` line exactly as Viam created them.
- Replace `validate_config`, `new`, and `do_command` with the versions from `src/models/picker.py` (copy method bodies / adapt `self.` fields).
- Keep these imports (add any missing):

```python
import asyncio
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple
from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.services.vision import Vision
from viam.utils import ValueTypes, struct_to_dict
```

Helper functions `_req_str`, `_num`, `_pose_dict` from `picker.py` can live above the class.

4. Click **Save & Deploy** and wait for the build (~1–3 minutes). Check **Logs**.

## 3. Configure the service attributes

On the generic service card:

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

Save.

## 4. Run from the Control tab

**Preview (no motion):**

```json
{"action": "preview"}
```

**Pick** (after you set `"enable_pick": true` and Save):

```json
{"action": "pick"}
```

Or from your laptop (after the service is online):

```bash
cd /Users/waj/Desktop/Viam-Hackathon/WorldArm
source ../.venv/bin/activate
python run_pick_module.py
python run_pick_module.py --pick   # only with enable_pick true
```
