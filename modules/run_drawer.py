"""Run the drawer service from this laptop instead of as an inline module on the machine.

Builds MyGenericService from drawer_service.py with the machine's own arm, gripper, camera,
vision and motion clients, then sends it one DoCommand. MOVES THE REAL ARM (asks first).

    WORLDARM_ENV=.env.armfarm03 python modules/run_drawer.py '{"action": "find_knobs"}'
    WORLDARM_ENV=.env.armfarm03 python modules/run_drawer.py '{"action": "locate"}'
    WORLDARM_ENV=.env.armfarm03 python modules/run_drawer.py '{"action": "drawer_run", "drawer": 1}'

Over the network the joint-load polling during the descent is slower than on the machine,
so contact is noticed a little later. Keep a hand on the E-stop.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from modules.drawer_service import MyGenericService
from worldarm.connection import connect


def build(machine) -> MyGenericService:
    svc = MyGenericService("drawer-local")
    svc.gripper_name, svc.camera_name = "gripper", "cam"
    svc.arm = Arm.from_robot(machine, "arm")
    svc.gripper = Gripper.from_robot(machine, "gripper")
    svc.camera = Camera.from_robot(machine, "cam")
    svc.knob_seg = VisionClient.from_robot(machine, "knob-segmenter")
    svc.knob_det = VisionClient.from_robot(machine, "knob-color")
    svc.motion = MotionClient.from_robot(machine, "builtin")
    svc.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawer_last_look.json")
    return svc


async def check_corrections(svc: MyGenericService) -> dict:
    """No touching: one overhead look, then the two-look (180 deg) correction on every knob.
    Shows how far 'overhead + OVERHEAD_CORRECTION' is from the self-corrected position per drawer."""
    from modules.drawer_service import OVERHEAD_CORRECTION
    fk = await svc.find_knobs()
    if not fk.get("found"):
        return {"find": fk}
    pull = fk["open_dir"]
    wrist, wrist_alt = svc._wrist_for(fk["row_dir"], [-pull[0], -pull[1]])
    corr = OVERHEAD_CORRECTION.get(float(fk.get("look_wrist") or 0.0), (0.0, 0.0))
    rows = []
    for k in fk["knobs"]:
        ref = await svc._refine_knob(fk, k["drawer"], pull, wrist, wrist_alt)
        est = [k["x"] + corr[0], k["y"] + corr[1]]
        row = {"drawer": k["drawer"], "overhead": [k["x"], k["y"]], "overhead+corr": [round(v, 1) for v in est]}
        if ref.get("ok"):
            true = [ref["x"], ref["y"]]
            row.update({"two_look": [round(v, 1) for v in true],
                        "true_minus_overhead": [round(true[0] - k["x"], 1), round(true[1] - k["y"], 1)],
                        "aim_error_mm": [round(est[0] - true[0], 1), round(est[1] - true[1], 1)]})
        else:
            row["two_look_error"] = ref.get("error")
        rows.append(row)
    return {"look_wrist": fk.get("look_wrist"), "correction_used": list(corr), "row_dir": fk["row_dir"],
            "drawers": rows}


async def main():
    command = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"action": "find_knobs"}
    print(f"Command: {command}  (machine: {os.getenv('VIAM_MACHINE_ADDRESS', '?')})")
    if input("This MOVES THE REAL ARM. Someone on the E-stop? Type 'yes': ").strip() != "yes":
        print("Cancelled.")
        return
    async with await connect() as machine:
        svc = build(machine)
        if command.get("action") == "check_corrections":
            result = await check_corrections(svc)
        else:
            result = await svc.do_command(command)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
