"""Synthetic top-down wrist-camera frames for tests: table, blocks, and the 3-drawer organizer with each
drawer shut, partly or fully out, and blocks inside. Real perspective and depth (nearest surface wins)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from worldarm.see.frame import Frame, Intrinsics

TABLE = (225, 225, 225)
CREAM = (175, 205, 220)      # the organizer's plastic: low saturation (S ~51)
FLOOR = (46, 78, 100)        # inside a drawer, in shadow: yellow hue, saturation 138
KNOB = (30, 30, 200)
PAINT = {"green": (40, 160, 40), "yellow": (30, 200, 230), "blue": (200, 90, 30), "black": (15, 15, 15),
         "orange": (20, 120, 240)}


class Scene3D:
    """Paint prisms and balls into color + depth for a camera looking straight down."""

    def __init__(self, cam_xy=(260.0, -420.0), cam_z=620.0, f=900.0, size=(1280, 720), rot=None):
        self.cam = np.array([cam_xy[0], cam_xy[1], cam_z], float)
        self.w, self.h = size
        self.intr = Intrinsics(f, f, self.w / 2.0, self.h / 2.0, self.w, self.h)
        # camera x = world x, y = -world y; any other turn about the vertical keeps it looking straight down
        self.rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]) if rot is None else np.asarray(rot, float)
        self.color = np.full((self.h, self.w, 3), TABLE, np.uint8)
        self.depth = np.full((self.h, self.w), int(cam_z), np.uint16)    # table at z = 0

    def project(self, x, y, z):
        cam = self.rot.T @ (np.array([x, y, z], float) - self.cam)
        return [self.intr.focal_x_px * cam[0] / cam[2] + self.intr.center_x_px,
                self.intr.focal_y_px * cam[1] / cam[2] + self.intr.center_y_px]

    def prism(self, corners_xy, z_top, bgr):
        poly = np.array([self.project(x, y, z_top) for x, y in corners_xy], np.int32)
        x0, y0 = np.clip(poly.min(axis=0), 0, [self.w, self.h])
        x1, y1 = np.clip(poly.max(axis=0) + 1, 0, [self.w, self.h])
        if x0 >= x1 or y0 >= y1:
            return
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)       # only the polygon's bounding box
        cv2.fillPoly(mask, [poly - [x0, y0]], 1)
        d = int(round(self.cam[2] - z_top))
        depth = self.depth[y0:y1, x0:x1]
        paint = (mask > 0) & (d < depth)
        depth[paint] = d
        self.color[y0:y1, x0:x1][paint] = bgr

    def rect(self, center, along_vec, size_along, size_across, z_top, bgr):
        c, a = np.asarray(center, float), np.asarray(along_vec, float)
        b = np.array([-a[1], a[0]])
        corners = [c + sa * size_along / 2 * a + sb * size_across / 2 * b for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        self.prism(corners, z_top, bgr)

    def ball(self, center, r, bgr):
        cx, cy, cz = center
        u0, v0 = self.project(cx, cy, cz)
        rad = int(self.intr.focal_x_px * r / (self.cam[2] - cz - r)) + 12
        v_lo, v_hi = max(0, int(v0) - rad), min(self.h, int(v0) + rad + 1)
        u_lo, u_hi = max(0, int(u0) - rad), min(self.w, int(u0) + rad + 1)
        if v_lo >= v_hi or u_lo >= u_hi:
            return
        vv, uu = np.mgrid[v_lo:v_hi, u_lo:u_hi]
        rays = np.stack([(uu - self.intr.center_x_px) / self.intr.focal_x_px,
                         (vv - self.intr.center_y_px) / self.intr.focal_y_px, np.ones(uu.shape)], axis=-1) @ self.rot.T
        oc = self.cam - np.asarray(center, float)
        aa = np.einsum("...i,...i", rays, rays)
        b = rays @ oc
        disc = b * b - aa * (oc @ oc - r * r)
        hit = disc >= 0
        t = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0.0))) / aa, np.inf)
        d = np.round(t)  # camera z of the hit (each ray has unit camera z)
        patch = self.depth[v_lo:v_hi, u_lo:u_hi]
        paint = hit & (d < patch)
        patch[paint] = d[paint].astype(np.uint16)
        self.color[v_lo:v_hi, u_lo:u_hi][paint] = bgr

    def frame(self) -> Frame:
        return Frame(self.color.copy(), self.depth.copy(), self.intr, self.rot.copy(), self.cam.copy())


@dataclass
class Box:
    """The organizer: 3 drawers side by side. `front` is the middle of the box's front face (closed drawers'
    faces), `pull` the way drawers come out (unit), drawers numbered along `row` = left-hand order."""
    front: tuple = (190.0, -480.0)
    pull_deg: float = 150.0           # drawers come out toward the robot side
    spacing: float = 75.0
    depth: float = 200.0
    height: float = 95.0
    knob_z: float = 44.0
    knob_r: float = 11.5
    protrusion: float = 20.0
    openings: tuple = (0.0, 0.0, 0.0)  # per physical slot, in `slots` order
    rim: float = 70.0
    floor: float = 10.0
    face: float = 10.0
    wall: float = 6.0
    contents: dict = field(default_factory=dict)  # slot -> [(color, (along, from_face), (w, l), h)]

    @property
    def pull(self) -> np.ndarray:
        r = math.radians(self.pull_deg)
        return np.array([math.cos(r), math.sin(r)])

    @property
    def row(self) -> np.ndarray:
        """Direction from drawer 1 to drawer 3 (drawer 1 is leftmost for someone facing the fronts)."""
        p = self.pull
        return -np.array([p[1], -p[0]])

    def slot_along(self, i: int) -> float:
        return (i - 1) * self.spacing        # slot 0 = drawer 1

    def knob(self, i: int) -> np.ndarray:
        c = np.asarray(self.front) + self.slot_along(i) * self.row + (self.openings[i] + self.protrusion) * self.pull
        return np.array([c[0], c[1], self.knob_z])

    def draw(self, s: Scene3D):
        f, p, r = np.asarray(self.front, float), self.pull, self.row
        half_row = 1.5 * self.spacing + 5
        s.rect(f - p * self.depth / 2, r, 2 * half_row, self.depth, self.height, CREAM)  # box body
        for i in range(3):
            o = self.openings[i]
            a = self.slot_along(i)
            centre = f + a * r
            if o > 1.0:
                inner = self.spacing - 2 * self.wall
                s.rect(centre + p * (o - self.face) / 2, p, o - self.face, inner, self.floor, FLOOR)
                for side in (-1, 1):
                    s.rect(centre + side * (self.spacing / 2 - self.wall / 2) * r + p * (o - self.face) / 2,
                           p, o - self.face, self.wall, self.rim, CREAM)
                for color, (along, from_face), (w, l), h in self.contents.get(i, []):
                    c = centre + along * r + (o - self.face - from_face) * p
                    s.rect(c, p, l, w, self.floor + h, PAINT[color])
            s.rect(centre + p * (o - self.face / 2), r, self.spacing - 1, self.face, self.height - 5, CREAM)  # front
            s.ball(self.knob(i), self.knob_r, KNOB)


def block(s: Scene3D, color, center, size=(33.0, 41.0), long_deg=20.0, height=60.0):
    a = math.radians(long_deg)
    s.rect(center, (math.cos(a), math.sin(a)), size[1], size[0], height, PAINT[color])


def scene(box: Box | None = None, blocks=(), **cam):
    s = Scene3D(**cam)
    for b in blocks:
        block(s, *b) if isinstance(b, tuple) else block(s, **b)
    if box is not None:
        box.draw(s)
    return s
