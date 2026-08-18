# SPDX-License-Identifier: BSD-3-Clause
"""A small language for stating what a task is trying to achieve.

Instead of hand-editing command ranges in Python, a config says::

    pick[cube_red] place[0.20, 0.0, 0.12]
    pick[random] place[random(0.20, 0.0, 0.12, r0.06)]
    pick[random(cube_red, cube_blue)] place[box(0.20, 0.0, 0.12, 0.08, 0.16, 0.06)]

Grammar::

    objective := pick '[' selector ']' place '[' region ']'      (':' after pick/place optional)

    selector  := NAME                      one specific object
               | 'random'                  uniform over the pickable list
               | 'random(' NAME, ... ')'   uniform over a named subset

    region    := x, y, z                              a fixed point
               | 'random(' x, y, z, 'r' R ')'         uniform in a disc of radius R at height z
               | 'box(' x, y, z, dx, dy, dz ')'       uniform in a box of those half-extents

Every region is checked against the arm's reachable envelope at parse time. This is the whole
reason the module validates rather than just parses: a target the arm cannot reach produces a
task that trains to a flat reward and looks like a broken policy. A radius of 1 m on a 0.30 m
arm is a plausible thing to write and an expensive thing to discover after an hour of training.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# Reachable envelope for the SO-ARM101, base at the origin. MEASURED, not estimated:
# scripts/measure_workspace.py drives 1500 random joint configurations and reads where the
# gripper actually ends up. 1st-99th percentile of that sample, floored at the table.
#   radial: distance from the base axis in the xy plane
#   height: z above the table surface
# Absolute extremes were 0.370 m reach and 0.460 m height, but those are single fully-extended
# configurations; percentiles give a boundary tasks can actually be written against.
REACH_RADIAL = (0.02, 0.35)   # m
REACH_HEIGHT = (0.0, 0.45)    # m

# A top-down grasp needs the gripper pointing down, which only 28% of sampled configurations do,
# and it shrinks the usable region to radial <= 0.33 m. Not enforced here -- a place target is a
# position, not a grasp -- but a pick location outside it is unlikely to be graspable from above.
TOPDOWN_RADIAL_MAX = 0.33     # m


class ObjectiveError(ValueError):
    """Raised for a malformed or unreachable objective."""


# --------------------------------------------------------------------------- regions


@dataclass
class Point:
    """A single fixed position."""

    x: float
    y: float
    z: float

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return (self.x, self.x), (self.y, self.y), (self.z, self.z)

    def ranges(self) -> dict[str, tuple[float, float]]:
        return {"pos_x": (self.x, self.x), "pos_y": (self.y, self.y), "pos_z": (self.z, self.z)}

    def describe(self) -> str:
        return f"point({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class Disc:
    """Uniform over a disc in the xy plane, at a fixed height."""

    x: float
    y: float
    z: float
    r: float

    def bounds(self):
        return (self.x - self.r, self.x + self.r), (self.y - self.r, self.y + self.r), (self.z, self.z)

    def ranges(self) -> dict[str, tuple[float, float]]:
        # Isaac Lab command ranges are axis-aligned, so the disc is expressed as its bounding
        # square. Sampling is therefore over the square, not the disc -- corners included. Kept
        # deliberately: an inscribed square would silently shrink the region the config asked
        # for, and for a workspace this small the difference is under a centimetre.
        return {
            "pos_x": (self.x - self.r, self.x + self.r),
            "pos_y": (self.y - self.r, self.y + self.r),
            "pos_z": (self.z, self.z),
        }

    def describe(self) -> str:
        return f"disc(centre=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}), r={self.r:.3f})"


@dataclass
class Box:
    """Uniform over a box, given as centre and half-extents."""

    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float

    def bounds(self):
        return (self.x - self.dx, self.x + self.dx), (self.y - self.dy, self.y + self.dy), (
            self.z - self.dz,
            self.z + self.dz,
        )

    def ranges(self) -> dict[str, tuple[float, float]]:
        bx, by, bz = self.bounds()
        return {"pos_x": bx, "pos_y": by, "pos_z": bz}

    def describe(self) -> str:
        return (
            f"box(centre=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}), "
            f"half=({self.dx:.3f}, {self.dy:.3f}, {self.dz:.3f}))"
        )


Region = Point | Disc | Box


def to_root_frame(region: Region, rot_xyzw) -> tuple[Region, list[str]]:
    """Re-express an env-frame region in the robot's root frame.

    Everything else a config states -- ``spawn``, ``scene.robot.pos``, a camera's ``pos`` and
    ``look_at`` -- is in the environment frame. Isaac Lab's pose command ranges are in the robot
    *root* frame, and every config here rotates the base 90 degrees about z, so writing a place
    target straight into those ranges puts the goal 90 degrees around from where the config says.
    Measured before this existed: ``place[0.20, 0.0, 0.10]`` landed at env (0.000, 0.200, 0.100),
    0.297 m from a cube the same file spawned at (0.200, 0.000, 0.010).

    Returns the converted region and any warnings.
    """
    qx, qy, qz, qw = (float(c) for c in rot_xyzw)
    if abs(qx) > 1e-6 or abs(qy) > 1e-6:
        raise ObjectiveError(
            f"base rotation {tuple(rot_xyzw)} tilts the robot out of the xy plane; a place region "
            "is an axis-aligned box and cannot be expressed in that root frame. State the goal "
            "with an explicit command range in Python instead."
        )

    theta = 2.0 * math.atan2(qz, qw)          # root -> env, about z
    c, s = math.cos(-theta), math.sin(-theta)  # inverse: env -> root
    cx = c * region.x - s * region.y
    cy = s * region.x + c * region.y

    warnings: list[str] = []
    if isinstance(region, Point):
        return Point(cx, cy, region.z), warnings
    if isinstance(region, Disc):
        # A disc is invariant under rotation about its own axis; only the centre moves.
        return Disc(cx, cy, region.z, region.r), warnings

    # A rotated box is not axis-aligned, so take its axis-aligned envelope. Exact for the
    # quarter-turns that actually occur (the extents just swap); a superset otherwise.
    dx = abs(c) * region.dx + abs(s) * region.dy
    dy = abs(s) * region.dx + abs(c) * region.dy
    if min(abs(c), abs(s)) > 1e-6:
        warnings.append(
            f"base is rotated {math.degrees(theta):.1f} deg, which is not a quarter turn; the box "
            f"place region is widened to its axis-aligned envelope "
            f"({region.dx:.3f}, {region.dy:.3f}) -> ({dx:.3f}, {dy:.3f})"
        )
    return Box(cx, cy, region.z, dx, dy, region.dz), warnings


@dataclass
class Objective:
    """A parsed ``pick[...] place[...]`` statement."""

    pick_names: list[str]          # candidates; length 1 means a fixed choice
    pick_random: bool
    place: Region
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        pick = self.pick_names[0] if not self.pick_random else f"random({', '.join(self.pick_names)})"
        return f"pick {pick} -> place {self.place.describe()}"


# --------------------------------------------------------------------------- parsing

_STMT = re.compile(
    r"^\s*pick\s*:?\s*\[(?P<pick>[^\]]*)\]\s*place\s*:?\s*\[(?P<place>[^\]]*)\]\s*$",
    re.IGNORECASE,
)
_RANDOM_CALL = re.compile(r"^random\s*\((?P<args>.*)\)$", re.IGNORECASE)
_BOX_CALL = re.compile(r"^box\s*\((?P<args>.*)\)$", re.IGNORECASE)


def _floats(parts: list[str], n: int, where: str) -> list[float]:
    if len(parts) != n:
        raise ObjectiveError(f"{where}: expected {n} numbers, got {len(parts)} ({', '.join(parts) or 'none'})")
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            raise ObjectiveError(f"{where}: {p!r} is not a number") from None
    return out


def _parse_pick(text: str, pickable: list[str]) -> tuple[list[str], bool]:
    text = text.strip()
    if not text:
        raise ObjectiveError("pick[...] is empty")

    if m := _RANDOM_CALL.match(text):
        args = [a.strip() for a in m.group("args").split(",") if a.strip()]
        names = args or list(pickable)
        if not names:
            raise ObjectiveError("pick[random(...)] has no candidates and no 'pickable' list is declared")
    elif text.lower() == "random":
        if not pickable:
            raise ObjectiveError("pick[random] needs objective.pickable to list the candidate objects")
        names, args = list(pickable), []
    else:
        names, args = [text], [text]

    if pickable:
        unknown = [n for n in names if n not in pickable]
        if unknown:
            raise ObjectiveError(f"pick names {unknown} are not in objective.pickable {pickable}")
    return names, len(names) > 1 or text.lower().startswith("random")


def _parse_region(text: str) -> Region:
    text = text.strip()
    if not text:
        raise ObjectiveError("place[...] is empty")

    if m := _BOX_CALL.match(text):
        v = _floats([a.strip() for a in m.group("args").split(",")], 6, "place[box(...)]")
        if min(v[3:]) < 0:
            raise ObjectiveError("place[box(...)]: half-extents must be non-negative")
        return Box(*v)

    if m := _RANDOM_CALL.match(text):
        parts = [a.strip() for a in m.group("args").split(",")]
        if len(parts) != 4:
            raise ObjectiveError(
                "place[random(...)]: expected random(x, y, z, r<R>), e.g. random(0.20, 0.0, 0.12, r0.06)"
            )
        radius_tok = parts[3]
        if not radius_tok.lower().startswith("r"):
            raise ObjectiveError(f"place[random(...)]: radius must be written r<R>, got {radius_tok!r}")
        x, y, z = _floats(parts[:3], 3, "place[random(...)]")
        r = _floats([radius_tok[1:]], 1, "place[random(...)] radius")[0]
        if r < 0:
            raise ObjectiveError("place[random(...)]: radius must be non-negative")
        return Disc(x, y, z, r)

    return Point(*_floats([a.strip() for a in text.split(",")], 3, "place[...]"))


def _sample_region(region: Region, n: int = 4000) -> list[tuple[float, float, float]]:
    """Deterministic sample of a region, used to estimate how much of it the arm can reach."""
    import random as _random

    rng = _random.Random(0)  # fixed: a config must validate the same way every run
    pts = []
    if isinstance(region, Point):
        return [(region.x, region.y, region.z)]
    for _ in range(n):
        if isinstance(region, Disc):
            # sqrt for uniform area density; sampling the radius directly clusters at the centre
            rad = region.r * math.sqrt(rng.random())
            ang = rng.uniform(0.0, 2.0 * math.pi)
            pts.append((region.x + rad * math.cos(ang), region.y + rad * math.sin(ang), region.z))
        else:
            pts.append((
                rng.uniform(region.x - region.dx, region.x + region.dx),
                rng.uniform(region.y - region.dy, region.y + region.dy),
                rng.uniform(region.z - region.dz, region.z + region.dz),
            ))
    return pts


def reachable_fraction(region: Region) -> float:
    """Fraction of the region inside the arm's envelope, by sampling."""
    rmin, rmax = REACH_RADIAL
    zmin, zmax = REACH_HEIGHT
    pts = _sample_region(region)
    ok = sum(1 for x, y, z in pts if rmin <= math.hypot(x, y) <= rmax and zmin <= z <= zmax)
    return ok / len(pts)


def _check_reach(region: Region, label: str) -> list[str]:
    """Reject regions the arm mostly cannot reach; warn when it can only partly reach them.

    Judged by the fraction of the region that is reachable, not by its nearest and furthest
    points. A disc centred on the base looks fine by those measures -- its nearest point is zero
    away -- while almost none of it is actually reachable. ``place[random(0,0,0,r1)]`` is exactly
    that case, and rejecting it is the main thing this function is for.
    """
    frac = reachable_fraction(region)
    zmin, zmax = REACH_HEIGHT
    rmin, rmax = REACH_RADIAL

    if frac < 0.20:
        raise ObjectiveError(
            f"{label} is only {frac * 100:.0f}% reachable; the arm works between "
            f"{rmin:.2f} and {rmax:.2f} m from its base and up to z={zmax:.2f} m.\n"
            f"  region: {region.describe()}\n"
            f"  most episodes would be impossible, so the task could not train."
        )

    warnings: list[str] = []
    if frac < 0.90:
        warnings.append(
            f"{label} is {frac * 100:.0f}% reachable; the remaining "
            f"{(1 - frac) * 100:.0f}% of episodes cannot succeed"
        )
    return warnings


def parse_region(text: str, label: str = "region", validate_reach: bool = True):
    """Parse a standalone region (used by ``objective.spawn``); returns (region, warnings)."""
    region = _parse_region(text)
    return region, (_check_reach(region, label) if validate_reach else [])


def parse_objective(text: str, pickable: list[str] | None = None, validate_reach: bool = True) -> Objective:
    """Parse a ``pick[...] place[...]`` statement, checking it against the arm's envelope."""
    if not isinstance(text, str):
        raise ObjectiveError(f"objective.sequence must be a string, got {type(text).__name__}")
    m = _STMT.match(text)
    if not m:
        raise ObjectiveError(
            f"could not parse objective {text!r}.\n"
            "  expected:  pick[<name>|random|random(a,b)] place[x,y,z | random(x,y,z,r<R>) | "
            "box(x,y,z,dx,dy,dz)]\n"
            "  example :  pick[random] place[random(0.20, 0.0, 0.12, r0.06)]"
        )

    names, is_random = _parse_pick(m.group("pick"), list(pickable or []))
    region = _parse_region(m.group("place"))
    warnings = _check_reach(region, "place region") if validate_reach else []
    return Objective(pick_names=names, pick_random=is_random, place=region, source=text, warnings=warnings)


def apply_objective(env_cfg, obj: Objective, spawn: Region | None = None) -> Any:
    """Write the parsed regions into the task config.

    ``place`` sets the goal command ranges. ``spawn`` sets where the object is reset to. Both are
    written by a config in the environment frame, and each needs a different conversion to get
    there.

    Place needs a frame change: command ranges are in the robot root frame, which is rotated 90
    degrees from the environment frame in every config here. See :func:`to_root_frame`.

    Spawn needs an origin change: Isaac Lab's ``reset_root_state_uniform`` takes an offset from
    the asset's default pose, not an absolute position. Writing absolute coordinates straight
    into ``pose_range`` would offset them by the object's spawn position a second time, putting
    the cube somewhere neither the author nor the reward function expects.
    """
    commands = getattr(env_cfg, "commands", None)
    cmd = getattr(commands, "object_pose", None) or getattr(commands, "ee_pose", None)
    if cmd is None:
        raise ObjectiveError("task has no object_pose or ee_pose command term to apply an objective to")

    robot = getattr(env_cfg.scene, "robot", None)
    rot = getattr(getattr(robot, "init_state", None), "rot", None) or (0.0, 0.0, 0.0, 1.0)
    place, warnings = to_root_frame(obj.place, rot)
    obj.warnings.extend(warnings)
    for key, value in place.ranges().items():
        setattr(cmd.ranges, key, value)

    if spawn is not None:
        target = getattr(env_cfg.scene, "object", None)
        if target is None:
            raise ObjectiveError("objective.spawn is set but the task has no scene.object to place")
        base = tuple(target.init_state.pos)
        (x0, x1), (y0, y1), (z0, z1) = spawn.bounds()
        event = getattr(getattr(env_cfg, "events", None), "reset_object_position", None)
        if event is None:
            raise ObjectiveError("task has no reset_object_position event to apply objective.spawn to")
        event.params["pose_range"] = {
            "x": (x0 - base[0], x1 - base[0]),
            "y": (y0 - base[1], y1 - base[1]),
            "z": (z0 - base[2], z1 - base[2]),
        }
    return env_cfg


def demo() -> None:
    """Self-check: the grammar, and the reach guard that is the point of it."""
    pickable = ["cube_red", "cube_blue"]

    # Frame conversion: place is written env-frame, commands want the root frame.
    q90 = (0.0, 0.0, 0.70710678, 0.70710678)          # 90 deg about z, what every config uses
    r, w = to_root_frame(Point(0.0, 0.20, 0.10), q90)
    assert not w and abs(r.x - 0.20) < 1e-6 and abs(r.y) < 1e-6 and abs(r.z - 0.10) < 1e-6, r
    r, w = to_root_frame(Point(0.20, 0.0, 0.10), (0.0, 0.0, 0.0, 1.0))
    assert abs(r.x - 0.20) < 1e-6 and abs(r.y) < 1e-6, r          # identity is a no-op
    r, _ = to_root_frame(Disc(0.0, 0.20, 0.12, 0.06), q90)
    assert abs(r.x - 0.20) < 1e-6 and abs(r.r - 0.06) < 1e-6, r   # radius survives rotation
    r, w = to_root_frame(Box(0.0, 0.20, 0.12, 0.03, 0.08, 0.02), q90)
    assert not w and abs(r.dx - 0.08) < 1e-6 and abs(r.dy - 0.03) < 1e-6, r   # extents swap
    r, w = to_root_frame(Box(0.0, 0.2, 0.1, 0.04, 0.02, 0.01), (0.0, 0.0, 0.3827, 0.9239))
    assert w and r.dx > 0.04, (r, w)                  # 45 deg: envelope, and it says so
    try:
        to_root_frame(Point(0.1, 0.0, 0.1), (0.7071, 0.0, 0.0, 0.7071))
    except ObjectiveError:
        pass
    else:
        raise AssertionError("a base tilted out of the xy plane should be rejected")

    o = parse_objective("pick[cube_red] place[0.20, 0.0, 0.12]", pickable)
    assert o.pick_names == ["cube_red"] and not o.pick_random
    assert isinstance(o.place, Point) and not o.warnings

    o = parse_objective("pick[random] place[random(0.20, 0.0, 0.12, r0.06)]", pickable)
    assert o.pick_random and o.pick_names == pickable
    assert isinstance(o.place, Disc) and o.place.r == 0.06
    assert o.place.ranges()["pos_x"] == (0.14, 0.26)

    # the colon form from the original sketch
    o = parse_objective("pick:[cube_blue]place:[0.20,0.0,0.12]", pickable)
    assert o.pick_names == ["cube_blue"]

    o = parse_objective("pick[random(cube_red)] place[box(0.20, 0.0, 0.12, 0.05, 0.10, 0.03)]", pickable)
    assert isinstance(o.place, Box) and o.place.ranges()["pos_y"] == (-0.10, 0.10)

    def fails(text, why, names=pickable):
        try:
            parse_objective(text, names)
        except ObjectiveError:
            return
        raise AssertionError(f"should have been rejected ({why}): {text}")

    # The example from the original sketch: r1 is a 1 m radius on a 0.30 m arm.
    fails("pick[random] place[random(0, 0, 0, r1)]", "1 m radius, unreachable centre")
    fails("pick[apple] place[0.20, 0.0, 0.12]", "apple is not in pickable")
    fails("pick[random] place[0.9, 0.0, 0.12]", "0.9 m is out of reach")
    fails("pick[random] place[0.20, 0.0, 0.9]", "0.9 m high")
    fails("pick[cube_red] place[0.2, 0.0]", "only two coordinates")
    fails("pick[cube_red] place[random(0.2, 0.0, 0.1, 0.06)]", "radius missing the r prefix")
    fails("grab[cube_red] drop[0,0,0]", "not the grammar")

    # Partly reachable is a warning, not an error: the config may want the hard episodes.
    # Centred near the 0.35 m limit so part of the disc genuinely falls outside it.
    o = parse_objective("pick[random] place[random(0.30, 0.0, 0.12, r0.12)]", pickable)
    assert o.warnings, "a region overhanging the envelope should warn"

    # Fully inside the measured envelope: no warning at all.
    clean = parse_objective("pick[random] place[random(0.20, 0.0, 0.12, r0.06)]", pickable)
    assert not clean.warnings, clean.warnings

    print("objective demo OK: grammar, colon form, subsets, boxes, reach rejection and warnings")
    print(f"  example -> {o.describe()}")
    print(f"  warning -> {o.warnings[0][:96]}...")


if __name__ == "__main__":
    demo()
