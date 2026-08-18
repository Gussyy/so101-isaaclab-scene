# SPDX-License-Identifier: BSD-3-Clause
"""Measure every named prop and say which ones this arm can actually pick up.

The SO-101 has a single jaw. scripts/measure_workspace.py put its jaw bodies 36.2 mm apart, so
the shortest side of an object's bounding box decides whether it can be pinched at all -- a
0.5 kg sugar box and a 0.1 m bowl fail for different reasons, and neither failure is visible in
a config that names them. This turns "which of these can I use?" into a table, measured by
opening each asset rather than by copying the YCB spec sheet.

    python scripts/measure_objects.py --out docs/objects.txt

Assets stream from the Isaac asset server, so the first run needs a network connection and takes
a few minutes. Later runs hit the local cache.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the named props.")
parser.add_argument("--out", default="docs/objects.txt")
parser.add_argument("--jaw", type=float, default=None,
                    help="jaw opening in metres; defaults to simbridge.objective.JAW_WIDTH")
parser.add_argument("--only", default=None, help="measure one prop by name")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from simbridge.objective import JAW_WIDTH, PROP_CATALOGUE, graspable  # noqa: E402
from simbridge.scene.builtins import ycb_usd_path  # noqa: E402

# The task's own cube, for scale: 25 mm, 15 g, and the policy that lifts it is 92% successful.
REFERENCE_MASS = 0.015


def measure(name: str) -> dict | None:
    """Open one asset and read its extents and authored mass."""
    try:
        stage = Usd.Stage.Open(ycb_usd_path(name))
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<20} could not open: {exc}")
        return None
    if stage is None:
        print(f"  {name:<20} could not open")
        return None

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    if rng.IsEmpty():
        print(f"  {name:<20} empty bounding box")
        return None
    size = rng.GetSize()

    # USD stages carry their own linear unit; YCB assets are authored in centimetres.
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    dims = sorted(float(c) * scale for c in (size[0], size[1], size[2]))

    mass = None
    for prim in stage.Traverse():
        api = UsdPhysics.MassAPI(prim)
        if api:
            attr = api.GetMassAttr()
            if attr and attr.HasAuthoredValue():
                mass = float(attr.Get())
                break

    return {"name": name, "min": dims[0], "mid": dims[1], "max": dims[2], "mass": mass}


def verdict(m: dict, jaw: float) -> tuple[str, str]:
    """Graspable, marginal or not, and why. Width first -- it is the hard constraint."""
    ok, why = graspable(m["min"], jaw)
    if not ok:
        return "no", why
    if "tight" in why:
        return "tight", why
    if m["mass"] is not None and m["mass"] > 10 * REFERENCE_MASS:
        return "heavy", f"{m['mass'] * 1000:.0f} g against a 15 g reference cube"
    return "yes", why


def main() -> None:
    names = [args_cli.only] if args_cli.only else sorted(PROP_CATALOGUE)
    rows = [r for r in (measure(n) for n in names) if r]
    rows.sort(key=lambda r: r["min"])

    jaw = args_cli.jaw if args_cli.jaw is not None else JAW_WIDTH
    lines = [
        "Named props, measured by opening each asset (scripts/measure_objects.py).",
        "",
        f"SO-101 jaw opening: {jaw * 1000:.1f} mm.  The shortest side has to fit between the jaws.",
        "Reference: the task's own cube is 25 mm and 15 g, and is picked at 92% success.",
        "",
        "CAVEAT on that 36.2 mm. measure_workspace.py reported it as the separation between the",
        "two jaw *body origins*, and reported no measurable travel between the closed and open",
        "extremes -- the revolute jaw sweeps its tips while the body origins stay put. So it is a",
        "reasonable scale for the gripper, not a measured maximum opening. Treat the boundary as",
        "soft: 'tight' may well work, and something a few mm over may too. What the column does",
        "rule out confidently is the 100 mm end of the table.",
        "",
        "Mass is blank because these assets do not author UsdPhysics mass; PhysX derives it from",
        "density at spawn. The width column is the one doing the work here.",
        "",
        f"{'prop':<20} {'shortest':>9} {'longest':>9} {'mass':>8}   pickable",
        "-" * 78,
    ]
    for r in rows:
        ok, why = verdict(r, jaw)
        mass = f"{r['mass'] * 1000:.0f} g" if r["mass"] is not None else "-"
        lines.append(
            f"{r['name']:<20} {r['min'] * 1000:>7.1f}mm {r['max'] * 1000:>7.1f}mm {mass:>8}   {ok:<6} {why}"
        )
    counts = {}
    for r in rows:
        counts[verdict(r, jaw)[0]] = counts.get(verdict(r, jaw)[0], 0) + 1
    lines += ["-" * 78, "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))]

    # The catalogue is transcribed measurements, so it can drift from what the assets say -- the
    # first version of it had banana at 39.4 mm against a real 38.6 mm. Report that here rather
    # than letting a stale number decide whether a config is rejected.
    drift = [
        f"  {r['name']}: measured {r['min'] * 1000:.1f} mm, catalogue says "
        f"{PROP_CATALOGUE[r['name']][1] * 1000:.1f} mm"
        for r in rows
        if r["name"] in PROP_CATALOGUE and abs(r["min"] - PROP_CATALOGUE[r["name"]][1]) > 5e-4
    ]
    if drift:
        lines += ["", "DRIFT from simbridge.objective.PROP_CATALOGUE -- update it:"] + drift
    elif rows:
        lines += ["", f"matches simbridge.objective.PROP_CATALOGUE ({len(rows)} props)"]

    text = "\n".join(lines)
    print("\n" + text + "\n")
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
