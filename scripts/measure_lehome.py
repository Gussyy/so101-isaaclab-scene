# SPDX-License-Identifier: BSD-3-Clause
"""Measure every LeHome asset and say which ones this arm can pick up.

Same job as scripts/measure_objects.py does for the YCB props, for the household library
downloaded by scripts/fetch_lehome.py. It also answers the question that decides how the assets
have to be spawned: **does the USD already carry physics?** LeHome's own configs point a
``RigidObjectCfg`` straight at these files with no ``rigid_props``, which only works if the
asset authors ``UsdPhysics.RigidBodyAPI`` itself. The YCB props do not, and needed
``_spawn_prop`` to define the schemas after spawning. This prints which camp each asset is in.

No Kit app: USD opens standalone, so this is seconds rather than minutes.

    python scripts/measure_lehome.py --out docs/lehome_objects.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "lehome"

# The parallel gripper measured 128.6 mm open at the finger origins (docs/joint_range.txt).
# The single jaw is 36.2 mm; pass --jaw 0.0362 to see the table for that robot instead.
DEFAULT_JAW = 0.1286


def measure(path: Path) -> dict | None:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    size = rng.GetSize()
    dims = sorted(float(c) * scale for c in (size[0], size[1], size[2]))

    rigid = collide = mass = deform = False
    mass_kg = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = True
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collide = True
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = True
            attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            if attr and attr.HasAuthoredValue() and mass_kg is None:
                mass_kg = float(attr.Get())
        # PhysX particle cloth / deformable bodies. LeHome's garments and towels are these, and
        # they are exactly what does NOT carry over to Newton -- see docs/LEHOME.md.
        if any(s.startswith("Physx") and ("Particle" in s or "Deformable" in s)
               for s in prim.GetAppliedSchemas()):
            deform = True

    return {
        "path": path.relative_to(ASSETS).as_posix(),
        "min": dims[0], "mid": dims[1], "max": dims[2],
        "units": scale, "rigid": rigid, "collide": collide, "mass": mass,
        "mass_kg": mass_kg, "deform": deform,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/lehome_objects.txt")
    ap.add_argument("--jaw", type=float, default=DEFAULT_JAW)
    ap.add_argument("--catalogue-only", action="store_true", help="only the named entries")
    args = ap.parse_args()

    if not ASSETS.exists():
        raise SystemExit(f"no assets at {ASSETS}. Run: python scripts/fetch_lehome.py")

    from simbridge.objective import LEHOME_CATALOGUE, graspable

    named = {spec[0]: name for name, spec in LEHOME_CATALOGUE.items()}
    if args.catalogue_only:
        files = [ASSETS / rel for rel in named]
    else:
        files = sorted(p for p in (ASSETS / "objects").rglob("*.usd*") if p.suffix in (".usd", ".usda", ".usdc"))

    rows = []
    for f in files:
        try:
            m = measure(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not open {f.name}: {exc}")
            continue
        if m:
            m["name"] = named.get(m["path"], "")
            rows.append(m)
    rows.sort(key=lambda r: r["min"])

    lines = [
        "LeHome household assets, measured by opening each USD (scripts/measure_lehome.py).",
        "",
        f"Gripper opening assumed: {args.jaw * 1000:.1f} mm (SO-ARM101-FULL parallel gripper, at the",
        "finger origins). Pass --jaw 0.0362 for the single-jaw so101.",
        "",
        "The `phys` column decides how the asset has to be spawned:",
        "  R = UsdPhysics.RigidBodyAPI authored   C = CollisionAPI   M = MassAPI",
        "  D = PhysX particle or deformable schemas authored in the file.",
        "An asset with no R spawns as scenery unless something defines the schemas for it, which",
        "is what simbridge's `lehome` object factory does.",
        "",
        "Measured result: NOTHING in this library carries a D. The `_Def` variants LeHome loads",
        "as DeformableObjectCfg are plain meshes with a mass and no colliders -- the deformable",
        "schema is applied at spawn time by LeHome's PhysX config, not stored in the USD. Under",
        "Newton those files therefore spawn as rigid visual meshes. See docs/LEHOME.md.",
        "",
        f"{'name':<18} {'shortest':>9} {'longest':>9} {'mass':>8} {'phys':>6}  path",
        "-" * 112,
    ]
    for r in rows:
        flags = "".join(c for c, on in (("R", r["rigid"]), ("C", r["collide"]), ("M", r["mass"]), ("D", r["deform"])) if on) or "-"
        fit = ""
        if r["name"]:
            ok, why = graspable(r["min"], args.jaw)
            fit = "  <- " + ("pick: " if ok else "TOO WIDE: ") + why
        mass = f"{r['mass_kg'] * 1000:.0f} g" if r["mass_kg"] is not None else "-"
        lines.append(
            f"{r['name']:<18} {r['min'] * 1000:>7.1f}mm {r['max'] * 1000:>7.1f}mm "
            f"{mass:>8} {flags:>6}  {r['path']}{fit}"
        )

    lines += ["-" * 112, f"{len(rows)} assets, {sum(1 for r in rows if r['rigid'])} with rigid bodies, "
              f"{sum(1 for r in rows if r['deform'])} with PhysX particle/deformable schemas"]

    # Two things in the catalogue are transcribed measurements and can therefore go stale. The
    # width decides whether a config is warned about; the `physics` flag decides whether the
    # spawner defines schemas or trusts the asset's own. A wrong flag is the quiet kind of bug
    # this repo keeps meeting, so check it here rather than at 3 am in a simulation.
    drift = [
        f"  {r['name']}: width measured {r['min'] * 1000:.1f} mm, catalogue says "
        f"{LEHOME_CATALOGUE[r['name']].width * 1000:.1f} mm"
        for r in rows
        if r["name"] and abs(r["min"] - LEHOME_CATALOGUE[r["name"]].width) > 5e-4
    ] + [
        f"  {r['name']}: physics={r['rigid']} in the file, catalogue says "
        f"{LEHOME_CATALOGUE[r['name']].physics}"
        for r in rows
        if r["name"] and r["rigid"] != LEHOME_CATALOGUE[r["name"]].physics
    ]
    if drift:
        lines += ["", "DRIFT from simbridge.objective.LEHOME_CATALOGUE -- update it:"] + drift
    else:
        lines += ["", f"matches simbridge.objective.LEHOME_CATALOGUE "
                      f"({sum(1 for r in rows if r['name'])} named props, width and physics flag)"]

    text = "\n".join(lines)
    print("\n" + text + "\n")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
