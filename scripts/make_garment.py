# SPDX-License-Identifier: BSD-3-Clause
"""Remesh LeHome's shirt into something a cloth solver can actually run.

LeHome's garment (`Thin-Shells/garment/.../TCLC_002_obj.usd`) is 14,746 vertices and 28,726
triangles. Loaded straight into Newton's VBD solver it diverges at step 7: worn at a size this
arm can reach, its triangles are ~2.4 mm across against a 2 mm particle radius, so the particles
start out overlapping. Isaac Lab's own cloth demo uses an 8x8 grid -- 81 particles.

So the mesh has to come down. **How** it comes down turned out to matter more than how far:

    quadric decimation (fast_simplification)   edges 0.21 .. 35.5 mm   ratio 170x
    voxel clustering   (open3d)                edges 1.04 .. 11.07 mm  ratio  11x

Both give ~3000 vertices. Quadric decimation optimises for silhouette, which is the right goal
for rendering and the wrong one here: it leaves slivers two orders of magnitude below the mean,
and a self-contact radius has to sit under the *smallest* edge or every particle collides with
its own neighbours. At 0.21 mm there is no usable radius. Uniform triangles are not a nicety for
a cloth solver, they are what makes self-collision possible at all.

Hence voxel clustering, and hence the knob being **particle spacing**, not vertex count.

    python scripts/make_garment.py                  # the shipped shirt, 6 mm spacing
    python scripts/make_garment.py --spacing 4      # finer, slower, folds more sharply
    python scripts/make_garment.py --sweep          # write 4 / 6 / 9 mm and compare

Output goes to assets/garment/, which IS committed -- unlike assets/lehome/ -- so the shirt
config works without a 1.7 GB download. Source mesh by LeHome, CC-BY-4.0; see docs/LEHOME.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = (
    REPO / "assets/lehome/objects/Thin-Shells/garment/Tops/Collar_Lsleeve_FrontClose/TCLC_002"
    / "TCLC_002_obj.usd"
)
OUT = REPO / "assets" / "garment" / "shirt.usd"

# The `scale:` the shirt config uses. Only affects the reported millimetres -- the USD is written
# at full size, and spacing is quoted at the size the robot actually meets.
SCALE = 0.25


def load(path: Path):
    """Points in METRES and triangle indices, from a USD authored in whatever unit it likes."""
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SystemExit(f"cannot open {path}")
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise SystemExit(f"expected exactly one Mesh in {path.name}, found {len(meshes)}")
    mesh = UsdGeom.Mesh(meshes[0])
    unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not (counts == 3).all():
        raise SystemExit("mesh is not triangulated; this script does not fan-triangulate")
    # np.array, not asarray: an array wrapping a Vt buffer is READ-ONLY, and both open3d and
    # fast_simplification reject that with messages that look like dtype problems.
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) * unit
    faces = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    return pts, faces


def remesh(pts, faces, voxel: float):
    """Near-uniform triangles at roughly ``voxel`` spacing, by open3d vertex clustering."""
    import numpy as np
    import open3d as o3d

    src = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(pts), o3d.utility.Vector3iVector(faces),
    )
    out = src.simplify_vertex_clustering(voxel, o3d.geometry.SimplificationContraction.Average)
    # Clustering merges vertices, which leaves zero-area triangles and orphan points behind. VBD
    # builds one particle per vertex and one spring per edge, so both would be silent nonsense.
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_duplicated_vertices()
    out.remove_unreferenced_vertices()
    return np.asarray(out.vertices), np.asarray(out.triangles)


def edges_mm(pts, faces, scale: float):
    """Edge lengths in millimetres at the scale the robot sees: (min, median, max)."""
    import numpy as np

    e = np.concatenate([
        pts[faces[:, 0]] - pts[faces[:, 1]],
        pts[faces[:, 1]] - pts[faces[:, 2]],
        pts[faces[:, 2]] - pts[faces[:, 0]],
    ])
    lengths = np.linalg.norm(e, axis=1) * scale * 1000
    return tuple(np.percentile(lengths, [0, 50, 100]))


def write(path: Path, pts, faces) -> None:
    """A minimal metres-authored USD with one Mesh at /World/mesh.

    Exactly one mesh, because Isaac Lab's ``define_deformable_body_properties`` traverses for a
    single Mesh under the prim and raises on more.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, Vt

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()                                  # CreateNew refuses to overwrite
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype("float32")))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype="int32")))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype("int32").ravel()))
    # Otherwise Kit subdivides it for display and the render stops matching what is simulated.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacing", type=float, default=6.0,
                    help="target particle spacing in mm, at the config's scale (default 6)")
    ap.add_argument("--sweep", action="store_true", help="write 4 / 6 / 9 mm and compare")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--scale", type=float, default=SCALE)
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} is missing. Run: python scripts/fetch_lehome.py")

    pts, faces = load(args.source)
    lo, med, hi = edges_mm(pts, faces, args.scale)
    print(f"source {args.source.name}: {len(pts)} verts, {len(faces)} tris, "
          f"edges at scale {args.scale}: {lo:.2f} / {med:.2f} / {hi:.2f} mm (min/median/max)")

    jobs = ([(s, args.out.with_name(f"shirt_{s:g}mm.usd")) for s in (4.0, 6.0, 9.0)]
            if args.sweep else [(args.spacing, args.out)])

    print(f"\n{'file':<16} {'verts':>7} {'tris':>7} {'min':>8} {'median':>8} {'max':>8}   asked")
    for spacing, out in jobs:
        p, f = remesh(pts, faces, spacing / 1000.0 / args.scale)
        write(out, p, f)
        lo, med, hi = edges_mm(p, f, args.scale)
        print(f"{out.name:<16} {len(p):>7} {len(f):>7} {lo:>6.2f}mm {med:>6.2f}mm {hi:>6.2f}mm"
              f"   {spacing:g} mm")

    print("\nIn the config, particle_radius is about a quarter of the median. The solver's"
          "\nparticle_self_contact_radius must stay under the MIN -- so101_scene/pick_place_env_cfg.py.")


if __name__ == "__main__":
    main()
