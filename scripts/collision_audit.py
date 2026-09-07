"""What PhysX collides with on the SO-101: every collider, its approximation, and how fat its hull is.

    python scripts/collision_audit.py              # the asset, from USD alone (no Kit): 18 colliders
    python scripts/collision_audit.py --cooked     # boots the VR scene, reads back what PhysX cooked
                                                   # (hulls, vertices per collider) and draws them

The first mode is the check: which parts carry a collider, whether each is a convex hull or a
decomposition, how many source triangles it has, and the hull-to-mesh volume ratio -- a convex
hull of a U-shaped bracket is a block several times the part (docs/VR.md, "The robot's
colliders, checked"). The second reads the cooked hulls through omni.physx's
request_convex_collision_representation and writes docs/vr_cooked_hulls.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSET = REPO / "robot_description/IsaacAssets/SO-ARM101-FULL/SO-ARM101-FULL.usda"


def audit(asset: Path = ASSET) -> list[dict]:
    """Every collider on the asset: body, name, approximation, triangles, hull/mesh volume."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    stage = Usd.Stage.Open(str(asset))

    def body_of(prim):
        p = prim
        while p and p.GetPath() != Sdf.Path.absoluteRootPath:
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return p.GetName()
            p = p.GetParent()
        return "-"

    rows = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        row = {"body": body_of(prim), "collider": prim.GetName(), "type": prim.GetTypeName(), "approximation": "-",
               "tris": 0, "hull_over_mesh": None, "bbox_mm": None,
               "contact_offset": prim.GetAttribute("physxCollision:contactOffset").Get(),
               "rest_offset": prim.GetAttribute("physxCollision:restOffset").Get()}
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            row["approximation"] = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pts = np.array([list(xf.Transform(Gf.Vec3d(*p))) for p in m.GetPointsAttr().Get()])
            fvc = np.array(m.GetFaceVertexCountsAttr().Get())
            fvi = np.array(m.GetFaceVertexIndicesAttr().Get())
            tris, k = [], 0
            for n in fvc:
                idx = fvi[k:k + n]
                k += n
                tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, n - 1)]
            t = np.array(tris)
            a, b, c = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
            vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
            row["tris"] = len(t)
            row["hull_over_mesh"] = ConvexHull(pts).volume / max(vol, 1e-12)
            row["bbox_mm"] = tuple(((pts.max(0) - pts.min(0)) * 1000).round().astype(int))
        rows.append(row)
    return sorted(rows, key=lambda r: (r["body"], r["collider"]))


def print_audit(rows: list[dict]) -> None:
    print(f"{'body':<16}{'collider':<34}{'approximation':<21}{'tris':>7}{'hull/mesh':>10}  bbox mm   contact/rest offset")
    for r in rows:
        ratio = f"{r['hull_over_mesh']:.2f}" if r["hull_over_mesh"] else "-"
        bb = " ".join(str(v) for v in r["bbox_mm"]) if r["bbox_mm"] else "-"
        print(f"{r['body']:<16}{r['collider']:<34}{str(r['approximation']):<21}{r['tris']:>7}{ratio:>10}  "
              f"{bb:<10} {r['contact_offset']}/{r['rest_offset']}")
    print(f"{len(rows)} colliders, {sum(r['tris'] for r in rows)} source triangles; "
          f"{sum(1 for r in rows if r['approximation'] == 'convexHull')} convexHull, "
          f"{sum(1 for r in rows if r['approximation'] == 'convexDecomposition')} convexDecomposition")


def cooked(config: str, out: Path) -> None:
    """Boot the scene and read back what PhysX cooked for each collider; draw the right arm's hulls."""
    ap = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    a = ap.parse_args([])
    a.headless = True
    app = AppLauncher(a).app
    import json
    import gymnasium as gym
    import numpy as np
    import torch
    import so101_scene, simbridge.sources  # noqa: F401
    from simbridge.builder import build_env_cfg, load_config, resolve_task
    from simbridge.registry import register_task
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics
    import omni.usd
    from omni.physx import get_physx_cooking_interface

    register_task("pick_place", "SO101-PickPlace-v0")
    raw = load_config(config)
    raw["control"]["source"] = "zero"
    raw["scene"].pop("cameras", None)
    env = gym.make(resolve_task(raw), cfg=build_env_cfg(raw, device=raw["sim"].get("device", "cuda:0"), num_envs=1)).unwrapped
    env.reset()
    act = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    for _ in range(30):
        env.step(act)
    stage = omni.usd.get_context().get_stage()
    stage_id = omni.usd.get_context().get_stage_id()
    cook = get_physx_cooking_interface()
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    hulls = {}
    print(f"{'collider':<36}{'approximation':<21}{'hulls':>6}{'vertices':>9}{'polygons':>9}")
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        got = []

        def on_result(result, data, _g=got):
            _g.extend((np.array([list(p) for p in h.vertices]), len(h.polygons)) for h in data)

        cook.request_convex_collision_representation(
            stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
            run_asynchronously=False, on_result=on_result)
        m = xf.GetLocalToWorldTransform(prim)
        hulls[prim.GetName()] = [np.array([list(m.Transform(Gf.Vec3d(*p))) for p in v]) for v, _ in got]
        approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else "-"
        print(f"{prim.GetName():<36}{str(approx):<21}{len(got):>6}{sum(len(v) for v, _ in got):>9}{sum(n for _, n in got):>9}")
    print(f"cooked: {sum(len(v) for v in hulls.values())} hulls, {sum(len(x) for v in hulls.values() for x in v)} vertices (one arm)")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: [x.tolist() for x in v] for k, v in hulls.items()}, open(out.with_suffix(".json"), "w"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    origin = env.scene.env_origins[0].cpu().numpy()

    def draw(ax, elev, azim, lim, title):
        cmap = plt.get_cmap("tab20")
        for i, (name, vs) in enumerate(hulls.items()):
            for v in vs:
                v = v - origin
                if lim and not ((v[:, 0] > lim[0][0]) & (v[:, 0] < lim[0][1]) & (v[:, 1] > lim[1][0]) & (v[:, 1] < lim[1][1])).any():
                    continue
                try:
                    h = ConvexHull(v)
                    ax.add_collection3d(Poly3DCollection([v[s] for s in h.simplices], facecolor=cmap(i % 20), edgecolor="k", linewidths=0.2, alpha=0.45))
                except Exception:  # noqa: BLE001  -- a degenerate hull
                    pass
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        lim = lim or ((-0.15, 0.45), (-0.55, 0.05), (0, 0.45))
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
        ax.set_box_aspect((1, 1, 0.75))

    fig = plt.figure(figsize=(18, 6))
    draw(fig.add_subplot(131, projection="3d"), 15, -60, None, "right arm: cooked collision hulls (side)")
    draw(fig.add_subplot(132, projection="3d"), 89, -90, None, "top")
    pads = [x for k, v in hulls.items() if k.startswith("arm_") for x in v]
    if pads:
        c = (np.concatenate(pads) - origin).mean(0)
        r = 0.12
        draw(fig.add_subplot(133, projection="3d"), 10, -30,
             ((c[0] - r, c[0] + r), (c[1] - r, c[1] + r), (max(0, c[2] - r), c[2] + r)), "gripper close-up")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    env.close()
    app.close()


def demo() -> None:
    rows = audit()
    assert len(rows) == 18, len(rows)
    by = {r["collider"]: r for r in rows}
    assert by["arm_r_visual"]["approximation"] == "convexDecomposition"
    assert by["base_visual"]["approximation"] == "convexDecomposition", "the gripper housing is decomposed too"
    assert by["base_visual"]["hull_over_mesh"] > 3, "the housing hull is a block several times the part"
    assert all(r["tris"] > 0 for r in rows), "every collider is a triangle mesh"
    print("collision_audit demo OK: 18 colliders, fingers and housing decomposed, hull ratios computed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
        sys.exit(0)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cooked", action="store_true", help="boot the VR scene and read back the cooked hulls")
    p.add_argument("--config", default=str(REPO / "configs/vr_teleop.yaml"))
    p.add_argument("--out", default=str(REPO / "docs/vr_cooked_hulls.png"))
    args, _ = p.parse_known_args()
    if args.cooked:
        sys.path.insert(0, str(REPO))
        cooked(args.config, Path(args.out))
    else:
        print_audit(audit())
