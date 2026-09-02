# SPDX-License-Identifier: BSD-3-Clause
"""Everything a new user would hit, checked in one run.

Split into two tiers because they cost very different amounts:

* **fast** -- no Isaac Sim. Module self-checks, every config parsing, the objective grammar and
  its rejection paths, the registry, the ZeroMQ round-trip, the dataset writer, and whether the
  files the docs point at actually exist. Seconds.
* **sim** -- boots Isaac Sim. Loading the robot, the cameras seeing something, a policy server
  driving the environment over a socket. Minutes.

    python scripts/run_all_tests.py            # fast tier
    python scripts/run_all_tests.py --sim      # everything

Exit code is non-zero if anything fails, so it works as a pre-push check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import textwrap
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  {PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail and not ok else ""), flush=True)
    return ok


def skip(name: str, why: str) -> None:
    results.append((name, SKIP, why))
    print(f"  {SKIP}  {name}   ({why})", flush=True)


def sh(cmd: list[str], timeout: int = 300, env: dict | None = None) -> tuple[int, str]:
    e = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "yes", "PYTHONIOENCODING": "utf-8", **(env or {})}
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=e)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"


# --------------------------------------------------------------------- fast tier


def test_self_checks() -> None:
    print("\nmodule self-checks")
    for mod in ("simbridge.schema", "simbridge.interfaces", "simbridge.objective",
                "simbridge.lerobot", "simbridge.lerobot_recorder", "simbridge.transport.test_roundtrip"):
        code, out = sh([PY, "-m", mod], timeout=180)
        record(f"python -m {mod}", code == 0, out.strip().splitlines()[-1] if out.strip() else "")

    # Run as a path: the package import already registered these factories, and -m would run the
    # file again into a registry that rejects duplicate names.
    for path in ("simbridge/scene/builtins.py", "simbridge/sources/basic.py"):
        code, out = sh([PY, path], timeout=180)
        record(f"python {path}", code == 0, out.strip().splitlines()[-1] if out.strip() else "")


def test_configs() -> None:
    print("\nevery config parses")
    src = (
        "import sys; sys.path.insert(0,'.');"
        "from simbridge.builder import load_config, describe;"
        "from simbridge.registry import register_task;"
        "[register_task(n, i) for n, i in ("
        "('pick_place','SO101-PickPlace-v0'),('pick_place_play','SO101-PickPlace-Play-v0'),"
        "('reach','SO101-Reach-v0'),('reach_play','SO101-Reach-Play-v0'))];"
        "import sys as s; c = load_config(s.argv[1]); print(describe(c))"
    )
    for cfg in sorted((REPO / "configs").rglob("*.yaml")):
        code, out = sh([PY, "-c", src, str(cfg)], timeout=120)
        record(f"configs/{cfg.relative_to(REPO / 'configs').as_posix()}", code == 0, out.strip().splitlines()[-1] if out.strip() else "")


def test_objective_grammar() -> None:
    print("\nobjective grammar and reach validation")
    src = """
import sys; sys.path.insert(0, '.')
from simbridge.objective import parse_objective, ObjectiveError
ok = [
  "pick[object] place[0.20, 0.0, 0.12]",
  "pick[random] place[random(0.20, 0.0, 0.12, r0.06)]",
  "pick:[object]place:[0.20,0.0,0.12]",
  "pick[random] place[box(0.20, 0.0, 0.12, 0.03, 0.06, 0.02)]",
]
bad = [
  "pick[random] place[random(0, 0, 0, r1)]",
  "pick[apple] place[0.20, 0.0, 0.12]",
  "pick[object] place[0.2, 0.0]",
  "pick[object] place[random(0.2, 0, 0.1, 0.06)]",
  "grab[object] drop[0,0,0]",
  "pick[object] place[0.9, 0.0, 0.12]",
]
for e in ok:
    parse_objective(e, ["object"])
for e in bad:
    try:
        parse_objective(e, ["object"]); raise AssertionError("should have been rejected: " + e)
    except ObjectiveError:
        pass
print(f"{len(ok)} accepted, {len(bad)} rejected")
"""
    code, out = sh([PY, "-c", src], timeout=120)
    record("grammar: 4 valid accepted, 6 invalid rejected", code == 0, out.strip().splitlines()[-1] if out else "")


def test_physics_selection() -> None:
    """A named backend must actually take, and an unknown one must be refused.

    Both halves matter. ``parse_env_cfg`` resolves every ``PresetCfg`` to its ``.default``, and
    ``SimulationContext`` silently collapses an unresolved one the same way -- so a config asking
    for Newton ran on PhysX and said nothing about it. ``resolve_presets`` then skips the
    validation Hydra performs, so a typo'd name falls back to default just as quietly.
    """
    print("\nphysics backend selection")
    src = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, '.')
        import so101_scene  # noqa: F401
        from simbridge.builder import load_env_cfg

        for name, expect in (("newton_mjwarp", "NewtonCfg"), ("isaacsim_physx", "PhysxCfg")):
            got = type(load_env_cfg("SO101-PickPlace-v0", "cuda:0", 2, name).sim.physics).__name__
            if got != expect:
                print(f"WRONG {name} -> {got}, expected {expect}")
                raise SystemExit(1)
        print("SELECTED")

        try:
            load_env_cfg("SO101-PickPlace-v0", "cuda:0", 2, "definitely_not_a_backend")
        except ValueError as exc:
            print("REJECTED" if "definitely_not_a_backend" in str(exc) else f"BAD MESSAGE: {exc}")
        else:
            print("NOT REJECTED")
        """
    )
    code, out = sh([PY, "-c", src], timeout=300)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    record("a named backend resolves to its own class", code == 0 and "SELECTED" in out, last)
    record("an unknown backend name is refused", "REJECTED" in out and "NOT REJECTED" not in out, last)


def test_gripper_config() -> None:
    """Selecting so101_full must repoint the task wiring, and the gripper must be configurable.

    A task hardcodes body and joint names. Naming a robot that has none of them has to move the
    ee_frame and the gripper action with it, or the scene fails on a path no config mentions.
    """
    print("\nparallel gripper wiring and config")
    src = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, '.')
        import so101_scene  # noqa: F401
        from simbridge.builder import build_env_cfg, load_config
        from simbridge.registry import register_task

        register_task("pick_place", "SO101-PickPlace-v0")
        cfg = load_config("configs/pick_place_full.yaml")

        env_cfg = build_env_cfg(cfg, device="cuda:0", num_envs=1)
        grip = env_cfg.actions.gripper_action
        if set(grip.joint_names) != {"base_gripper_left_joint", "base_gripper_right_joint"}:
            print(f"WRONG JOINTS {grip.joint_names}")
            raise SystemExit(1)
        if "gripper_base" not in env_cfg.scene.ee_frame.target_frames[0].prim_path:
            print("EE FRAME NOT REPOINTED")
            raise SystemExit(1)
        print("WIRED")

        cfg["scene"]["robot"]["gripper"]["close"] = -0.022
        half = build_env_cfg(cfg, device="cuda:0", num_envs=1).actions.gripper_action
        print("CONFIGURED" if set(half.close_command_expr.values()) == {-0.022} else "NOT CONFIGURED")

        cfg["scene"]["robot"]["gripper"]["close"] = -0.06
        try:
            build_env_cfg(cfg, device="cuda:0", num_envs=1)
        except ValueError:
            print("RANGE CHECKED")
        else:
            print("OUT OF RANGE ACCEPTED")

        cfg["scene"]["robot"]["gripper"] = {"stifness": 1}
        try:
            build_env_cfg(cfg, device="cuda:0", num_envs=1)
        except ValueError:
            print("TYPO CHECKED")
        else:
            print("TYPO ACCEPTED")
        """
    )
    code, out = sh([PY, "-c", src], timeout=300)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    record("so101_full repoints the task wiring", code == 0 and "WIRED" in out, last)
    record("gripper open/close come from the config", "CONFIGURED" in out and "NOT CONFIGURED" not in out, last)
    record("travel outside the joint range is refused", "RANGE CHECKED" in out, last)
    record("an unknown gripper key is refused", "TYPO CHECKED" in out, last)


def test_lehome() -> None:
    """The LeHome catalogue, and the two things that break when a USD prop replaces the cube.

    Both failures found here were silent-ish: a body-name regex that stops matching because the
    rigid body is a child prim, and a lift threshold that pays out at step 0 because a household
    prop is taller than a 25 mm cube.
    """
    print("\nLeHome household assets")
    src_code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, '.')
        from pathlib import Path
        from simbridge.objective import LEHOME_CATALOGUE, graspable

        paths = [p.path for p in LEHOME_CATALOGUE.values()]
        assert len(paths) == len(set(paths)), 'duplicate asset paths'
        assert all(p.width > 0 for p in LEHOME_CATALOGUE.values())
        assert all(p.path.startswith('objects/') for p in LEHOME_CATALOGUE.values())
        print('CATALOGUE', len(LEHOME_CATALOGUE))

        # The parallel gripper is what makes these props usable; the single jaw is not.
        wide = sum(1 for p in LEHOME_CATALOGUE.values() if not graspable(p.width, 0.1286)[0])
        narrow = sum(1 for p in LEHOME_CATALOGUE.values() if not graspable(p.width, 0.0362)[0])
        assert narrow > wide, (narrow, wide)
        print('GRIPPER', len(LEHOME_CATALOGUE) - wide, 'of', len(LEHOME_CATALOGUE))

        root = Path('assets/lehome')
        if not root.exists():
            print('NOASSETS')
            raise SystemExit(0)
        missing = [p.path for p in LEHOME_CATALOGUE.values() if not (root / p.path).exists()]
        print('FILES' if not missing else f'MISSING {missing[:3]}')

        import so101_scene  # noqa: F401
        from simbridge.builder import build_env_cfg, load_config, relax_object_bodies
        from simbridge.scene.builtins import lehome_usd_path
        from simbridge.registry import register_task
        register_task('pick_place', 'SO101-PickPlace-v0')

        try:
            lehome_usd_path('definitely_not_a_prop')
        except KeyError as exc:
            print('TYPO' if 'burger_patty' in str(exc) else 'TYPO NO LIST')
        else:
            print('TYPO ACCEPTED')

        cfg = load_config('configs/lehome_livingroom_cup.yaml')
        env_cfg = build_env_cfg(cfg, device='cuda:0', num_envs=1)

        # relax_object_bodies runs inside build_env_cfg; run it again on a fresh cfg to see what
        # it claims to touch, then assert the built cfg agrees.
        fresh = build_env_cfg(load_config('configs/pick_place.yaml'), device='cuda:0', num_envs=1)
        touched = relax_object_bodies(fresh, {'object'})
        print('RELAX', len(touched), touched[:2])

        from isaaclab.managers import SceneEntityCfg
        bad = [
            f'{n}:{k}'
            for n, t in vars(env_cfg.events).items()
            if isinstance(getattr(t, 'params', None), dict)
            for k, v in t.params.items()
            if isinstance(v, SceneEntityCfg) and v.name == 'object' and v.body_names not in (None, '.*')
        ]
        print('BODIES' if not bad else f'STILL PINNED {bad}')

        want = cfg['sim']['lift_height']
        got = env_cfg.rewards.lifting_object.params['minimal_height']
        print('LIFT' if abs(got - want) < 1e-9 else f'LIFT NOT APPLIED {got} != {want}')
        """
    )
    code, out = sh([PY, "-c", src_code], timeout=300)
    # Match on whole tokens: "TYPO NO LIST" also contains "TYPO", and a substring check would
    # pass on the failure it is meant to catch.
    lines = {ln.split(" ")[0] for ln in out.splitlines()}
    last = out.strip().splitlines()[-1] if out.strip() else ""
    record("catalogue is well formed", code == 0 and "CATALOGUE" in lines, last)
    record("the parallel gripper is what makes these props usable", "GRIPPER" in lines, last)
    if "NOASSETS" in lines:
        skip("[lehome] assets on disk", "run scripts/fetch_lehome.py")
        return
    record("every catalogued asset is on disk", "FILES" in lines, last)
    record("an unknown prop name is refused with the list", "TYPO" in lines, last)
    record("body-name regexes are relaxed for a USD prop", "RELAX" in lines and "BODIES" in lines, last)
    record("sim.lift_height reaches the reward term", "LIFT" in lines, last)


def test_garment() -> None:
    """The shirt: a USD mesh as Newton cloth, and the two things that silently stop working.

    The coupler's body list is written for the single-jaw robot, so selecting the parallel
    gripper used to fail the scene outright. And the shirt mesh has to stay near-uniform or
    self-collision is impossible -- that is a property of the committed asset, so check it.
    """
    print("\ngarment cloth")
    src_code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, '.')
        import numpy as np
        from pathlib import Path
        from pxr import Usd, UsdGeom

        shirt = Path('assets/garment/shirt.usd')
        assert shirt.exists(), 'assets/garment/shirt.usd is missing; run scripts/make_garment.py'
        stage = Usd.Stage.Open(str(shirt))
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert len(meshes) == 1, f'deformables need exactly one Mesh, found {len(meshes)}'
        m = UsdGeom.Mesh(meshes[0])
        pts = np.array(m.GetPointsAttr().Get()); f = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        e = np.concatenate([pts[f[:,0]]-pts[f[:,1]], pts[f[:,1]]-pts[f[:,2]], pts[f[:,2]]-pts[f[:,0]]])
        L = np.linalg.norm(e, axis=1)
        # Quadric decimation gave 170x here and self-collision was impossible. Voxel clustering
        # gives ~9x. Anything above 20x means the asset was regenerated the wrong way.
        ratio = L.max() / L.min()
        print('MESH', len(pts), f'{ratio:.0f}x')
        assert ratio < 20, f'edge-length spread {ratio:.0f}x is too irregular for self-collision'

        import so101_scene  # noqa: F401
        from simbridge.builder import build_env_cfg, load_config
        from simbridge.registry import register_task
        register_task('pick_place', 'SO101-PickPlace-v0')

        cfg = build_env_cfg(load_config('configs/lehome_bedroom_shirt.yaml'), device='cuda:0', num_envs=1)
        spawn = cfg.scene.shirt.spawn
        print('SPAWN' if type(spawn).__name__ == 'UsdFileCfg' else f'WRONG SPAWN {type(spawn).__name__}')

        proxies = cfg.sim.physics.solver_cfg.proxies
        bodies = proxies[0].bodies
        ok = any('gripper_base' in b for b in bodies) and not any(b.endswith('/gripper') for b in bodies)
        print('PROXY' if ok else f'PROXY NOT REPOINTED {bodies}')

        vbd = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
        # The radius has to stay under the mesh's tightest edge at scale, or every particle
        # collides with its own neighbour and the shirt tears itself apart in 60 steps.
        tight = float(L.min()) * cfg.scene.shirt.spawn.scale[0]
        print('SELFCONTACT' if vbd.particle_enable_self_contact and vbd.particle_self_contact_radius < tight
              else f'SELFCONTACT BAD {vbd.particle_self_contact_radius} vs {tight}')

        from simbridge.registry import OBJECTS
        try:
            OBJECTS['light']({'kind': 'floodlight'})
        except KeyError:
            print('LIGHT')
        else:
            print('LIGHT KIND ACCEPTED')
        """
    )
    code, out = sh([PY, "-c", src_code], timeout=300)
    lines = {ln.split(" ")[0] for ln in out.splitlines()}
    last = out.strip().splitlines()[-1] if out.strip() else ""
    record("the shipped shirt is one near-uniform mesh", code == 0 and "MESH" in lines, last)
    record("a USD mesh spawns as cloth", "SPAWN" in lines, last)
    record("the cloth coupler follows the chosen robot", "PROXY" in lines, last)
    record("self-contact radius is under the mesh spacing", "SELFCONTACT" in lines, last)
    record("an unknown light kind is refused", "LIGHT" in lines, last)


def test_registry() -> None:
    print("\nregistry")
    src = (
        "import sys; sys.path.insert(0,'.');"
        "import simbridge.sources, simbridge.scene;"
        "from simbridge.registry import ROBOTS, OBJECTS, CAMERAS, SOURCES;"
        "assert 'so101' in ROBOTS, ROBOTS;"
        "assert {'cuboid','static_cuboid','usd','ycb','cloth','soft_body','lehome','light'} <= set(OBJECTS), OBJECTS;"
        "assert 'tiled' in CAMERAS, CAMERAS;"
        "assert {'zero','random','rl_checkpoint','zmq','keyboard'} <= set(SOURCES), SOURCES;"
        "print('robots', len(ROBOTS), 'objects', len(OBJECTS), 'cameras', len(CAMERAS), 'sources', len(SOURCES))"
    )
    code, out = sh([PY, "-c", src], timeout=120)
    record("all documented names are registered", code == 0, out.strip().splitlines()[-1] if out else "")


def test_docs_links() -> None:
    print("\ndocs point at files that exist")
    import re

    missing: list[str] = []
    for md in list(REPO.glob("*.md")) + list((REPO / "docs").glob("*.md")) + list((REPO / "experiment").glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        for target in re.findall(r"\]\(([^)#:]+?)\)", text):
            if target.startswith(("http", "mailto")):
                continue
            p = (md.parent / target).resolve()
            if not p.exists():
                missing.append(f"{md.name} -> {target}")
    record("no broken relative links", not missing, "; ".join(missing[:4]))

    print("\nscripts referenced by the docs exist")
    referenced = set()
    for md in list(REPO.glob("*.md")) + list((REPO / "docs").glob("*.md")) + list((REPO / "experiment").glob("*.md")):
        referenced |= set(re.findall(r"(?:python\s+)((?:experiment/)?scripts/[\w./]+\.py)", md.read_text(encoding="utf-8", errors="ignore")))
    absent = [s for s in referenced if not (REPO / s).exists()]
    record(f"{len(referenced)} referenced scripts present", not absent, "; ".join(absent[:4]))


def test_scripts_help() -> None:
    """--help must work without Isaac Sim for the scripts that do not need it."""
    print("\nCLI help for simulator-free scripts")
    for script in ("scripts/policy_server.py", "scripts/teleop_server.py",
                   "scripts/lerobot_server.py", "scripts/make_tutorial.py",
                   "scripts/run_all_tests.py"):
        code, out = sh([PY, str(REPO / script), "--help"], timeout=120)
        record(f"{script} --help", code == 0, out.strip().splitlines()[-1] if out.strip() else "")


def test_zmq_roundtrip() -> None:
    print("\nZeroMQ policy server, cross-process")
    server = subprocess.Popen(
        [PY, str(REPO / "scripts/policy_server.py"), "--policy", "sine",
         "--action-dim", "6", "--endpoint", "tcp://127.0.0.1:5613"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(3)
    src = """
import sys; sys.path.insert(0, '.')
import numpy as np
from simbridge.interfaces import RemoteActionSource
from simbridge.schema import ObsPacket
from simbridge.transport import ZmqClientTransport
src = RemoteActionSource(ZmqClientTransport('tcp://127.0.0.1:5613', timeout_ms=5000), action_horizon=1)
obs = ObsPacket(step=1, num_envs=4,
                state={'joint_pos': np.zeros((4,6), np.float32)},
                images={'scene_cam': np.zeros((4,64,64,3), np.uint8)})
a = src.advance(obs)
assert a.shape == (4, 6), a.shape
print('round-trip shape', a.shape)
"""
    try:
        code, out = sh([PY, "-c", src], timeout=120)
        record("action returned over a real socket", code == 0, out.strip().splitlines()[-1] if out else "")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def test_lerobot_bridge() -> None:
    print("\nLeRobot bridge (no lerobot install needed)")
    src = """
import sys; sys.path.insert(0, '.')
import numpy as np
from simbridge.lerobot import lerobot_action_to_array, array_to_lerobot_action, JOINTS
arr = np.array([0.2, -0.3, 0.4, -0.1, 0.05, 0.0], dtype=np.float32)
back = lerobot_action_to_array(array_to_lerobot_action(arr), binary_gripper=False)
assert np.allclose(arr, back, atol=1e-3), (arr, back)
assert JOINTS[0] == 'shoulder_pan' and JOINTS[-1] == 'gripper'
assert lerobot_action_to_array({'gripper.pos': 100.0})[5] == 1.0
print('units round-trip, joint order matches SOFollower')
"""
    code, out = sh([PY, "-c", src], timeout=120)
    record("LeRobot unit conversion", code == 0, out.strip().splitlines()[-1] if out else "")

    code, out = sh([PY, str(REPO / "scripts/lerobot_server.py"), "policy", "--path", "nope"], timeout=120)
    helpful = "lerobot" in out.lower() and ("venv" in out.lower() or "import" in out.lower())
    record("missing-LeRobot error explains the venv setup", helpful, out.strip().splitlines()[-1] if out else "")


# ---------------------------------------------------------------------- sim tier


def test_sim_scene() -> None:
    print("\n[sim] scene loads and the robot is what we think it is")
    code, out = sh([PY, str(REPO / "scripts/scene_demo.py"), "--steps", "60", "--num_envs", "2"], timeout=900)
    ok = code == 0 and "shoulder_pan" in out and "moving_jaw_so101_v1" in out
    record("scene_demo: robot spawns, names as documented", ok)


def test_sim_cameras() -> None:
    print("\n[sim] cameras render something")
    code, out = sh([PY, str(REPO / "scripts/dump_camera_views.py"),
                    "--config", "configs/pick_place.yaml", "--num_envs", "2"], timeout=900)
    ok = code == 0 and "scene_cam" in out
    blank = "nearly constant" in out
    record("dump_camera_views: frames written", ok)
    record("no camera is blank", ok and not blank, "a camera saw nothing" if blank else "")


def test_sim_run_zmq() -> None:
    print("\n[sim] a policy server drives the environment")
    server = subprocess.Popen(
        [PY, str(REPO / "scripts/policy_server.py"), "--policy", "sine",
         "--action-dim", "6", "--endpoint", "tcp://127.0.0.1:5614"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(3)
    try:
        code, out = sh([PY, str(REPO / "scripts/run.py"),
                        "--config", "configs/pick_place_teleop.yaml",
                        "--steps", "60",
                        "--set", "control.endpoint=tcp://127.0.0.1:5614"], timeout=900)
        record("run.py driven over ZMQ", code == 0 and "completed" in out)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole test suite.")
    ap.add_argument("--sim", action="store_true", help="also run the tests that boot Isaac Sim")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 74)
    print("so101-scene test suite" + ("  (fast + sim)" if args.sim else "  (fast tier)"))
    print("=" * 74)

    test_self_checks()
    test_configs()
    test_objective_grammar()
    test_physics_selection()
    test_gripper_config()
    test_lehome()
    test_garment()
    test_registry()
    test_scripts_help()
    test_zmq_roundtrip()
    test_lerobot_bridge()
    test_docs_links()

    if args.sim:
        test_sim_scene()
        test_sim_cameras()
        test_sim_run_zmq()
    else:
        skip("[sim] scene, cameras, ZMQ-driven run", "pass --sim to include")

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)

    print("\n" + "=" * 74)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped   ({time.time() - t0:.0f}s)")
    print("=" * 74)
    if n_fail:
        print("\nfailures:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}" + (f"\n      {detail}" if detail else ""))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
