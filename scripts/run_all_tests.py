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
    code, out = sh([PY, "simbridge/scene/builtins.py"], timeout=180)
    record("python simbridge/scene/builtins.py", code == 0,
           out.strip().splitlines()[-1] if out.strip() else "")


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


def test_registry() -> None:
    print("\nregistry")
    src = (
        "import sys; sys.path.insert(0,'.');"
        "import simbridge.sources, simbridge.scene;"
        "from simbridge.registry import ROBOTS, OBJECTS, CAMERAS, SOURCES;"
        "assert 'so101' in ROBOTS, ROBOTS;"
        "assert {'cuboid','static_cuboid','usd'} <= set(OBJECTS), OBJECTS;"
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
