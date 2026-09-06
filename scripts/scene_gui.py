# SPDX-License-Identifier: BSD-3-Clause
"""Set the scene up in a window, press Start, and the simulator launches with exactly that.

    python scripts/scene_gui.py                      # blank scene
    python scripts/scene_gui.py configs/vr_teleop.yaml   # start from an existing config

Every control here is one key of the YAML the builder already takes -- there is no second
schema. What the window writes is a config file (``configs/gui_scene.yaml``, gitignored) and
what Start runs is ``scripts/run.py --config`` on it, in its own console so Isaac Sim's output
is visible. Load / Save round-trip any config in ``configs/``.

tkinter, because it is in the standard library, opens before Isaac Sim exists, and this is a
form with a button. The dropdowns are fed from the registry, so a robot or object added to
``simbridge.scene`` shows up here without touching this file.

Robot rotation is filled in with the robot: the single-jaw ``so101`` needs a +90 degree base
yaw and the parallel-gripper ``so101_full`` needs none. That sign was got wrong twice by
reasoning about it (configs/pick_place_full.yaml); a form should not ask a person to know it.
"""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import simbridge.scene  # noqa: E402,F401  (registers robots, objects, cameras)
import simbridge.sources  # noqa: E402,F401
from simbridge.registry import OBJECTS, ROBOTS, SOURCES  # noqa: E402

OUT = REPO / "configs" / "gui_scene.yaml"
PY = sys.executable

# Measured facts the form fills in rather than asks for. See configs/pick_place_full.yaml.
ROBOT_ROT = {
    "so101": [0.0, 0.0, 0.70710678, 0.70710678],
    "so101_full": [0.0, 0.0, 0.0, 1.0],
}
ROBOT_JOINTS = {
    "so101": {"shoulder_pan": 0.0, "shoulder_lift": -0.6, "elbow_flex": 0.8, "wrist_flex": 0.6,
              "wrist_roll": 0.0, "gripper": 0.0},
    "so101_full": {"shoulder_pan": 0.0, "shoulder_lift": -0.6, "elbow_flex": 0.8, "wrist_flex": 0.6,
                   "wrist_roll": 0.0},
}
PHYSICS = ["physx", "newton_mjwarp", "newton_vbd"]
# Object types that take a `name:` from a catalogue, and their default names.
NAMED = {"ycb": "gelatin_box", "lehome": "burger_patty"}
DEMO_CAM = {"type": "tiled", "prim_path": "{ENV_REGEX_NS}/DemoCam", "resolution": [640, 480],
            "pos": [0.50, -0.36, 0.34], "look_at": [0.20, 0.0, 0.05], "focal_length": 20.0}


class ObjectRow:
    """One line of the objects table: key, type, catalogue name, x y z, static."""

    def __init__(self, parent, on_remove, spec: dict | None = None, key: str = "object"):
        self.frame = ttk.Frame(parent)
        spec = spec or {}
        self.key = tk.StringVar(value=key)
        self.type = tk.StringVar(value=spec.get("type", "cuboid"))
        self.name = tk.StringVar(value=spec.get("name", ""))
        pos = spec.get("pos", [0.22, 0.0, 0.015])
        self.x, self.y, self.z = (tk.StringVar(value=f"{v:g}") for v in pos)
        self.static = tk.BooleanVar(value=bool(spec.get("static", False)))
        self._extra = {k: v for k, v in spec.items()
                       if k not in {"type", "name", "pos", "static"}}   # size, color, scale, ... kept as-is

        ttk.Entry(self.frame, textvariable=self.key, width=10).grid(row=0, column=0, padx=2)
        ttk.OptionMenu(self.frame, self.type, self.type.get(), *sorted(OBJECTS),
                       command=lambda _: self._on_type()).grid(row=0, column=1, padx=2)
        self.name_entry = ttk.Entry(self.frame, textvariable=self.name, width=14)
        self.name_entry.grid(row=0, column=2, padx=2)
        for i, var in enumerate((self.x, self.y, self.z)):
            ttk.Entry(self.frame, textvariable=var, width=7).grid(row=0, column=3 + i, padx=1)
        ttk.Checkbutton(self.frame, text="static", variable=self.static).grid(row=0, column=6, padx=4)
        ttk.Button(self.frame, text="x", width=2, command=lambda: on_remove(self)).grid(row=0, column=7)
        self._on_type()

    def _on_type(self) -> None:
        t = self.type.get()
        self.name_entry.configure(state="normal" if t in NAMED else "disabled")
        if t in NAMED and not self.name.get():
            self.name.set(NAMED[t])

    def to_spec(self) -> tuple[str, dict]:
        spec = dict(self._extra)
        spec["type"] = self.type.get()
        if self.type.get() in NAMED:
            spec["name"] = self.name.get()
        spec["pos"] = [float(self.x.get()), float(self.y.get()), float(self.z.get())]
        if self.static.get():
            spec["static"] = True
        return self.key.get().strip() or "object", spec


class SceneGui:
    def __init__(self, root: tk.Tk, initial: Path | None):
        self.root = root
        root.title("SO-101 scene setup")
        pad = {"padx": 6, "pady": 3}
        self.rows: list[ObjectRow] = []

        f = ttk.Frame(root, padding=10)
        f.grid(sticky="nsew")

        # -- scene ------------------------------------------------------------------
        ttk.Label(f, text="Robot").grid(row=0, column=0, sticky="w", **pad)
        self.robot = tk.StringVar(value="so101_full")
        ttk.OptionMenu(f, self.robot, "so101_full", *sorted(ROBOTS)).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(f, text="Physics").grid(row=0, column=2, sticky="w", **pad)
        self.physics = tk.StringVar(value="physx")
        ttk.OptionMenu(f, self.physics, "physx", *PHYSICS).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(f, text="Envs").grid(row=1, column=0, sticky="w", **pad)
        self.num_envs = tk.StringVar(value="1")
        ttk.Entry(f, textvariable=self.num_envs, width=6).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(f, text="Episode (s)").grid(row=1, column=2, sticky="w", **pad)
        self.episode = tk.StringVar(value="600")
        ttk.Entry(f, textvariable=self.episode, width=6).grid(row=1, column=3, sticky="w", **pad)

        # -- objects -----------------------------------------------------------------
        ttk.Label(f, text="Objects   (key, type, catalogue name, x, y, z)").grid(row=2, column=0, columnspan=4, sticky="w", **pad)
        self.objects = ttk.Frame(f)
        self.objects.grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Button(f, text="+ object", command=self.add_row).grid(row=4, column=0, sticky="w", **pad)

        # -- control ------------------------------------------------------------------
        ttk.Label(f, text="Control").grid(row=5, column=0, sticky="w", **pad)
        self.source = tk.StringVar(value="zmq")
        ttk.OptionMenu(f, self.source, "zmq", *sorted(SOURCES)).grid(row=5, column=1, sticky="w", **pad)
        self.ik = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="pose target (IK)  -- for VR / hand teleop", variable=self.ik).grid(row=5, column=2, columnspan=2, sticky="w", **pad)
        self.camera = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="demo camera (slow on Newton -- see docs/PHYSICS.md)", variable=self.camera).grid(row=6, column=0, columnspan=3, sticky="w", **pad)
        self.vr = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="also start the VR bridge (scripts/vr_gripper_server.py)", variable=self.vr).grid(row=7, column=0, columnspan=3, sticky="w", **pad)

        # -- buttons ------------------------------------------------------------------
        b = ttk.Frame(f)
        b.grid(row=8, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(b, text="Load...", command=self.load_dialog).grid(row=0, column=0, padx=4)
        ttk.Button(b, text="Save as...", command=self.save_dialog).grid(row=0, column=1, padx=4)
        ttk.Button(b, text="Start", command=self.start).grid(row=0, column=2, padx=16)
        self.status = ttk.Label(f, text=f"writes {OUT.relative_to(REPO)} then runs scripts/run.py")
        self.status.grid(row=9, column=0, columnspan=4, sticky="w", **pad)

        if initial:
            self.load(initial)
        else:
            self.add_row()

    # -- rows ---------------------------------------------------------------------------
    def add_row(self, spec: dict | None = None, key: str | None = None) -> None:
        row = ObjectRow(self.objects, self.remove_row, spec, key or ("object" if not self.rows else f"obj{len(self.rows)}"))
        row.frame.grid(sticky="w", pady=1)
        self.rows.append(row)

    def remove_row(self, row: ObjectRow) -> None:
        row.frame.destroy()
        self.rows.remove(row)

    # -- config <-> form ------------------------------------------------------------------
    def to_config(self) -> dict:
        robot = self.robot.get()
        objects = {}
        for r in self.rows:
            k, spec = r.to_spec()
            if k in objects:
                raise ValueError(f"two objects named {k!r}")
            objects[k] = spec
        cfg = {
            "meta": {"name": "gui-scene", "notes": "written by scripts/scene_gui.py"},
            "task": "pick_place",
            "scene": {
                "num_envs": int(self.num_envs.get()),
                "env_spacing": 1.0,
                "robot": {"type": robot, "rot": ROBOT_ROT[robot], "joint_pos": ROBOT_JOINTS[robot]},
                "objects": objects,
            },
            "sim": {"episode_length_s": float(self.episode.get()), "physics": self.physics.get()},
            "control": {"source": self.source.get(), "action_horizon": 1},
        }
        if self.camera.get():
            cfg["scene"]["cameras"] = {"demo_cam": dict(DEMO_CAM)}
        if self.ik.get():
            cfg["control"]["actions"] = "ik"
        if self.source.get() == "zmq":
            cfg["control"]["endpoint"] = "tcp://127.0.0.1:5555"
        return cfg

    def load(self, path: Path) -> None:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        scene, sim, ctl = cfg.get("scene") or {}, cfg.get("sim") or {}, cfg.get("control") or {}
        self.robot.set((scene.get("robot") or {}).get("type", "so101_full"))
        self.physics.set(sim.get("physics", "physx"))
        self.num_envs.set(str(scene.get("num_envs", 1)))
        self.episode.set(str(sim.get("episode_length_s", 600)))
        self.source.set(ctl.get("source", "zero"))
        self.ik.set(ctl.get("actions") == "ik")
        self.camera.set(bool(scene.get("cameras")))
        for r in list(self.rows):
            self.remove_row(r)
        for k, spec in (scene.get("objects") or {}).items():
            self.add_row(spec, k)
        self.status.configure(text=f"loaded {Path(path).name}")

    def load_dialog(self) -> None:
        p = filedialog.askopenfilename(initialdir=REPO / "configs", filetypes=[("YAML", "*.yaml")])
        if p:
            self.load(Path(p))

    def save_dialog(self) -> None:
        p = filedialog.asksaveasfilename(initialdir=REPO / "configs", defaultextension=".yaml",
                                         filetypes=[("YAML", "*.yaml")])
        if p:
            self.write(Path(p))
            self.status.configure(text=f"saved {Path(p).name}")

    def write(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.to_config(), sort_keys=False), encoding="utf-8")

    # -- start --------------------------------------------------------------------------------
    def start(self) -> None:
        try:
            self.write(OUT)
            # Refuse the same things the builder refuses, before a two-minute Kit boot.
            from simbridge.builder import load_config
            load_config(OUT)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("config", str(exc))
            return
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        if self.vr.get() and self.source.get() == "zmq":
            subprocess.Popen([PY, str(REPO / "scripts/vr_gripper_server.py")], cwd=REPO, creationflags=flags)
        cmd = [PY, str(REPO / "scripts/run.py"), "--config", str(OUT), "--viz", "kit", "--steps", "0"]
        subprocess.Popen(cmd, cwd=REPO, creationflags=flags)
        self.status.configure(text="started: " + " ".join(cmd[1:]))


def main() -> None:
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = tk.Tk()
    SceneGui(root, initial)
    root.mainloop()


def demo() -> None:
    """Self-check, headless: the form round-trips a config and refuses a duplicate object key."""
    root = tk.Tk()
    root.withdraw()
    gui = SceneGui(root, REPO / "configs" / "vr_teleop.yaml")
    cfg = gui.to_config()
    assert cfg["scene"]["robot"]["type"] == "so101_full" and cfg["scene"]["robot"]["rot"] == ROBOT_ROT["so101_full"]
    assert cfg["control"]["actions"] == "ik" and cfg["control"]["source"] == "zmq"
    assert set(cfg["scene"]["objects"]) == {"object", "tray"}, cfg["scene"]["objects"]
    assert cfg["scene"]["objects"]["tray"]["type"] == "static_cuboid"   # static-ness is the type here
    src = yaml.safe_load((REPO / "configs" / "vr_teleop.yaml").read_text(encoding="utf-8"))
    assert cfg["scene"]["objects"]["object"]["size"] == src["scene"]["objects"]["object"]["size"], "extra keys must survive"
    from simbridge.builder import load_config  # the builder must accept what the form writes
    tmp = REPO / "configs" / "gui_scene.yaml"
    gui.write(tmp)
    load_config(tmp)
    gui.rows[1].key.set("object")
    try:
        gui.to_config()
    except ValueError as exc:
        assert "two objects" in str(exc)
    else:
        raise AssertionError("duplicate key accepted")
    root.destroy()
    print("scene_gui demo OK: round-trip, builder accepts it, duplicate key refused")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
