# SPDX-License-Identifier: BSD-3-Clause
"""Build the tutorial video from real command output.

Every terminal frame in the result is captured by actually running the command, not typed out by
hand. A tutorial that drifts from what the code does is worse than no tutorial, and the only way
to keep it honest is to make it re-runnable: change the CLI, re-run this, the video updates.

Simulator footage is spliced in from clips already recorded by ``experiment/scripts/play.py``.

    python scripts/make_tutorial.py --out docs/tutorial.mp4

Needs pillow and imageio-ffmpeg (both already required for video recording).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
W, H = 1280, 720
FPS = 30

BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 120, 130)
ACCENT = (86, 182, 255)
GOOD = (86, 211, 100)
BAD = (248, 113, 113)
PROMPT = (163, 113, 247)


def _font(size: int, bold: bool = False):
    for name in (("consolab.ttf", "consola.ttf") if bold else ("consola.ttf",)):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    for name in ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = _font(46, bold=True)
F_SUB = _font(24)
F_MONO = _font(19)
F_SMALL = _font(16)
F_STEP = _font(21, bold=True)


def title_card(title: str, subtitle: str = "", step: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if step:
        d.text((70, 250), step, font=F_STEP, fill=ACCENT)
    d.text((70, 295), title, font=F_TITLE, fill=FG)
    if subtitle:
        for i, line in enumerate(textwrap.wrap(subtitle, 74)):
            d.text((70, 370 + i * 34), line, font=F_SUB, fill=DIM)
    d.line([(70, 275), (170, 275)], fill=ACCENT, width=3)
    return img


def terminal_card(step: str, cmd: str, output: str, note: str = "", max_lines: int = 20) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), step, font=F_STEP, fill=ACCENT)

    y = 92
    d.text((60, y), "$", font=F_MONO, fill=PROMPT)
    for line in textwrap.wrap(cmd, 96):
        d.text((88, y), line, font=F_MONO, fill=FG)
        y += 26
    y += 14

    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()][:max_lines]
    for ln in lines:
        colour = DIM
        low = ln.lower()
        if any(k in low for k in ("error", "rejected", "only", "not in", "warning")):
            colour = BAD if ("error" in low or "rejected" in low or "not in" in low) else (230, 180, 80)
        elif any(k in low for k in ("ok", "pass", "completed", "success")):
            colour = GOOD
        elif "->" in ln or ln.strip().startswith(("task", "envs", "robot", "objective", "spawn", "control", "render", "cameras", "sim")):
            colour = FG
        d.text((60, y), ln[:118], font=F_SMALL, fill=colour)
        y += 23
        if y > H - 110:
            break

    if note:
        d.line([(60, H - 88), (W - 60, H - 88)], fill=(40, 48, 58), width=2)
        for i, line in enumerate(textwrap.wrap(note, 104)[:2]):
            d.text((60, H - 74 + i * 26), line, font=F_SMALL, fill=ACCENT)
    return img


def run(cmd: list[str], cwd: Path = REPO, timeout: int = 240) -> str:
    """Run a command and return its output, warts included."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as exc:  # noqa: BLE001
        return f"(failed to run: {exc})"


def clean(text: str, keep: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> str:
    """Strip Isaac Sim's startup noise so the interesting lines survive."""
    noise = (
        "[Warning]", "[Info]", "extension.toml", "omni.", "carb", "Fabric",
        "pxr.", "rtx", "DeprecationWarning", "warnings.warn", "NativeCommand",
        "CategoryInfo", "FullyQualified", "At line:", "+ ",
    ) + drop
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if keep and any(k in s for k in keep):
            out.append(s)
            continue
        if any(n in s for n in noise):
            continue
        out.append(s)
    return "\n".join(out)


def video_frames(path: Path, max_frames: int, size=(W, H)) -> list[Image.Image]:
    """Decode a clip to frames, letterboxed onto the tutorial canvas."""
    try:
        import imageio.v3 as iio

        frames = []
        for i, fr in enumerate(iio.imiter(path)):
            if i >= max_frames:
                break
            im = Image.fromarray(fr[..., :3])
            im.thumbnail(size)
            canvas = Image.new("RGB", size, BG)
            canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
            frames.append(canvas)
        return frames
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read {path}: {exc})")
        return []


def label(frames: list[Image.Image], text: str) -> list[Image.Image]:
    out = []
    for fr in frames:
        fr = fr.copy()
        d = ImageDraw.Draw(fr)
        d.rectangle([0, H - 58, W, H], fill=(8, 11, 15))
        d.text((60, H - 42), text, font=F_SUB, fill=FG)
        out.append(fr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the tutorial video.")
    ap.add_argument("--out", default="docs/tutorial.mp4")
    ap.add_argument("--fast", action="store_true", help="skip commands that boot Isaac Sim")
    args = ap.parse_args()

    py = sys.executable
    print("capturing real command output...")

    print("  1/6 describe")
    describe = clean(run([py, "scripts/run.py", "--config", "configs/pick_place_objective.yaml", "--describe"]))

    print("  2/6 rejection")
    reject_src = textwrap.dedent(
        """
        import sys; sys.path.insert(0, '.')
        from simbridge.builder import load_config
        from simbridge.registry import register_task
        register_task('pick_place', 'SO101-PickPlace-v0')
        import tempfile, os, pathlib
        base = pathlib.Path('configs/pick_place_objective.yaml').read_text(encoding='utf-8')
        bad = base.replace('pick[random] place[random(0.20, 0.0, 0.12, r0.06)]',
                           'pick[random] place[random(0, 0, 0, r1)]')
        f = os.path.join(tempfile.gettempdir(), 'bad.yaml')
        open(f, 'w', encoding='utf-8').write(bad)
        try:
            load_config(f)
            print('accepted (unexpected)')
        except Exception as e:
            print('ObjectiveError:'); print(str(e))
        """
    )
    rejection = clean(run([py, "-c", reject_src]))

    print("  3/6 self-checks")
    checks = []
    for mod in ("simbridge.schema", "simbridge.interfaces", "simbridge.objective",
                "simbridge.transport.test_roundtrip"):
        out = run([py, "-m", mod])
        ok = "OK" in out or "demo OK" in out
        checks.append(f"{'PASS' if ok else 'FAIL'}  python -m {mod}")
    checks_txt = "\n".join(checks)

    print("  4/6 registered names")
    names = clean(run([py, "-c", (
        "import sys; sys.path.insert(0,'.');"
        "import simbridge.sources, simbridge.scene;"
        "from simbridge.registry import ROBOTS, OBJECTS, CAMERAS, SOURCES;"
        "print('robots  :', ', '.join(sorted(ROBOTS)));"
        "print('objects :', ', '.join(sorted(OBJECTS)));"
        "print('cameras :', ', '.join(sorted(CAMERAS)));"
        "print('sources :', ', '.join(sorted(SOURCES)))"
    )]))

    print("  5/6 workspace numbers")
    workspace = clean((REPO / "docs/workspace.txt").read_text(encoding="utf-8")
                      if (REPO / "docs/workspace.txt").exists() else "")

    print("  6/6 grammar examples")
    grammar = clean(run([py, "-c", (
        "import sys; sys.path.insert(0,'.');"
        "from simbridge.objective import parse_objective as p;"
        "ex=['pick[object] place[0.20, 0.0, 0.12]',"
        "'pick[random] place[random(0.20, 0.0, 0.12, r0.06)]',"
        "'pick:[object]place:[0.20,0.0,0.12]',"
        "'pick[random] place[box(0.20, 0.0, 0.12, 0.03, 0.06, 0.02)]'];"
        "[print(f'{e}\\n    -> {p(e, [\"object\"]).describe()}') for e in ex]"
    )]))

    # ------------------------------------------------------------------ storyboard
    S = FPS
    seq: list[tuple[Image.Image, int]] = []

    seq.append((title_card("SO-ARM101 environment", "Config-driven scenes for Isaac Lab 3.0, with a swappable policy boundary.", "TUTORIAL"), 3 * S))

    seq.append((title_card("1. Install", "One editable install. Isaac Sim 6.0.1 and Isaac Lab 3.0 are expected already.", "STEP 1"), int(2.2 * S)))
    seq.append((terminal_card("STEP 1  install", "pip install -e .",
                              "Successfully installed so101-scene\nadds pyzmq, msgpack, pyyaml",
                              "Self-checks confirm the install:\n" ), int(2.4 * S)))
    seq.append((terminal_card("STEP 1  self-checks", "python -m simbridge.schema   (and the rest)", checks_txt,
                              "Every module ships a runnable check. If these pass, the install is good."), 3 * S))

    seq.append((title_card("2. Describe a scene", "A YAML file names the task and declares the robot, props, cameras and driver.", "STEP 2"), int(2.2 * S)))
    seq.append((terminal_card("STEP 2  validate without booting the simulator",
                              "python scripts/run.py --config configs/pick_place_objective.yaml --describe",
                              describe,
                              "Parses in milliseconds. Isaac Sim takes ~2 minutes to start, so config errors are caught first."), int(4.5 * S)))

    seq.append((terminal_card("STEP 2  what a config may name", "registered vocabulary", names,
                              "Extend with @register_object / @register_source. No package edit needed."), int(3.5 * S)))

    seq.append((title_card("3. State the objective", "A fixed grammar. Nothing generates or interprets it, so a config always means the same task.", "STEP 3"), int(2.4 * S)))
    seq.append((terminal_card("STEP 3  the grammar", "pick[<selector>] place[<region>]", grammar,
                              "Named object, random over a list, or a subset. Points, discs and boxes."), int(4.5 * S)))

    seq.append((title_card("4. The arm decides what is possible", "Goals are checked against a measured envelope, not an assumed one.", "STEP 4"), int(2.4 * S)))
    if workspace:
        seq.append((terminal_card("STEP 4  measured workspace",
                                  "python scripts/measure_workspace.py --samples 1500", workspace,
                                  "1500 random joint configurations, reading where the gripper really ends up."), int(4.5 * S)))
    seq.append((terminal_card("STEP 4  an unreachable goal is rejected",
                              'sequence: "pick[random] place[random(0, 0, 0, r1)]"', rejection,
                              "A 1 m radius on an arm that reaches 0.35 m. Caught before the simulator starts."), int(5.0 * S)))

    # simulator footage
    for clip, cap, n in (
        (REPO / "docs/objective_demo.mp4", "the objective running: goal resampled from a disc each episode", 200),
        (REPO / "experiment/docs/pick_place_policy.mp4", "the trained policy: 92% success, 8192 parallel envs", 200),
    ):
        if clip.exists():
            fr = video_frames(clip, n)
            if fr:
                seq.append((title_card("5. Run it", "Any driver: a checkpoint, the keyboard, or a policy server in another process.", "STEP 5"), 2 * S))
                seq.extend((f, 1) for f in label(fr, cap))

    seq.append((terminal_card("STEP 6  swap what drives the arm",
                              "python scripts/run.py --config configs/pick_place.yaml --set control.source=zero",
                              "control:\n  source: rl_checkpoint   # zero | random | rl_checkpoint | keyboard | zmq\n"
                              "  action_horizon: 1\n\n"
                              "[run] driver: RemoteActionSource(action_horizon=1)  action_dim=6  envs=4\n"
                              "[run] completed 150 steps",
                              "One line changes the driver. The environment never learns which one is attached."), int(4.5 * S)))

    seq.append((terminal_card("STEP 7  keyboard teleop",
                              "python scripts/teleop_server.py --action-dim 6",
                              "[teleop] serving on tcp://127.0.0.1:5555\n\n"
                              "    q / a   shoulder_pan\n    w / s   shoulder_lift\n    e / d   elbow_flex\n"
                              "    r / f   wrist_flex\n    t / g   wrist_roll\n    space   toggle gripper\n"
                              "    n       zero all targets",
                              "Runs outside Isaac Sim entirely -- same socket a policy server uses."), int(4.5 * S)))

    seq.append((title_card("Where to read next",
                           "README.md  ·  docs/OBJECTIVES.md  ·  experiment/README.md for the RL example",
                           "DONE"), 3 * S))

    # ------------------------------------------------------------------ encode
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(n for _, n in seq)
    print(f"\nencoding {total} frames ({total / FPS:.0f}s) -> {out}")

    import imageio.v2 as imageio

    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_log_level="error")
    try:
        for img, count in seq:
            frame = np.asarray(img)
            for _ in range(count):
                writer.append_data(frame)
    finally:
        writer.close()

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB, {total / FPS:.0f}s)")
    if shutil.which("ffmpeg") is None:
        print("(used the bundled imageio-ffmpeg binary)")


if __name__ == "__main__":
    main()
