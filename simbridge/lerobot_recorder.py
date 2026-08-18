# SPDX-License-Identifier: BSD-3-Clause
"""Record simulated episodes in the LeRobot dataset layout.

A dataset produced here can be trained on with ``lerobot-train`` without conversion, which is
what makes the simulator useful as a data source: the 92%-success PPO policy replays as many
demonstrations as wanted, at ~27,000 rendered frames/s, with no teleoperation rig.

Layout written (LeRobotDataset v2):

    <root>/
      meta/info.json           fps, feature schema, episode and frame counts
      meta/episodes.jsonl      one line per episode
      meta/tasks.jsonl         task index -> language string
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/<camera>/episode_000000.mp4

Parquet and MP4 rather than raw arrays because that is what LeRobot reads, and because images
dominate the size: 200 episodes of 250 frames at 128 px is 2.5 GB raw and a fraction of that
encoded.

This module imports no LeRobot code. It writes the layout directly, so demonstrations can be
generated in the Isaac Sim environment without installing LeRobot beside it -- the two never
need to share a process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simbridge.lerobot import JOINTS

CODEBASE_VERSION = "v2.1"


@dataclass
class EpisodeBuffer:
    """Frames of one episode, held until it is known whether the episode succeeded."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray]) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        for name, img in images.items():
            self.images.setdefault(name, []).append(np.asarray(img, dtype=np.uint8))

    def __len__(self) -> int:
        return len(self.states)


class LeRobotRecorder:
    """Write episodes into a LeRobot-shaped dataset directory.

    Args:
        root: Output directory.
        fps: Control rate. 50 for this environment (dt 0.01, decimation 2).
        task: Language string stored against every episode. LeRobot policies condition on it.
        cameras: Camera names to record. Must match the keys in :attr:`ObsPacket.images`.
        state_dim: Width of the state vector; 6 joints by default.
        image_size: ``(height, width)`` of the recorded frames.
    """

    def __init__(
        self,
        root: str | Path,
        fps: int = 50,
        task: str = "Pick up the cube and place it at the target",
        cameras: list[str] | None = None,
        state_dim: int = 6,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.root = Path(root)
        self.fps = int(fps)
        self.task = task
        self.cameras = list(cameras or [])
        self.state_dim = int(state_dim)
        self.image_size = tuple(image_size)

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in self.cameras:
            (self.root / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

        self.episodes: list[dict[str, Any]] = []
        self.total_frames = 0

    # ---- schema -------------------------------------------------------

    def _features(self) -> dict[str, Any]:
        feats: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": JOINTS[: self.state_dim],
            },
            "action": {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": JOINTS[: self.state_dim],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        h, w = self.image_size
        for cam in self.cameras:
            feats[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": float(self.fps), "video.codec": "h264"},
            }
        return feats

    # ---- writing ------------------------------------------------------

    def add_episode(self, buf: EpisodeBuffer) -> int:
        """Append one episode. Returns its index."""
        if len(buf) == 0:
            raise ValueError("refusing to write an empty episode")
        idx = len(self.episodes)
        n = len(buf)

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("writing LeRobot datasets needs pandas (pip install pandas pyarrow)") from exc

        frame = pd.DataFrame(
            {
                "observation.state": [s.astype(np.float32) for s in buf.states],
                "action": [a.astype(np.float32) for a in buf.actions],
                "timestamp": np.arange(n, dtype=np.float32) / self.fps,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, idx, dtype=np.int64),
                "index": np.arange(self.total_frames, self.total_frames + n, dtype=np.int64),
                "task_index": np.zeros(n, dtype=np.int64),
            }
        )
        frame.to_parquet(self.root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet", index=False)

        for cam in self.cameras:
            frames = buf.images.get(cam)
            if not frames:
                continue
            self._write_video(self.root / "videos" / "chunk-000" / cam / f"episode_{idx:06d}.mp4", frames)

        self.episodes.append({"episode_index": idx, "tasks": [self.task], "length": n})
        self.total_frames += n
        return idx

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        import imageio.v2 as imageio

        # macro_block_size=1 so a 128x128 frame is not silently resized to a multiple of 16,
        # which would desynchronise the video from the parquet rows.
        writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error")
        try:
            for fr in frames:
                writer.append_data(fr)
        finally:
            writer.close()

    def finalize(self) -> Path:
        """Write the metadata LeRobot reads on load."""
        with (self.root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for ep in self.episodes:
                fh.write(json.dumps(ep) + "\n")

        with (self.root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": "so101",
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_videos": len(self.episodes) * len(self.cameras),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(),
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return self.root

    def summary(self) -> str:
        mb = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file()) / 1e6
        return (
            f"{len(self.episodes)} episodes, {self.total_frames} frames, "
            f"{len(self.cameras)} camera(s), {mb:.1f} MB at {self.root}"
        )


def demo() -> None:
    """Self-check: write a small dataset and read back the layout LeRobot expects."""
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp()) / "ds"
    rec = LeRobotRecorder(root, fps=50, cameras=["scene_cam"], image_size=(32, 32))

    rng = np.random.default_rng(0)
    for _ in range(3):
        buf = EpisodeBuffer()
        for _ in range(10):
            buf.add(
                rng.random(6).astype(np.float32),
                rng.random(6).astype(np.float32),
                {"scene_cam": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)},
            )
        rec.add_episode(buf)
    rec.finalize()

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3 and info["total_frames"] == 30
    assert info["fps"] == 50 and info["robot_type"] == "so101"
    assert "observation.images.scene_cam" in info["features"]
    assert info["features"]["observation.state"]["shape"] == [6]

    eps = [json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(eps) == 3 and eps[0]["length"] == 10

    for i in range(3):
        assert (root / "data" / "chunk-000" / f"episode_{i:06d}.parquet").exists()
        v = root / "videos" / "chunk-000" / "scene_cam" / f"episode_{i:06d}.mp4"
        assert v.exists() and v.stat().st_size > 0, v

    import pandas as pd

    df = pd.read_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
    assert list(df.columns)[:2] == ["observation.state", "action"]
    assert len(df) == 10 and df["episode_index"].iloc[0] == 0
    # index must run continuously across episodes, not restart -- LeRobot uses it as a global key
    df2 = pd.read_parquet(root / "data" / "chunk-000" / "episode_000001.parquet")
    assert df2["index"].iloc[0] == 10, df2["index"].iloc[0]

    print("lerobot recorder OK: parquet, mp4, info.json, episodes.jsonl, global frame index")
    print(f"  {rec.summary()}")
    shutil.rmtree(root.parent, ignore_errors=True)


if __name__ == "__main__":
    demo()
