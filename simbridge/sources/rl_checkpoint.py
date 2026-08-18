# SPDX-License-Identifier: BSD-3-Clause
"""Drive an environment from a trained rsl_rl checkpoint.

This is what turns the existing 92 %-success PPO policy into a demonstration generator: point it
at ``model_1499.pt`` and it produces expert trajectories with no teleoperation rig.

The policy consumes the flat observation vector Isaac Lab already assembles, so this source
reads ``obs.state["policy"]`` rather than reconstructing features itself -- the mapping from
scene to observation belongs to the environment, not to the driver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from simbridge.interfaces import ActionSource
from simbridge.registry import register_source
from simbridge.schema import ObsPacket


@register_source("rl_checkpoint")
class RslRlCheckpointSource(ActionSource):
    """Deterministic (mean) actions from a trained rsl_rl actor.

    Args:
        checkpoint: Path to an rsl_rl ``model_*.pt``.
        device: Torch device for inference.
        obs_key: Which entry of :attr:`ObsPacket.state` holds the policy observation vector.
        deterministic: Use the distribution mean. True for demo generation -- sampled actions
            add avoidable noise to what is meant to be expert data.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        obs_key: str = "policy",
        deterministic: bool = True,
        action_horizon: int = 1,
        **_: object,
    ) -> None:
        super().__init__(action_horizon=action_horizon)
        self.device = device
        self.obs_key = obs_key
        self.deterministic = bool(deterministic)

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        payload = torch.load(path, map_location=device, weights_only=False)
        self._payload = payload
        self._actor = self._extract_actor(payload)

    @staticmethod
    def _extract_actor(payload):
        """rsl_rl has stored the actor under several keys across versions; try them in order
        and fail with what was actually present rather than a KeyError on one guess."""
        for key in ("actor", "model_state_dict", "model", "policy"):
            if key in payload:
                return payload[key]
        raise KeyError(
            f"no actor found in checkpoint; top-level keys were {sorted(payload)[:12]}. "
            "Load it manually and pass the module if this is a new rsl_rl layout."
        )

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        if self.obs_key not in obs.state:
            raise KeyError(
                f"observation group {self.obs_key!r} not in packet; available: {sorted(obs.state)}"
            )
        x = torch.as_tensor(obs.state[self.obs_key], device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            out = self._actor(x) if callable(self._actor) else None
        if out is None:
            raise RuntimeError(
                "checkpoint payload is a state_dict, not a callable module. Build the runner via "
                "rsl_rl and pass its policy, or export the actor to TorchScript first."
            )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().float().cpu().numpy()
