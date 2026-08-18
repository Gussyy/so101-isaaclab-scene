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

        # rsl_rl writes two things: model_<n>.pt holds raw state_dicts, and exported/policy.pt is
        # a TorchScript module that can simply be called. Prefer the exported one -- rebuilding
        # the actor from a state_dict means reconstructing the exact network the run used, and
        # getting that subtly wrong produces a policy that loads cleanly and acts badly.
        exported = path.parent / "exported" / "policy.pt"
        if path.name.startswith("model_") and exported.is_file():
            print(f"[rl_checkpoint] using TorchScript export: {exported}")
            path = exported

        try:
            loaded = torch.jit.load(str(path), map_location=device)
            loaded.eval()
            self._actor = loaded
            self._payload = None
            return
        except Exception:
            pass  # not TorchScript; fall through to the checkpoint dict

        payload = torch.load(path, map_location=device, weights_only=False)
        self._payload = payload
        self._actor = self._extract_actor(payload)

    @staticmethod
    def _extract_actor(payload):
        """Find something callable in a raw checkpoint.

        rsl_rl has moved this key across versions, so try the known names and fail with what was
        actually present. A bare ``actor_state_dict`` is weights only -- there is no network to
        run -- so that case points at the TorchScript export instead of guessing an architecture.
        """
        for key in ("actor", "model", "policy"):
            obj = payload.get(key)
            if callable(obj):
                return obj
        if "actor_state_dict" in payload:
            raise KeyError(
                "this checkpoint holds state_dicts only (actor_state_dict), not a runnable module. "
                "Point at exported/policy.pt in the same run directory, which rsl_rl writes as "
                "TorchScript, or rebuild the runner with rsl_rl and pass its policy."
            )
        raise KeyError(f"no actor found in checkpoint; top-level keys were {sorted(payload)[:12]}")

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
