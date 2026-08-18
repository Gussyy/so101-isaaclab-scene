# SPDX-License-Identifier: BSD-3-Clause
"""SO-ARM101 scenes and tasks for Isaac Lab 3.0.

Importing this package registers its Gym environments, so ``import so101_scene`` is enough
to make the ids below resolvable.
"""

import gymnasium as gym

from so101_scene import agents

_AGENTS = agents.__name__

gym.register(
    id="SO101-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.reach_env_cfg:SO101ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101ReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="SO101-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.reach_env_cfg:SO101ReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101ReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="SO101-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.pick_place_env_cfg:SO101CubePickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101PickPlacePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="SO101-PickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.pick_place_env_cfg:SO101CubePickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101PickPlacePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# --- kill-check and wide-spawn variants (see docs/DESIGN_visuomotor_flow_matching.md, K1) ---

gym.register(
    id="SO101-PickPlace-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.pick_place_env_cfg:SO101CubePickPlaceBlindEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101PickPlacePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="SO101-PickPlace-Wide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.pick_place_env_cfg:SO101CubePickPlaceWideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101PickPlacePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="SO101-PickPlace-WideBlind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_scene.pick_place_env_cfg:SO101CubePickPlaceWideBlindEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:SO101PickPlacePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
