import numpy as np
from typing import Dict
from .gridworld_env import GridworldEnvironment
from gym.wrappers import TimeLimit

# Source env: goal at (6,0), no constraint wall
# Target env: goal at (6,0), wall at row 3 cols 0-3
_WALL = [(3, 0), (3, 1), (3, 2), (3, 3)]
_START_STATES = [(i, j) for i in range(3) for j in range(2)]
_TERMINALS = [(6, 0)]

def _make_reward():
    r = np.zeros((7, 7))
    r[6, 0] = 1.0
    return r

def call_gridworld_env(env_config: Dict):
    is_target = env_config.get('shift_level', None) is not None
    unsafe = _WALL if is_target else []
    env = GridworldEnvironment(
        r=_make_reward(),
        t=list(_TERMINALS),
        unsafe_states=list(unsafe),
        start_states=list(_START_STATES),
        stay_action=False,
    )
    return TimeLimit(env, max_episode_steps=50)
