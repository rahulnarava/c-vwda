from typing import Dict
from .gridworld_env import GridworldEnvironment
from gym.wrappers import TimeLimit

def call_gridworld_env(env_config: Dict):
    env_name = env_config['env_name'].lower()
    
    # We expect env_name to contain 'gridworld' or be 'gridworld-a'
    # Actually, the user might pass 'gridworld-a' or 'gridworld'
    
    # Just return Gridworld A for now as requested
    # Just return Gridworld A for now as requested
    # Thesis Custom1 uses stay_action=False (8 actions)
    env = GridworldEnvironment(stay_action=False)
    
    # Max steps? Gridworld usually short. Thesis used 50.
    return TimeLimit(env, max_episode_steps=50)
