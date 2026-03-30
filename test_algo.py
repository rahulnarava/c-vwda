import torch
from envs.gridworld.call_gridworld_env import call_gridworld_env
from algo.online_online.cppo_darc import CPPO_DARC

print("Imports successful")

config = {
    'env_name': 'gridworld', 
    'state_dim': 2,
    'action_dim': 8,
    'max_action': 1.0,
    'hidden_sizes': 64,
    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    'gamma': 0.99,
    'constraint_limit': 0.99,
    'penalty_coefficient': 1.0,
    'discrete_action': True
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Initializing Algo...")
try:
    policy = CPPO_DARC(config, device)
    print("Algo initialized")
except Exception as e:
    print(f"Algo failed: {e}")

print("Initializing Env...")
try:
    env_wrapper = call_gridworld_env({'env_name': 'gridworld'})
    env = env_wrapper.env
    print("Env initialized")
    s = env.reset()
    print(f"Env reset: {s}")
    ns, r, d, i = env.step(0)
    print(f"Env step: {ns}, {r}, {d}, {i}")
except Exception as e:
    print(f"Env failed: {e}")
