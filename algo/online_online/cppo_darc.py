import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                        )
    
    def forward(self, x):
        return self.network(x)

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DiscretePolicy, self).__init__()
        with open("debug_log.txt", "a") as f: f.write("DEBUG: DiscretePolicy init start\n")
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        with open("debug_log.txt", "a") as f: f.write("DEBUG: DiscretePolicy init end\n")

    def forward(self, x):
        logits = self.network(x)
        dist = Categorical(logits=logits)
        return dist

    def select_action(self, x):
        dist = self.forward(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class CPPO_DARC(object):
    def __init__(self, config, device):
        with open("debug_log.txt", "a") as f: f.write("DEBUG: CPPO_DARC init start\n")
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 5)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Policy & Value
        if config.get('discrete_action', False):
            with open("debug_log.txt", "a") as f: f.write("DEBUG: Creating DiscretePolicy\n")
            self.policy = DiscretePolicy(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
            with open("debug_log.txt", "a") as f: f.write("DEBUG: DiscretePolicy created and moved to device\n")
        else:
            raise NotImplementedError("Continuous policy not yet ported for CPPO-DARC")
            
        with open("debug_log.txt", "a") as f: f.write("DEBUG: Creating ValueNetwork\n")
        self.value_net = ValueNetwork(config['state_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        with open("debug_log.txt", "a") as f: f.write("DEBUG: ValueNetwork created\n")

        with open("debug_log.txt", "a") as f: f.write("DEBUG: Creating Optimizer\n")
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': config['actor_lr']},
            {'params': self.value_net.parameters(), 'lr': config['critic_lr']}
        ])
        with open("debug_log.txt", "a") as f: f.write("DEBUG: Optimizer created\n")

        self.beta = config.get('constraint_limit', 0.0) 
        self.epsilon = 1e-6
        self.total_it = 0
        with open("debug_log.txt", "a") as f: f.write("DEBUG: CPPO_DARC init end\n")

    def select_action(self, state, test=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy(state)
            if test:
                action = torch.argmax(dist.logits, dim=1)
            else:
                action = dist.sample()
        return action.item()

    def h_function(self, g_tau, beta):
        # h(g(\tau), \beta) = 0.5 - 0.5 * ( (g(\tau) - \beta) / sqrt((g(\tau) - \beta)^2 + \epsilon) )
        term = (g_tau - beta) / torch.sqrt((g_tau - beta)**2 + self.epsilon)
        return 0.5 - 0.5 * term

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        if not hasattr(src_replay_buffer, 'sample_trajectories'):
            return

        # PPO typically needs a batch of trajectories
        # For simplicity in this "Offline/Buffer" PPO version, we sample trajectories
        trajectories = src_replay_buffer.sample_trajectories(batch_size)
        
        if len(trajectories) == 0:
            trajectories = tar_replay_buffer.sample_trajectories(batch_size)
        if len(trajectories) == 0:
            return

        self.total_it += 1

        # Process trajectories to create a PPO batch
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_returns = []
        batch_advantages = []
        
        # Metrics
        avg_R_tau = 0
        avg_g_tau = 0
        avg_weight = 0

        for traj in trajectories:
            states = traj[0] # (T, state_dim)
            actions = traj[1] # (T, action_dim)
            rewards = traj[3] # (T, 1)
            costs = traj[5] # (T, 1)
            
            T = len(states)
            
            # 1. Calculate Constraint Term
            g_tau = torch.sum(costs)
            h_val = self.h_function(g_tau, self.beta)
            log_h = torch.log(h_val + 1e-8)
            
            # 2. Modify Rewards
            # Distributed approach: Add (C / T) to each reward
            # This ensures Sum(r') = Sum(r) + C
            constraint_bonus = (self.config['penalty_coefficient'] * log_h) / T
            modified_rewards = rewards + constraint_bonus
            
            # 3. Calculate GAE and Returns
            # We need values for states. 
            with torch.no_grad():
                values = self.value_net(states) # (T, 1)
                # Next value? For terminal state V=0? Trajectory is complete.
                next_value = 0 
                
            advantages = torch.zeros_like(modified_rewards)
            lasterr = 0
            
            for t in reversed(range(T)):
                next_v = values[t+1] if t < T - 1 else 0
                delta = modified_rewards[t] + self.discount * next_v - values[t]
                advantages[t] = lasterr = delta + self.discount * self.gae_lambda * lasterr
                
            returns = advantages + values
            
            batch_states.append(states)
            batch_actions.append(actions)
            # Recalculate old_log_probs since we don't store them in ReplayBuffer
            # This is a slight approximation (assuming policy hasn't changed much since sampling)
            # Ideally ReplayBuffer should store log_probs. 
            # Given we just sampled from buffer, data might be old.
            # This is "Off-Policy" PPO essentially.
            with torch.no_grad():
                dist = self.policy(states)
                log_probs = dist.log_prob(actions.squeeze().long())
            batch_log_probs.append(log_probs)
            
            batch_advantages.append(advantages)
            batch_returns.append(returns)

            avg_R_tau += torch.sum(rewards).item()
            avg_g_tau += g_tau.item()
            avg_weight += (torch.sum(rewards) + self.config['penalty_coefficient'] * log_h).item()

        # Flatten batch
        b_states = torch.cat(batch_states)
        b_actions = torch.cat(batch_actions).squeeze().long()
        b_log_probs = torch.cat(batch_log_probs).detach()
        b_advantages = torch.cat(batch_advantages).detach().squeeze()
        b_returns = torch.cat(batch_returns).detach().squeeze()
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO Update Loop
        for _ in range(self.ppo_epochs):
            dist = self.policy(b_states)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - b_log_probs)
            
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            values = self.value_net(b_states).squeeze()
            value_loss = F.mse_loss(values, b_returns)
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)
            self.optimizer.step()

        # Logging
        if writer is not None and self.total_it % 100 == 0:
            writer.add_scalar('train/policy_loss', policy_loss.item(), self.total_it)
            writer.add_scalar('train/value_loss', value_loss.item(), self.total_it)
            writer.add_scalar('train/traj_return', avg_R_tau / len(trajectories), self.total_it)
            writer.add_scalar('train/traj_cost', avg_g_tau / len(trajectories), self.total_it)
            writer.add_scalar('train/traj_weight', avg_weight / len(trajectories), self.total_it)

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.value_net.state_dict(), filename + "_critic")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.value_net.load_state_dict(torch.load(filename + "_critic"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
