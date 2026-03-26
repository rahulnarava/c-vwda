import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints, Categorical
from torch.distributions.transforms import Transform
import math

# Re-using Policy and TanhTransform from DARC implementation or defining simplified one
# For simplicity and consistency, let's redefine necessary classes or import if possible.
# Since we are in a different file, it's safer to copy the Policy class or similar structure to avoid circular imports or dependency issues.

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


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


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        
        return action * self.max_action, logprob, mean * self.max_action
        return action * self.max_action, logprob, mean * self.max_action


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DiscretePolicy, self).__init__()
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            # No activation at end, these are logits
        )

    def forward(self, x, get_logprob=False):
        logits = self.network(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        if get_logprob:
            logprob = dist.log_prob(action)
        else:
            logprob = None
            
        # For discrete, mean action doesn't make sense as a single scalar in the same way,
        # but we can return the argmax (most likely action) as "mean" equivalent
        mean = torch.argmax(logits, dim=1)
        
        # Return action, logprob, mean
        # Actions are integers 0..N-1
        return action.float(), logprob, mean.float()

class CDARC(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        
        # Policy
        # Policy
        if config.get('discrete_action', False):
            self.policy = DiscretePolicy(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        else:
            self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

        self.beta = config.get('constraint_limit', 0.0) # default limit beta
        self.epsilon = 1e-6

        self.total_it = 0

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def h_function(self, g_tau, beta):
        # h(g(\tau), \beta) = 0.5 - 0.5 * ( (g(\tau) - \beta) / sqrt((g(\tau) - \beta)^2 + \epsilon) )
        term = (g_tau - beta) / torch.sqrt((g_tau - beta)**2 + self.epsilon)
        return 0.5 - 0.5 * term

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        # We only care about the target domain trajectories for calculation as per user implication
        # "target has constraints". Source might not.
        # But wait, the objective is min D_KL(q_pi || p_C). q_pi is the policy distribution.
        # The user says "both source and target dynamics remain same".
        # So essentially we are training on the environment (which is effectively target).
        # We will use tar_replay_buffer because in train.py, tar_env is the one we primarily interact with or care about in some modes, 
        # but in 'online_online' mode (Mode 0), both are interacted with.
        # Since dynamics are same, we can use data from both?
        # The user request says "in my setting both source and target dynamics remain same".
        # This implies we can treat them as the same environment.
        # However, to be safe and consistent with "target has constraints", I will prioritize tar_replay_buffer data
        # or merge them if they are both collection data.
        # Since train.py populates both buffers, let's just use tar_replay_buffer if it has data, or src if it has data.
        # Actually, let's just use whatever buffer has trajectories.
        # Assuming we are running in Mode 0 (Online/Online) or similar.
        
        # User requested DARC-style interaction:
        # 1. Sample Source Buffer (as main training data)
        # 2. "Interaction" component: Apply Constraint Correction (simulating target constraint on source data)
        # 3. Update Policy

        # Check headers
        if not hasattr(src_replay_buffer, 'sample_trajectories'):
            return

        # Warmup phase (DARC style) - though for PG we might just start
        if self.total_it < 0: # Disable warmup for now or set to some value
             pass

        # 1. Sample Source Trajectories
        trajectories = src_replay_buffer.sample_trajectories(batch_size)
        
        if len(trajectories) == 0:
            # Fallback to target if source empty (e.g. at very start if mode 0 pushes target first? Unlikely)
            trajectories = tar_replay_buffer.sample_trajectories(batch_size)

        if len(trajectories) == 0:
            return

        self.total_it += 1

        # 2. Compute "Correction" / Constraint Term
        # In DARC, this is the classifier. Here, it's the analytical constraint.
        # We calculate it per trajectory.
        
        # Unpack trajectories
        states_list = [t[0] for t in trajectories]
        actions_list = [t[1] for t in trajectories]
        rewards_list = [t[3] for t in trajectories] # (T, 1)
        costs_list = [t[5] for t in trajectories]   # (T, 1)

        # 3. Policy Update with Corrected Reward
        policy_loss_total = 0
        # Not explicitly minimizing entropy in the simplified gradient, but derivation mentions it.
        # The derivation final equation:
        # \nabla_\theta J = E [ ( R(\tau) + \log h(g(\tau), \beta) ) \sum \nabla \log \pi ] + \nabla entropy
        # R(\tau) is sum of rewards.
        # g(\tau) is sum of costs.

        # We will approximate this expectation by averaging over the sampled trajectories.
        # Optimization: maximize J <=> minimize -J.
        
        loss_list = []

        for i in range(len(trajectories)):
            states = states_list[i]
            actions = actions_list[i]
            rewards = rewards_list[i]
            costs = costs_list[i]
            
            # Calculate Trajectory Return R(\tau)
            R_tau = torch.sum(rewards)
            
            # Calculate Trajectory Cost g(\tau)
            g_tau = torch.sum(costs)
            
            # Calculate h(g(\tau), \beta)
            h_val = self.h_function(g_tau, self.beta)
            
            # Avoid log(0)
            log_h = torch.log(h_val + 1e-8)
            
            weight = R_tau + self.config['penalty_coefficient'] * log_h
            
            # Recalculate log_probs for the current policy
            if self.config.get('discrete_action', False):
                 logits = self.policy.network(states)
                 dist = Categorical(logits=logits)
                 action_indices = actions.squeeze().long()
                 log_prob_a = dist.log_prob(action_indices)
            else:
                mu_logstd = self.policy.network(states)
                mu, logstd = mu_logstd.chunk(2, dim=1)
                logstd = torch.clamp(logstd, -20, 2)
                std = logstd.exp()
                dist = Normal(mu, std)
                
                clipped_actions = torch.clamp(actions / self.policy.max_action, -0.999999, 0.999999)
                x_actions = 0.5 * (torch.log1p(clipped_actions) - torch.log1p(-clipped_actions))
                log_prob_x = dist.log_prob(x_actions).sum(axis=-1, keepdim=True)
                log_det = 2. * (math.log(2.) - x_actions - F.softplus(-2. * x_actions))
                log_det = log_det.sum(axis=-1, keepdim=True)
                log_prob_a = log_prob_x - log_det
            
            sum_log_probs = torch.sum(log_prob_a)
            
            loss = - weight.detach() * sum_log_probs
            loss_list.append(loss)
            
            policy_loss_total += loss.item()

            if writer is not None and self.total_it % 100 == 0:
                 writer.add_scalar('train/traj_return', R_tau, self.total_it)
                 writer.add_scalar('train/traj_cost', g_tau, self.total_it)
                 writer.add_scalar('train/traj_weight', weight, self.total_it)

        if len(loss_list) > 0:
            policy_loss = torch.stack(loss_list).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            if writer is not None and self.total_it % 100 == 0:
                writer.add_scalar('train/policy_loss', policy_loss, self.total_it)

    def update_target(self):
        pass # No target networks in this simplified PG version

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
