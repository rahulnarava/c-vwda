"""
C-VWDA (Discrete): Constrained Value-Weighted Domain Adaptation
-- Discrete action spaces only (Gridworld etc.) --

Key difference from the continuous variant:
  - Q-networks output per-action values  Q(s) → (B, action_dim)
    rather than Q(s, a) → scalar, so there is no need to pass
    one-hot actions through the critic.
  - Actor update uses the *expected* Q-value under the current
    policy:  V(s) = Σ_a π(a|s) [Q(s,a) − α log π(a|s)],
    which is fully differentiable w.r.t. the policy logits.
  - Bellman targets likewise use the expected next-state value.
  - Classifiers and the cost network still receive one-hot actions
    (they model per-transition quantities and need the explicit
    action input).

Algorithm outline
-----------------
Networks
  - Policy π_θ          (Categorical, outputs logits over actions)
  - Twin reward critics  Q^r_{ψ1}(s), Q^r_{ψ2}(s) → (action_dim,)  + targets
  - Cost critic          Q^c_ξ(s) → (action_dim,)                    + target
  - Cost network         c_φ(s,a) → [0,1]  (sigmoid, regression)
  - SA  classifier       q^sa_ω(s,a)    (DARC-style binary)
  - SAS classifier       q^sas_ω(s,a,s') (DARC-style binary)

Per-step training loop
  1. Collect: store (s,a,r,s') in B_src
  2. Every K_cls steps: update SA / SAS classifiers via BCE
  3. Sample mini-batch from B_src
  4. Importance weights:
       log w = logit(q^sas(s,a,s')) − logit(q^sa(s,a))
       w     = exp(clip(log_w, −W, W))
  Phase 1 – Cost learning
  5. Regress c_φ toward min(½|w−1|, 1)
  Phase 2 – Primal step
  6. Update twin reward critics (discrete Bellman with expected value)
  7. Update cost critic (cost Bellman, no entropy)
  8. Update policy:  maximize  Σ_a π(a|s)[Q^r(s,a) − λ Q^c(s,a) − α log π(a|s)]
  Phase 3 – Dual step
  9. λ ← [λ + η_λ (J^c − ε)]_+
  10. Soft-update target networks
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DiscretePolicy(nn.Module):
    """Categorical policy: state → logits over actions."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)

    def forward(self, state):
        """Return logits, probs, log_probs for all actions."""
        logits = self.network(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return logits, probs, log_probs

    def sample(self, state):
        """Sample an action; return (action_index, logprob)."""
        logits, _, _ = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()                              # (B,)
        logprob = dist.log_prob(action).unsqueeze(-1)       # (B,1)
        return action, logprob

    def greedy(self, state):
        """Deterministic greedy action."""
        logits, _, _ = self.forward(state)
        return torch.argmax(logits, dim=-1)                 # (B,)


class DiscreteDoubleQFunc(nn.Module):
    """Twin Q-networks: state → per-action Q-values (B, action_dim)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network1 = MLPNetwork(state_dim, action_dim, hidden_size)
        self.network2 = MLPNetwork(state_dim, action_dim, hidden_size)

    def forward(self, state):
        return self.network1(state), self.network2(state)


class DiscreteSingleQFunc(nn.Module):
    """Single Q-network: state → per-action Q-values (B, action_dim)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)

    def forward(self, state):
        return self.network(state)


class CostNetwork(nn.Module):
    """
    c_φ(s, a) → [0,1].  Receives one-hot action because it models
    per-transition domain divergence cost.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action_onehot):
        x = torch.cat((state, action_onehot), dim=1)
        return torch.sigmoid(self.network(x))


class DARCClassifier(nn.Module):
    """
    Binary SA / SAS classifiers (single logit → BCEWithLogitsLoss).
    Receives one-hot actions.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256,
                 gaussian_noise_std=1.0):
        super().__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_net  = MLPNetwork(state_dim + action_dim,     1, hidden_size)
        self.sas_net = MLPNetwork(2 * state_dim + action_dim, 1, hidden_size)

    def forward_sa(self, state, action_onehot, with_noise=False):
        x = torch.cat([state, action_onehot], dim=-1)
        if with_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.sa_net(x)

    def forward_sas(self, state, action_onehot, next_state, with_noise=False):
        x = torch.cat([state, action_onehot, next_state], dim=-1)
        if with_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.sas_net(x)


# ---------------------------------------------------------------------------
# Helpers: buffer action ↔ one-hot / index
# ---------------------------------------------------------------------------
# The replay buffer may store discrete actions in two formats depending on the
# training loop:
#   1. One-hot vectors  (B, action_dim)   – e.g. run_train.py for gridworld
#   2. Scalar indices   (B, 1)            – other possible callers
# The helpers below detect the format and convert as needed.

def _action_to_onehot(action, num_actions):
    """Buffer action → one-hot (B, action_dim)."""
    if action.shape[-1] == num_actions:
        return action.float()                               # already one-hot
    idx = action.long().squeeze(-1)                         # (B,)
    return F.one_hot(idx, num_classes=num_actions).float()


def _action_to_index(action, num_actions):
    """Buffer action → integer index (B,)."""
    if action.shape[-1] == num_actions:
        return action.argmax(dim=-1)                        # one-hot → index
    return action.long().squeeze(-1)                        # scalar → index


# ---------------------------------------------------------------------------
# C-VWDA Discrete
# ---------------------------------------------------------------------------

class C_VWDA_Discrete(object):
    """
    Constrained Value-Weighted Domain Adaptation — discrete action variant.

    Uses per-action Q-networks and the expected-value formulation for
    policy / critic updates, giving proper gradient flow through the
    policy logits.
    """

    def __init__(self, config, device, target_entropy=None):
        self.config  = config
        self.device  = device
        self.discount = config['gamma']
        self.tau      = config['tau']
        self.action_dim = config['action_dim']

        # Target entropy for discrete: 0.98 * log(|A|)
        if target_entropy is not None:
            self.target_entropy = target_entropy
        else:
            self.target_entropy = -0.98 * np.log(1.0 / self.action_dim)

        # C-VWDA hyperparameters
        self.weight_clip       = config.get('weight_clip', 2.0)
        self.constraint_budget = config.get('constraint_budget', 0.99)
        self.dual_step_size    = config.get('dual_step_size', 3e-4)
        self.cls_update_freq   = config.get('tar_env_interact_freq', 10)
        self.warmup_steps      = int(config.get('warmup_steps', 1e4))
        # Dual variable update is delayed beyond warmup so the cost critic
        # has time to stabilise before λ starts accumulating.
        self.dual_warmup_steps = int(config.get('dual_warmup_steps',
                                                 self.warmup_steps + 5000))

        self.total_it = 0

        # ---- Reward critics (per-action output) ----------------------------
        self.q_funcs = DiscreteDoubleQFunc(
            config['state_dim'], self.action_dim,
            hidden_size=config['hidden_sizes']
        ).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # ---- Cost critic (per-action output) -------------------------------
        self.qc_func = DiscreteSingleQFunc(
            config['state_dim'], self.action_dim,
            hidden_size=config['hidden_sizes']
        ).to(self.device)
        self.target_qc_func = copy.deepcopy(self.qc_func)
        self.target_qc_func.eval()
        for p in self.target_qc_func.parameters():
            p.requires_grad = False

        # ---- Cost network (still takes one-hot action) ---------------------
        self.cost_net = CostNetwork(
            config['state_dim'], self.action_dim,
            hidden_size=config['hidden_sizes']
        ).to(self.device)

        # ---- Policy --------------------------------------------------------
        self.policy = DiscretePolicy(
            config['state_dim'], self.action_dim,
            hidden_size=config['hidden_sizes']
        ).to(self.device)

        # ---- Temperature (log α) ------------------------------------------
        if config.get('temperature_opt', False):
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(
                torch.FloatTensor([config.get('alpha', 0.2)])
            ).to(self.device)

        # ---- Classifiers ---------------------------------------------------
        self.classifier = DARCClassifier(
            config['state_dim'], self.action_dim,
            hidden_size=config['hidden_sizes'],
            gaussian_noise_std=config.get('gaussian_noise_std', 1.0)
        ).to(self.device)

        # ---- Dual variable (λ ≥ 0) ----------------------------------------
        self.lambda_val = torch.zeros(1, device=self.device)

        # ---- Optimizers ----------------------------------------------------
        self.q_optimizer = torch.optim.Adam(
            self.q_funcs.parameters(), lr=config['critic_lr'])
        self.qc_optimizer = torch.optim.Adam(
            self.qc_func.parameters(), lr=config['critic_lr'])
        self.cost_optimizer = torch.optim.Adam(
            self.cost_net.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=config['actor_lr'])

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, test=True):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).view(1, -1).to(self.device)
            if test:
                action = self.policy.greedy(state_t)
            else:
                action, _ = self.policy.sample(state_t)
        return int(action.item())

    def _to_onehot(self, action):
        """Buffer action → one-hot (B, action_dim)."""
        return _action_to_onehot(action, self.action_dim)

    def _to_index(self, action):
        """Buffer action → integer index (B,)."""
        return _action_to_index(action, self.action_dim)

    # -----------------------------------------------------------------------
    # Expected value under current policy  (used in Bellman targets & actor)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _expected_value_reward(self, state, target=True):
        """
        V^r(s) = Σ_a π(a|s) [min(Q1,Q2)(s,a) − α log π(a|s)]
        Returns (B, 1).
        """
        _, probs, log_probs = self.policy(state)        # (B, A)
        if target:
            q1, q2 = self.target_q_funcs(state)
        else:
            q1, q2 = self.q_funcs(state)
        q_min = torch.min(q1, q2)                       # (B, A)
        v = (probs * (q_min - self.alpha * log_probs)).sum(dim=-1, keepdim=True)
        return v

    @torch.no_grad()
    def _expected_value_cost(self, state, target=True):
        """
        V^c(s) = Σ_a π(a|s) Q^c(s,a)       (no entropy term for cost)
        Returns (B, 1).
        """
        _, probs, _ = self.policy(state)
        if target:
            qc = self.target_qc_func(state)
        else:
            qc = self.qc_func(state)
        v = (probs * qc).sum(dim=-1, keepdim=True)
        return v

    # -----------------------------------------------------------------------
    # Step 2 – Classifier update
    # -----------------------------------------------------------------------

    def update_classifier(self, src_buf, tar_buf, batch_size, writer=None):
        src_s, src_a, src_ns, _, _, _ = src_buf.sample(batch_size)
        tar_s, tar_a, tar_ns, _, _, _ = tar_buf.sample(batch_size)

        src_a_oh = self._to_onehot(src_a)
        tar_a_oh = self._to_onehot(tar_a)

        state  = torch.cat([src_s,    tar_s],    0)
        action = torch.cat([src_a_oh, tar_a_oh], 0)
        ns     = torch.cat([src_ns,   tar_ns],   0)

        label = torch.cat([
            torch.zeros(batch_size, 1),
            torch.ones(batch_size,  1)
        ], dim=0).to(self.device)

        idx = torch.randperm(label.shape[0])
        state, action, ns, label = state[idx], action[idx], ns[idx], label[idx]

        sa_logits  = self.classifier.forward_sa(state, action, with_noise=True)
        sas_logits = self.classifier.forward_sas(state, action, ns, with_noise=True)

        loss_sa  = F.binary_cross_entropy_with_logits(sa_logits,  label)
        loss_sas = F.binary_cross_entropy_with_logits(sas_logits, label)
        cls_loss = loss_sa + loss_sas

        self.classifier_optimizer.zero_grad()
        cls_loss.backward()
        self.classifier_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/cls_sa_loss',  loss_sa.item(),  self.total_it)
            writer.add_scalar('train/cls_sas_loss', loss_sas.item(), self.total_it)

    # -----------------------------------------------------------------------
    # Step 4 – Importance weights
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _importance_weights(self, state, action, next_state):
        action_oh = self._to_onehot(action)
        sa_logit  = self.classifier.forward_sa(state, action_oh)
        sas_logit = self.classifier.forward_sas(state, action_oh, next_state)
        log_w = sas_logit - sa_logit
        log_w = torch.clamp(log_w, -self.weight_clip, self.weight_clip)
        return log_w.exp()                                  # (B, 1)

    # -----------------------------------------------------------------------
    # Phase 1 – Cost network update  (Step 5)
    # -----------------------------------------------------------------------

    def _update_cost_network(self, state, action, weights):
        with torch.no_grad():
            c_target = torch.clamp(0.5 * (weights - 1.0).abs(), max=1.0)

        action_oh = self._to_onehot(action)
        c_pred = self.cost_net(state, action_oh)
        cost_loss = F.mse_loss(c_pred, c_target)

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()
        return cost_loss.item()

    # -----------------------------------------------------------------------
    # Phase 2a – Reward critic update  (Step 6)
    # -----------------------------------------------------------------------

    def _update_reward_critics(self, state, action, reward, next_state,
                                not_done, writer=None):
        """
        Discrete Bellman backup for twin reward critics.
        Target: y_r = r + γ (1−d) V^r_target(s')
        where V^r(s') = Σ_a π(a|s')[min(Q1_targ, Q2_targ)(s',a) − α log π(a|s')]
        """
        with torch.no_grad():
            v_next = self._expected_value_reward(next_state, target=True)
            y_r = reward + not_done * self.discount * v_next

        action_idx = self._to_index(action)                  # (B,)
        q1_all, q2_all = self.q_funcs(state)                 # (B, A), (B, A)
        q1 = q1_all.gather(1, action_idx.unsqueeze(-1))      # (B, 1)
        q2 = q2_all.gather(1, action_idx.unsqueeze(-1))      # (B, 1)

        loss_r = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)

        self.q_optimizer.zero_grad()
        loss_r.backward()
        self.q_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_r1', q1.mean().item(), self.total_it)
            writer.add_scalar('train/reward_target', y_r.mean().item(), self.total_it)

        return loss_r.item()

    # -----------------------------------------------------------------------
    # Phase 2b – Cost critic update  (Step 7)
    # -----------------------------------------------------------------------

    def _update_cost_critic(self, state, action, next_state, not_done,
                             writer=None):
        """
        Cost Bellman backup (no entropy).
        Target: y_c = c_φ(s,a) + γ (1−d) V^c_target(s')
        where V^c(s') = Σ_a π(a|s') Q^c_target(s', a)
        """
        action_oh  = self._to_onehot(action)
        action_idx = self._to_index(action)

        with torch.no_grad():
            c_step   = self.cost_net(state, action_oh)       # (B, 1)
            v_c_next = self._expected_value_cost(next_state, target=True)
            y_c = c_step + not_done * self.discount * v_c_next

        qc_all = self.qc_func(state)                         # (B, A)
        qc = qc_all.gather(1, action_idx.unsqueeze(-1))      # (B, 1)
        loss_c = F.mse_loss(qc, y_c)

        self.qc_optimizer.zero_grad()
        loss_c.backward()
        self.qc_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_c', qc.mean().item(), self.total_it)

        return loss_c.item()

    # -----------------------------------------------------------------------
    # Phase 2c – Actor update  (Step 8)
    # -----------------------------------------------------------------------

    def _update_actor(self, state, writer=None):
        """
        Fully differentiable expected-value policy loss:
          L = − Σ_a π(a|s) [Q^r(s,a) − λ Q^c(s,a) − α log π(a|s)]

        Gradients flow through π(a|s) and log π(a|s) to the policy logits.
        """
        _, probs, log_probs = self.policy(state)            # (B, A)

        # Freeze critics so policy gradients don't leak into them
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        for p in self.qc_func.parameters():
            p.requires_grad = False

        q1, q2 = self.q_funcs(state)                        # (B, A)
        q_r = torch.min(q1, q2)                              # (B, A)
        q_c = self.qc_func(state)                            # (B, A)

        for p in self.q_funcs.parameters():
            p.requires_grad = True
        for p in self.qc_func.parameters():
            p.requires_grad = True

        lam = self.lambda_val.detach()

        inside = q_r - lam * q_c - self.alpha * log_probs   # (B, A)
        policy_loss = -(probs * inside).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature update
        if self.config.get('temperature_opt', False):
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)  # (B,)
            temp_loss = (self.alpha * (entropy - self.target_entropy)).mean()
            # temp_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/policy_loss', policy_loss.item(), self.total_it)
            writer.add_scalar('train/alpha',       self.alpha.item(),  self.total_it)
            writer.add_scalar('train/lambda',      lam.item(),        self.total_it)

        return policy_loss.item()

    # -----------------------------------------------------------------------
    # Phase 3 – Dual update  (Step 9)
    # -----------------------------------------------------------------------

    def _update_lambda(self, state, writer=None):
        """
        Ĵ_c = Σ_a π(a|s) Q^c(s,a)   (expected cost value)
        λ ← [λ + η_λ (Ĵ_c − ε)]_+

        Skipped until dual_warmup_steps so the cost critic stabilises
        before λ starts accumulating.
        """
        if self.total_it <= self.dual_warmup_steps:
            return

        with torch.no_grad():
            j_c = self._expected_value_cost(state, target=False).mean()

        self.lambda_val = torch.clamp(
            self.lambda_val + self.dual_step_size * (j_c - self.constraint_budget),
            min=0.0
        )

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/j_c',    j_c.item(),              self.total_it)
            writer.add_scalar('train/lambda',  self.lambda_val.item(), self.total_it)

    # -----------------------------------------------------------------------
    # Step 10 – Target network soft update
    # -----------------------------------------------------------------------

    def update_target(self):
        with torch.no_grad():
            for tp, p in zip(self.target_q_funcs.parameters(),
                             self.q_funcs.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(self.target_qc_func.parameters(),
                             self.qc_func.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # -----------------------------------------------------------------------
    # Main training step
    # -----------------------------------------------------------------------

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=256,
              writer=None):
        self.total_it += 1

        if src_replay_buffer.size < 2 * batch_size:
            return

        in_warmup = self.total_it <= self.warmup_steps

        # ------------------------------------------------------------------
        # WARMUP – pure discrete SAC on reward critic + actor
        # ------------------------------------------------------------------
        if in_warmup:
            src_s, src_a, src_ns, src_r, src_not_done, _ = \
                src_replay_buffer.sample(batch_size)

            self._update_reward_critics(src_s, src_a, src_r, src_ns,
                                        src_not_done, writer=writer)
            self._update_actor(src_s, writer)

            # Soft-update reward target only
            with torch.no_grad():
                for tp, p in zip(self.target_q_funcs.parameters(),
                                 self.q_funcs.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

            if writer is not None and self.total_it % 5000 == 0:
                writer.add_scalar('train/warmup_step', self.total_it, self.total_it)
            return

        # ------------------------------------------------------------------
        # FULL C-VWDA (post-warmup)
        # ------------------------------------------------------------------
        if tar_replay_buffer.size < batch_size:
            return

        # Step 2 – Classifier update
        if self.total_it % self.cls_update_freq == 0:
            self.update_classifier(src_replay_buffer, tar_replay_buffer,
                                   batch_size, writer)

        # Step 3 – Sample from source
        (src_s, src_a, src_ns, src_r,
         src_not_done, _) = src_replay_buffer.sample(batch_size)

        # Step 4 – Importance weights
        weights = self._importance_weights(src_s, src_a, src_ns)

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/iw_mean', weights.mean().item(), self.total_it)
            writer.add_scalar('train/iw_std',  weights.std().item(),  self.total_it)

        # Phase 1 – Cost network
        self._update_cost_network(src_s, src_a, weights)

        # Phase 2
        # 6. Reward critics
        self._update_reward_critics(src_s, src_a, src_r, src_ns,
                                    src_not_done, writer=writer)

        # 7. Cost critic
        self._update_cost_critic(src_s, src_a, src_ns, src_not_done, writer)

        # 8. Actor
        self._update_actor(src_s, writer)

        # Phase 3 – Dual update (gated by dual_warmup_steps)
        self._update_lambda(src_s, writer)

        # Step 10 – Soft-update targets
        self.update_target()

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(),          filename + "_qr")
        torch.save(self.q_optimizer.state_dict(),      filename + "_qr_opt")
        torch.save(self.qc_func.state_dict(),          filename + "_qc")
        torch.save(self.qc_optimizer.state_dict(),     filename + "_qc_opt")
        torch.save(self.cost_net.state_dict(),         filename + "_cost")
        torch.save(self.cost_optimizer.state_dict(),   filename + "_cost_opt")
        torch.save(self.policy.state_dict(),           filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_opt")
        torch.save(self.classifier.state_dict(),       filename + "_cls")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_cls_opt")
        torch.save({'lambda': self.lambda_val},        filename + "_dual")

    def load(self, filename):
        self.q_funcs.load_state_dict(
            torch.load(filename + "_qr", weights_only=True))
        self.q_optimizer.load_state_dict(
            torch.load(filename + "_qr_opt", weights_only=True))
        self.qc_func.load_state_dict(
            torch.load(filename + "_qc", weights_only=True))
        self.qc_optimizer.load_state_dict(
            torch.load(filename + "_qc_opt", weights_only=True))
        self.cost_net.load_state_dict(
            torch.load(filename + "_cost", weights_only=True))
        self.cost_optimizer.load_state_dict(
            torch.load(filename + "_cost_opt", weights_only=True))
        self.policy.load_state_dict(
            torch.load(filename + "_actor", weights_only=True))
        self.policy_optimizer.load_state_dict(
            torch.load(filename + "_actor_opt", weights_only=True))
        self.classifier.load_state_dict(
            torch.load(filename + "_cls", weights_only=True))
        self.classifier_optimizer.load_state_dict(
            torch.load(filename + "_cls_opt", weights_only=True))
        dual = torch.load(filename + "_dual", weights_only=False)
        self.lambda_val = dual['lambda'].to(self.device)
