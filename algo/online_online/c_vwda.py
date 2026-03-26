"""
C-VWDA: Constrained Value-Weighted Domain Adaptation

Algorithm outline
-----------------
Networks
  - Policy π_θ
  - Twin reward critics Q^r_{ψ1}, Q^r_{ψ2} + target networks
  - Cost critic  Q^c_ξ                      + target network
  - Cost network c_φ(s,a) → [0,1]  (sigmoid output, learned via regression)
  - SA  classifier  q^sa_ω(s,a)    (DARC-style binary classifier)
  - SAS classifier  q^sas_ω(s,a,s') (DARC-style binary classifier)

Per-step training loop
  1. Collect: store (s,a,r,s') in B_src
  2. Every K_cls steps: update SA / SAS classifiers via BCE on balanced
     source/target batch.
  3. Sample mini-batch from B_src.
  4. Compute importance weights:
       log w_i = logit(q^sas(s,a,s')) - logit(q^sa(s,a))
       w_i     = exp(clip(log_w, -W, W))
  Phase 1 – Cost learning (independent of λ)
  5. Regress c_φ toward (1/2)|w_i - 1| (clamped to [0,1])
  Phase 2 – Primal step
  6. Update twin reward critics Q^r (standard SAC Bellman)
  7. Update cost critic  Q^c (cost Bellman, no entropy bonus)
  8. Update policy: maximize  Q^r - λ Q^c - α log π
  Phase 3 – Dual step
  9. λ ← [λ + η_λ (J^c - ε)]_+
  10. Soft-update target networks
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


# ---------------------------------------------------------------------------
# Shared network building blocks (mirror DARC style)
# ---------------------------------------------------------------------------

class TanhTransform(Transform):
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
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


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


class Policy(nn.Module):
    """Squashed Gaussian policy (same as DARC)."""

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        action = dist.rsample()
        logprob = dist.log_prob(action).sum(axis=-1, keepdim=True) if get_logprob else None
        mean = torch.tanh(mu)
        return action * self.max_action, logprob, mean * self.max_action


class DiscretePolicy(nn.Module):
    """Categorical policy for discrete action spaces (e.g. Gridworld)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)

    def forward(self, x, get_logprob=False):
        logits = self.network(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()                         # (B,)
        logprob = dist.log_prob(action).unsqueeze(-1) if get_logprob else None
        mean = torch.argmax(logits, dim=-1)            # deterministic greedy
        # Return float tensors to match continuous API (B,1) / (B,) shapes
        return action.float().unsqueeze(-1), logprob, mean.float().unsqueeze(-1)


class DoubleQFunc(nn.Module):
    """Twin Q-networks (for reward critic)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SingleQFunc(nn.Module):
    """Single Q-network (for cost critic)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network(x)


class CostNetwork(nn.Module):
    """
    c_φ(s,a) → [0,1].
    Learned by regressing to (1/2)|w-1| (capped at 1), where w are the
    importance weights derived from the classifiers.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return torch.sigmoid(self.network(x))


class DARCClassifier(nn.Module):
    """
    Binary SA / SAS classifiers, identical to DARC's Classifier but
    outputting a single logit (BCEWithLogitsLoss) rather than 2-class
    softmax, which enables the logit trick for importance weights.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256,
                 gaussian_noise_std=1.0):
        super().__init__()
        self.gaussian_noise_std = gaussian_noise_std
        # Output single logit → sigmoid → Bernoulli probability of being
        # from the *target* domain (label 1)
        self.sa_net  = MLPNetwork(state_dim + action_dim,         1, hidden_size)
        self.sas_net = MLPNetwork(2 * state_dim + action_dim,     1, hidden_size)

    def forward_sa(self, state, action, with_noise=False):
        x = torch.cat([state, action], dim=-1)
        if with_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.sa_net(x)                 # raw logit

    def forward_sas(self, state, action, next_state, with_noise=False):
        x = torch.cat([state, action, next_state], dim=-1)
        if with_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.sas_net(x)               # raw logit


# ---------------------------------------------------------------------------
# C-VWDA main class
# ---------------------------------------------------------------------------

class C_VWDA(object):
    """
    Constrained Value-Weighted Domain Adaptation (C-VWDA).

    Required config keys
    --------------------
    state_dim, action_dim, max_action, hidden_sizes,
    actor_lr, critic_lr, gamma, tau,
    temperature_opt (bool), update_interval,
    tar_env_interact_freq (classifier update period K_cls),
    gaussian_noise_std,
    weight_clip  (W)        – default 10.0,
    constraint_budget  (ε)  – default 1.0,
    dual_step_size     (η_λ) – default 3e-4,
    warmup_steps            – default 10 000.
      During warmup the algorithm runs as pure SAC (reward critics + actor
      only).  Classifiers, cost network, cost critic, and λ are frozen.
      This mirrors DARC's 1e5-step warmup and ensures classifiers have
      reasonable outputs before importance weights are computed.
    """

    def __init__(self, config, device, target_entropy=None):
        self.config  = config
        self.device  = device
        self.discount = config['gamma']
        self.tau      = config['tau']
        self.target_entropy = (target_entropy if target_entropy is not None
                               else -config['action_dim'])
        self.update_interval = config.get('update_interval', 2)

        # Hyperparameters specific to C-VWDA
        self.weight_clip       = config.get('weight_clip', 2.0)
        self.constraint_budget = config.get('constraint_budget', 0.99)
        self.dual_step_size    = config.get('dual_step_size', 3e-4)
        self.cls_update_freq   = config.get('tar_env_interact_freq', 10)
        # Warmup: run as pure SAC for this many steps before enabling
        # classifiers / cost / dual update (mirrors DARC's warmup).
        self.warmup_steps      = int(config.get('warmup_steps', 1e4))

        self.total_it = 0

        # ---- Reward critics ------------------------------------------------
        self.q_funcs = DoubleQFunc(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # ---- Cost critic ---------------------------------------------------
        self.qc_func = SingleQFunc(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(self.device)
        self.target_qc_func = copy.deepcopy(self.qc_func)
        self.target_qc_func.eval()
        for p in self.target_qc_func.parameters():
            p.requires_grad = False

        # ---- Cost network --------------------------------------------------
        self.cost_net = CostNetwork(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(self.device)

        # ---- Policy --------------------------------------------------------
        if config.get('discrete_action', False):
            self.policy = DiscretePolicy(
                config['state_dim'], config['action_dim'],
                hidden_size=config['hidden_sizes']
            ).to(self.device)
        else:
            self.policy = Policy(
                config['state_dim'], config['action_dim'], config['max_action'],
                hidden_size=config['hidden_sizes']
            ).to(self.device)
        self.discrete_action = config.get('discrete_action', False)

        # ---- Temperature ---------------------------------------------------
        if config.get('temperature_opt', False):
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(
                torch.FloatTensor([config.get('alpha', 0.2)])
            ).to(self.device)

        # ---- Classifiers ---------------------------------------------------
        self.classifier = DARCClassifier(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes'],
            gaussian_noise_std=config.get('gaussian_noise_std', 1.0)
        ).to(self.device)

        # ---- Dual variable (Lagrange multiplier) ---------------------------
        # λ ≥ 0, projected to non-negative after each gradient-ascent step.
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
            action, _, mean = self.policy(
                torch.FloatTensor(state).view(1, -1).to(self.device))
        if test:
            out = mean.squeeze().cpu().numpy()
        else:
            out = action.squeeze().cpu().numpy()
        # For discrete envs, return an int scalar
        if self.discrete_action:
            return int(round(float(out)))
        return out

    # -----------------------------------------------------------------------
    # Internal helper: convert policy action to Q-net compatible tensor
    # -----------------------------------------------------------------------

    def _policy_action_for_qnet(self, action):
        """
        Continuous: returns action as-is  (B, action_dim).
        Discrete, two callers with different shapes:
          - Policy output (B, 1)         → integer index → one-hot (B, action_dim)
          - Buffer action (B, action_dim) → already one-hot (stored that way) → passthrough
        """
        if self.discrete_action:
            if action.shape[-1] == 1:
                # Policy output: integer index → one-hot
                idx = action.long().squeeze(-1)           # (B,)
                return F.one_hot(idx, num_classes=self.config['action_dim']).float()
            else:
                # Buffer action: already stored as one-hot
                return action.float()
        return action

    # -----------------------------------------------------------------------
    # Step 2 – Classifier update (every K_cls steps)
    # -----------------------------------------------------------------------

    def update_classifier(self, src_buf, tar_buf, batch_size, writer=None):
        """
        Binary BCE on balanced source (y=0) / target (y=1) batches.
        Mirrors DARC's update_classifier but uses single-logit BCE.

        For discrete action spaces the stored buffer action is a scalar index
        (B,1); we convert it to one-hot so the classifier input size matches
        its expected state_dim + action_dim.
        """
        src_s, src_a, src_ns, _, _, _ = src_buf.sample(batch_size)
        tar_s, tar_a, tar_ns, _, _, _ = tar_buf.sample(batch_size)

        # Convert actions to Q-net / classifier compatible form
        src_a_q = self._policy_action_for_qnet(src_a)
        tar_a_q = self._policy_action_for_qnet(tar_a)

        state  = torch.cat([src_s,   tar_s],   0)
        action = torch.cat([src_a_q, tar_a_q], 0)
        ns     = torch.cat([src_ns,  tar_ns],  0)

        # labels: 0 = source, 1 = target
        label = torch.cat([
            torch.zeros(batch_size, 1),
            torch.ones(batch_size,  1)
        ], dim=0).to(self.device)

        # shuffle
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
            writer.add_scalar('train/cls_sa_loss',  loss_sa,  self.total_it)
            writer.add_scalar('train/cls_sas_loss', loss_sas, self.total_it)

    # -----------------------------------------------------------------------
    # Step 4 – Importance weights
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _importance_weights(self, state, action, next_state):
        """
        log w_i = logit(q^sas(s,a,s')) - logit(q^sa(s,a))
        w_i     = exp(clip(log_w, -W, W))

        For discrete envs the buffer action is a scalar index; convert to
        one-hot so classifier input dimensions match.
        """
        action_q = self._policy_action_for_qnet(action)
        sa_logit  = self.classifier.forward_sa(state, action_q, with_noise=False)
        sas_logit = self.classifier.forward_sas(state, action_q, next_state,
                                                 with_noise=False)
        log_w = sas_logit - sa_logit
        log_w = torch.clamp(log_w, -self.weight_clip, self.weight_clip)
        w = log_w.exp()
        return w  # shape (B, 1)

    # -----------------------------------------------------------------------
    # Phase 1 – Cost network update  (Step 5)
    # -----------------------------------------------------------------------

    def _update_cost_network(self, state, action, weights):
        """
        Target: c^target_i = min(0.5 * |w_i - 1|, 1)
        Loss:   MSE(c_φ(s,a), c^target)

        CostNetwork expects concatenated (state, action) where action has
        the same dimensionality as action_dim (one-hot for discrete).
        """
        with torch.no_grad():
            c_target = torch.clamp(0.5 * (weights - 1.0).abs(), max=1.0)

        action_q = self._policy_action_for_qnet(action)
        c_pred = self.cost_net(state, action_q)
        cost_loss = F.mse_loss(c_pred, c_target)

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()
        return cost_loss.item()

    # -----------------------------------------------------------------------
    # Phase 2a – Reward critic update  (Step 6)
    # -----------------------------------------------------------------------

    def _update_reward_critics(self, state, action, reward, next_state,
                                not_done, next_action_q=None, next_logprob=None,
                                writer=None):
        """
        Standard SAC double-Q Bellman backup (pure reward, no λ or c).

        If `next_action_q` and `next_logprob` are provided (pre-sampled in the
        train loop), they are reused so that steps 6 and 7 share the same ã
        sample as specified by the pseudocode.  Otherwise they are sampled here
        (used during warmup where step 7 does not run).
        """
        with torch.no_grad():
            if next_action_q is None:
                next_action_raw, next_logprob, _ = self.policy(next_state,
                                                               get_logprob=True)
                next_action_q = self._policy_action_for_qnet(next_action_raw)
            q_t1, q_t2 = self.target_q_funcs(next_state, next_action_q)
            v_next = torch.min(q_t1, q_t2) - self.alpha * next_logprob
            y_r = reward + not_done * self.discount * v_next

        action_q = self._policy_action_for_qnet(action)
        q1, q2 = self.q_funcs(state, action_q)
        loss_r = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)

        self.q_optimizer.zero_grad()
        loss_r.backward()
        self.q_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_r1', q1.mean(), self.total_it)
            writer.add_scalar('train/reward_target', y_r.mean(), self.total_it)

        return loss_r.item()

    # -----------------------------------------------------------------------
    # Phase 2b – Cost critic update  (Step 7)
    # -----------------------------------------------------------------------

    def _update_cost_critic(self, state, action, next_state, not_done,
                             next_action_q, writer=None):
        """
        Cost Bellman backup (no entropy bonus — cost is not a reward).
          y^c_i = c_φ(s,a) + γ * Q^c_target(s', ã)
        `next_action_q` must already be Q-net compatible (one-hot for discrete).
        c_φ also receives the Q-net-compatible action form.
        """
        action_q = self._policy_action_for_qnet(action)
        with torch.no_grad():
            c_step   = self.cost_net(state, action_q)          # ← one-hot for discrete
            v_c_next = self.target_qc_func(next_state, next_action_q)
            y_c = c_step + not_done * self.discount * v_c_next

        qc = self.qc_func(state, action_q)
        loss_c = F.mse_loss(qc, y_c)

        self.qc_optimizer.zero_grad()
        loss_c.backward()
        self.qc_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_c', qc.mean(), self.total_it)

        return loss_c.item()

    # -----------------------------------------------------------------------
    # Phase 2c – Actor update  (Step 8)
    # -----------------------------------------------------------------------

    def _update_actor(self, state, writer=None):
        """
        ∇_θ E[ Q^r(s,â) - λ Q^c(s,â) - α log π(â|s) ]
        Works for both continuous (squashed Gaussian) and discrete (Categorical) policies.
        """
        action, logprob, _ = self.policy(state, get_logprob=True)

        if self.discrete_action:
            # action is a float index (B,1); Q-nets expect (B, action_dim)-style
            # input but our SingleQFunc / DoubleQFunc take raw action vectors.
            # Convert integer index → one-hot for Q-net input.
            action_idx = action.long().squeeze(-1)          # (B,)
            action_onehot = F.one_hot(
                action_idx, num_classes=self.config['action_dim']
            ).float()                                       # (B, action_dim)
            q_r1, q_r2 = self.q_funcs(state, action_onehot)
            q_c         = self.qc_func(state, action_onehot)
        else:
            q_r1, q_r2 = self.q_funcs(state, action)
            q_c         = self.qc_func(state, action)

        q_r = torch.min(q_r1, q_r2)
        lam = self.lambda_val.detach()
        policy_loss = (self.alpha * logprob - q_r + lam * q_c).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature update (only meaningful for continuous / entropy-based)
        if self.config.get('temperature_opt', False):
            temp_loss = -(self.alpha * (logprob.detach() + self.target_entropy)).mean()
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/policy_loss',  policy_loss, self.total_it)
            writer.add_scalar('train/alpha',        self.alpha,  self.total_it)
            writer.add_scalar('train/lambda',       lam,         self.total_it)

        # Return actor actions in Q-net-compatible form for downstream use
        if self.discrete_action:
            return policy_loss.item(), action_onehot.detach(), logprob.detach()
        return policy_loss.item(), action.detach(), logprob.detach()

    # -----------------------------------------------------------------------
    # Phase 3 – Dual update  (Step 9)
    # -----------------------------------------------------------------------

    def _update_lambda(self, state, actor_action, writer=None):
        """
        Ĵ_c = (1/B) Σ Q^c(s, â)
        λ ← [λ + η_λ (Ĵ_c − ε)]_+
        """
        with torch.no_grad():
            j_c = self.qc_func(state, actor_action).mean()

        self.lambda_val = torch.clamp(
            self.lambda_val + self.dual_step_size * (j_c - self.constraint_budget),
            min=0.0
        )

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/j_c',     j_c,            self.total_it)
            writer.add_scalar('train/lambda',   self.lambda_val, self.total_it)

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
        """
        Matches the interface used by run_train.py:
          policy.train(src_replay_buffer, tar_replay_buffer, batch_size, writer)

        Both buffers are utils.ReplayBuffer instances whose .sample() returns
          (state, action, next_state, reward, not_done, cost).

        The target buffer (D_target) is used ONLY for classifier training;
        all Bellman / actor updates run on source data.

        Warmup phase (first `warmup_steps` steps)
        ------------------------------------------
        Identical in spirit to DARC's 1e5-step warmup:
          - Only reward critics and actor are updated (pure SAC, λ=0).
          - Classifiers, cost network, cost critic, and dual variable are
            frozen so that importance weights are not computed from an
            untrained classifier.
        After warmup the full C-VWDA loop runs every step.
        """
        self.total_it += 1

        # Wait until source buffer has enough data.
        # During warmup we don't need the target buffer.
        if src_replay_buffer.size < 2 * batch_size:
            return

        in_warmup = self.total_it <= self.warmup_steps

        # ------------------------------------------------------------------
        # WARMUP PATH – pure SAC on reward critic + actor only
        # ------------------------------------------------------------------
        if in_warmup:
            src_s, src_a, src_ns, src_r, src_not_done, _ = \
                src_replay_buffer.sample(batch_size)

            # Reward critics (standard SAC Bellman, no cost)
            self._update_reward_critics(src_s, src_a, src_r, src_ns,
                                        src_not_done, writer=writer)

            # Actor with λ=0; freeze both Q networks (consistent with post-warmup)
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            for p in self.qc_func.parameters():
                p.requires_grad = False
            self._update_actor(src_s, writer)
            for p in self.q_funcs.parameters():
                p.requires_grad = True
            for p in self.qc_func.parameters():
                p.requires_grad = True

            # Soft-update reward target only
            with torch.no_grad():
                for tp, p in zip(self.target_q_funcs.parameters(),
                                 self.q_funcs.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

            if writer is not None and self.total_it % 5000 == 0:
                writer.add_scalar('train/warmup_step', self.total_it, self.total_it)
            return

        # ------------------------------------------------------------------
        # FULL C-VWDA PATH  (post-warmup)
        # ------------------------------------------------------------------

        # Target buffer must be populated before classifier training starts.
        if tar_replay_buffer.size < batch_size:
            return

        # Step 2 – Classifier update (every K_cls steps)
        if self.total_it % self.cls_update_freq == 0:
            self.update_classifier(src_replay_buffer, tar_replay_buffer,
                                   batch_size, writer)

        # Step 3 – Sample mini-batch from source buffer
        (src_s, src_a, src_ns, src_r,
         src_not_done, _) = src_replay_buffer.sample(batch_size)

        # Step 4 – Importance weights
        weights = self._importance_weights(src_s, src_a, src_ns)

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/iw_mean', weights.mean(), self.total_it)
            writer.add_scalar('train/iw_std',  weights.std(),  self.total_it)

        # Phase 1 – Cost network update
        self._update_cost_network(src_s, src_a, weights)

        # Phase 2 – Primal step

        # Pre-sample ã once (shared by steps 6 and 7 per pseudocode)
        with torch.no_grad():
            next_action_raw, next_logprob, _ = self.policy(src_ns, get_logprob=True)
            next_action_q = self._policy_action_for_qnet(next_action_raw)

        # 6. Reward critics (reuse shared ã)
        self._update_reward_critics(src_s, src_a, src_r, src_ns,
                                    src_not_done,
                                    next_action_q=next_action_q,
                                    next_logprob=next_logprob,
                                    writer=writer)

        # 7. Cost critic (reuse same ã)
        self._update_cost_critic(src_s, src_a, src_ns, src_not_done,
                                 next_action_q, writer)

        # 8. Actor  (freeze Q grads during policy backward, same as DARC)
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        for p in self.qc_func.parameters():
            p.requires_grad = False

        _, actor_action, _ = self._update_actor(src_s, writer)

        for p in self.q_funcs.parameters():
            p.requires_grad = True
        for p in self.qc_func.parameters():
            p.requires_grad = True

        # Phase 3 – Dual update
        self._update_lambda(src_s, actor_action, writer)

        # Step 10 – Soft-update targets (both reward and cost)
        self.update_target()

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(),        filename + "_qr")
        torch.save(self.q_optimizer.state_dict(),    filename + "_qr_opt")
        torch.save(self.qc_func.state_dict(),        filename + "_qc")
        torch.save(self.qc_optimizer.state_dict(),   filename + "_qc_opt")
        torch.save(self.cost_net.state_dict(),       filename + "_cost")
        torch.save(self.cost_optimizer.state_dict(), filename + "_cost_opt")
        torch.save(self.policy.state_dict(),         filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_opt")
        torch.save(self.classifier.state_dict(),     filename + "_cls")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_cls_opt")
        torch.save({'lambda': self.lambda_val},      filename + "_dual")

    def load(self, filename):
        self.q_funcs.load_state_dict(
            torch.load(filename + "_qr"))
        self.q_optimizer.load_state_dict(
            torch.load(filename + "_qr_opt"))
        self.qc_func.load_state_dict(
            torch.load(filename + "_qc"))
        self.qc_optimizer.load_state_dict(
            torch.load(filename + "_qc_opt"))
        self.cost_net.load_state_dict(
            torch.load(filename + "_cost"))
        self.cost_optimizer.load_state_dict(
            torch.load(filename + "_cost_opt"))
        self.policy.load_state_dict(
            torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(
            torch.load(filename + "_actor_opt"))
        self.classifier.load_state_dict(
            torch.load(filename + "_cls"))
        self.classifier_optimizer.load_state_dict(
            torch.load(filename + "_cls_opt"))
        dual = torch.load(filename + "_dual")
        self.lambda_val = dual['lambda'].to(self.device)
