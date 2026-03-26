"""
C-VWDA (Continuous): Constrained Value-Weighted Domain Adaptation
-- Continuous action spaces only (MuJoCo etc.) --

Algorithm outline
-----------------
Networks
  - Policy π_θ  (squashed Gaussian)
  - Twin reward critics Q^r_{ψ1}, Q^r_{ψ2} + target networks
  - Cost critic  Q^c_ξ                      + target network
  - Cost network c_φ(s,a) → [0,1]  (sigmoid output, learned via regression)
  - SA  classifier  q^sa_ω(s,a)
  - SAS classifier  q^sas_ω(s,a,s')

Per-step training loop
  1. Collect: store (s,a,r,s') in B_src
  2. Every K_cls steps: update SA / SAS classifiers via BCE
  3. Sample mini-batch from B_src.
  4. Importance weights:
       log w_i = logit(q^sas(s,a,s')) - logit(q^sa(s,a))
       w_i     = exp(clip(log_w, -W, W))
  Phase 1 – Cost learning
  5. Regress c_φ toward min(½|w_i-1|, 1)
  Phase 2 – Primal step
  6. Update twin reward critics Q^r (standard SAC Bellman)
  7. Update cost critic  Q^c (cost Bellman, no entropy)
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
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


# ---------------------------------------------------------------------------
# Network building blocks
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
    """Squashed Gaussian policy."""

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super().__init__()
        self.action_dim = action_dim
        self.max_action  = max_action
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


class DoubleQFunc(nn.Module):
    """Twin Q-networks (reward critic)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SingleQFunc(nn.Module):
    """Single Q-network (cost critic)."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network(x)


class CostNetwork(nn.Module):
    """c_φ(s,a) → [0,1], learned by regression toward ½|w-1|."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return torch.sigmoid(self.network(x))


class DARCClassifier(nn.Module):
    """
    Binary SA / SAS classifiers outputting a single logit
    (BCEWithLogitsLoss) for the logit-difference importance weight trick.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256,
                 gaussian_noise_std=1.0):
        super().__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_net  = MLPNetwork(state_dim + action_dim,     1, hidden_size)
        self.sas_net = MLPNetwork(2 * state_dim + action_dim, 1, hidden_size)

    def forward_sa(self, state, action, with_noise=False):
        x = torch.cat([state, action], dim=-1)
        if with_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.sa_net(x)

    def forward_sas(self, state, action, next_state, with_noise=False):
        x = torch.cat([state, action, next_state], dim=-1)
        if with_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.sas_net(x)


# ---------------------------------------------------------------------------
# C-VWDA (Continuous)
# ---------------------------------------------------------------------------

class C_VWDA_Continuous(object):
    """
    Constrained Value-Weighted Domain Adaptation — continuous action spaces.

    Required config keys
    --------------------
    state_dim, action_dim, max_action, hidden_sizes,
    actor_lr, critic_lr, gamma, tau,
    temperature_opt (bool), update_interval,
    tar_env_interact_freq (classifier update period K_cls),
    gaussian_noise_std,
    weight_clip  (W)        – default 2.0,
    constraint_budget  (ε)  – default 0.99,
    dual_step_size     (η_λ) – default 3e-4,
    warmup_steps            – default 10 000.
    """

    def __init__(self, config, device, target_entropy=None):
        self.config   = config
        self.device   = device
        self.discount = config['gamma']
        self.tau      = config['tau']
        self.target_entropy = (target_entropy if target_entropy is not None
                               else -config['action_dim'])
        self.update_interval = config.get('update_interval', 2)

        # C-VWDA hyperparameters
        self.weight_clip       = config.get('weight_clip', 2.0)
        self.constraint_budget = config.get('constraint_budget', 0.99)
        self.dual_step_size    = config.get('dual_step_size', 3e-4)
        self.cls_update_freq   = config.get('tar_env_interact_freq', 10)
        self.warmup_steps      = int(config.get('warmup_steps', 1e4))

        self.total_it = 0

        # ---- Reward critics ------------------------------------------------
        self.q_funcs = DoubleQFunc(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # ---- Cost critic ---------------------------------------------------
        self.qc_func = SingleQFunc(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(device)
        self.target_qc_func = copy.deepcopy(self.qc_func)
        self.target_qc_func.eval()
        for p in self.target_qc_func.parameters():
            p.requires_grad = False

        # ---- Cost network --------------------------------------------------
        self.cost_net = CostNetwork(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(device)

        # ---- Policy --------------------------------------------------------
        self.policy = Policy(
            config['state_dim'], config['action_dim'], config['max_action'],
            hidden_size=config['hidden_sizes']
        ).to(device)

        # ---- Temperature ---------------------------------------------------
        if config.get('temperature_opt', False):
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        else:
            self.log_alpha = torch.log(
                torch.FloatTensor([config.get('alpha', 0.2)])
            ).to(device)

        # ---- Classifiers ---------------------------------------------------
        self.classifier = DARCClassifier(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes'],
            gaussian_noise_std=config.get('gaussian_noise_std', 1.0)
        ).to(device)

        # ---- Dual variable -------------------------------------------------
        self.lambda_val = torch.zeros(1, device=device)

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
        return (mean if test else action).squeeze().cpu().numpy()

    # -----------------------------------------------------------------------
    # Step 2 – Classifier update
    # -----------------------------------------------------------------------

    def update_classifier(self, src_buf, tar_buf, batch_size, writer=None):
        src_s, src_a, src_ns, _, _, _ = src_buf.sample(batch_size)
        tar_s, tar_a, tar_ns, _, _, _ = tar_buf.sample(batch_size)

        state  = torch.cat([src_s,  tar_s],  0)
        action = torch.cat([src_a,  tar_a],  0)
        ns     = torch.cat([src_ns, tar_ns], 0)

        label = torch.cat([
            torch.zeros(batch_size, 1),
            torch.ones(batch_size,  1),
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
            writer.add_scalar('train/cls_sa_loss',  loss_sa,  self.total_it)
            writer.add_scalar('train/cls_sas_loss', loss_sas, self.total_it)

    # -----------------------------------------------------------------------
    # Step 4 – Importance weights
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _importance_weights(self, state, action, next_state):
        sa_logit  = self.classifier.forward_sa(state, action)
        sas_logit = self.classifier.forward_sas(state, action, next_state)
        log_w = torch.clamp(sas_logit - sa_logit,
                            -self.weight_clip, self.weight_clip)
        return log_w.exp()   # (B, 1)

    # -----------------------------------------------------------------------
    # Phase 1 – Cost network (Step 5)
    # -----------------------------------------------------------------------

    def _update_cost_network(self, state, action, weights):
        with torch.no_grad():
            c_target = torch.clamp(0.5 * (weights - 1.0).abs(), max=1.0)

        c_pred    = self.cost_net(state, action)
        cost_loss = F.mse_loss(c_pred, c_target)

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()
        return cost_loss.item()

    # -----------------------------------------------------------------------
    # Phase 2a – Reward critic (Step 6)
    # -----------------------------------------------------------------------

    def _update_reward_critics(self, state, action, reward, next_state,
                                not_done, next_action=None, next_logprob=None,
                                writer=None):
        """
        If next_action / next_logprob are given (pre-sampled, shared with
        cost critic per pseudocode), reuse them; otherwise sample here
        (warmup path where cost critic is not updated).
        """
        with torch.no_grad():
            if next_action is None:
                next_action, next_logprob, _ = self.policy(next_state,
                                                           get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(next_state, next_action)
            v_next = torch.min(q_t1, q_t2) - self.alpha * next_logprob
            y_r    = reward + not_done * self.discount * v_next

        q1, q2    = self.q_funcs(state, action)
        loss_r    = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)

        self.q_optimizer.zero_grad()
        loss_r.backward()
        self.q_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_r1',         q1.mean(),  self.total_it)
            writer.add_scalar('train/reward_target', y_r.mean(), self.total_it)

        return loss_r.item()

    # -----------------------------------------------------------------------
    # Phase 2b – Cost critic (Step 7)
    # -----------------------------------------------------------------------

    def _update_cost_critic(self, state, action, next_state, not_done,
                             next_action, writer=None):
        """next_action is the same sample used in step 6."""
        with torch.no_grad():
            c_step   = self.cost_net(state, action)
            v_c_next = self.target_qc_func(next_state, next_action)
            y_c      = c_step + not_done * self.discount * v_c_next

        qc     = self.qc_func(state, action)
        loss_c = F.mse_loss(qc, y_c)

        self.qc_optimizer.zero_grad()
        loss_c.backward()
        self.qc_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_c', qc.mean(), self.total_it)

        return loss_c.item()

    # -----------------------------------------------------------------------
    # Phase 2c – Actor (Step 8)
    # -----------------------------------------------------------------------

    def _update_actor(self, state, writer=None):
        """∇_θ E[ Q^r(s,â) - λ Q^c(s,â) - α log π(â|s) ]"""
        action, logprob, _ = self.policy(state, get_logprob=True)

        q_r1, q_r2 = self.q_funcs(state, action)
        q_c        = self.qc_func(state, action)
        q_r        = torch.min(q_r1, q_r2)
        lam        = self.lambda_val.detach()

        policy_loss = (self.alpha * logprob - q_r + lam * q_c).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.config.get('temperature_opt', False):
            temp_loss = -(self.alpha * (logprob.detach() + self.target_entropy)).mean()
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/policy_loss', policy_loss, self.total_it)
            writer.add_scalar('train/alpha',       self.alpha,  self.total_it)
            writer.add_scalar('train/lambda',      lam,         self.total_it)

        return policy_loss.item(), action.detach(), logprob.detach()

    # -----------------------------------------------------------------------
    # Phase 3 – Dual update (Step 9)
    # -----------------------------------------------------------------------

    def _update_lambda(self, state, actor_action, writer=None):
        with torch.no_grad():
            j_c = self.qc_func(state, actor_action).mean()

        self.lambda_val = torch.clamp(
            self.lambda_val + self.dual_step_size * (j_c - self.constraint_budget),
            min=0.0
        )

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/j_c',    j_c,            self.total_it)
            writer.add_scalar('train/lambda', self.lambda_val, self.total_it)

    # -----------------------------------------------------------------------
    # Step 10 – Target soft update
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
        Both buffers are utils.ReplayBuffer whose .sample() returns
          (state, action, next_state, reward, not_done, cost).

        Warmup (first warmup_steps): pure SAC on reward critics + actor only.
        Post-warmup: full C-VWDA loop.
        """
        self.total_it += 1

        if src_replay_buffer.size < 2 * batch_size:
            return
        if tar_replay_buffer.size < batch_size:
            return


        in_warmup = self.total_it <= self.warmup_steps

        # ------------------------------------------------------------------
        # WARMUP – pure SAC
        # ------------------------------------------------------------------
        if in_warmup:
            src_s, src_a, src_ns, src_r, src_not_done, _ = \
                src_replay_buffer.sample(batch_size)

            self._update_reward_critics(src_s, src_a, src_r, src_ns,
                                        src_not_done, writer=writer)

            for p in self.q_funcs.parameters():
                p.requires_grad = False
            for p in self.qc_func.parameters():
                p.requires_grad = False
            self._update_actor(src_s, writer)
            for p in self.q_funcs.parameters():
                p.requires_grad = True
            for p in self.qc_func.parameters():
                p.requires_grad = True

            # Soft-update reward target only during warmup
            with torch.no_grad():
                for tp, p in zip(self.target_q_funcs.parameters(),
                                 self.q_funcs.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

            if writer is not None and self.total_it % 5000 == 0:
                writer.add_scalar('train/warmup_step', self.total_it, self.total_it)
            return

        # ------------------------------------------------------------------
        # FULL C-VWDA
        # ------------------------------------------------------------------



        # Step 2 – Classifiers
        if self.total_it % self.cls_update_freq == 0:
            self.update_classifier(src_replay_buffer, tar_replay_buffer,
                                   batch_size, writer)

        # Step 3 – Sample
        src_s, src_a, src_ns, src_r, src_not_done, _ = \
            src_replay_buffer.sample(batch_size)

        # Step 4 – Importance weights
        weights = self._importance_weights(src_s, src_a, src_ns)

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/iw_mean', weights.mean(), self.total_it)
            writer.add_scalar('train/iw_std',  weights.std(),  self.total_it)

        # Phase 1 – Cost network
        self._update_cost_network(src_s, src_a, weights)

        # Phase 2 – Pre-sample ã once (shared by steps 6 and 7)
        with torch.no_grad():
            next_action, next_logprob, _ = self.policy(src_ns, get_logprob=True)

        # Step 6 – Reward critics
        self._update_reward_critics(src_s, src_a, src_r, src_ns, src_not_done,
                                    next_action=next_action,
                                    next_logprob=next_logprob,
                                    writer=writer)

        # Step 7 – Cost critic (reuse same ã)
        self._update_cost_critic(src_s, src_a, src_ns, src_not_done,
                                 next_action, writer)

        # Step 8 – Actor
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

        # Step 10 – Soft-update targets
        self.update_target()

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(),           filename + "_qr")
        torch.save(self.q_optimizer.state_dict(),       filename + "_qr_opt")
        torch.save(self.qc_func.state_dict(),           filename + "_qc")
        torch.save(self.qc_optimizer.state_dict(),      filename + "_qc_opt")
        torch.save(self.cost_net.state_dict(),          filename + "_cost")
        torch.save(self.cost_optimizer.state_dict(),    filename + "_cost_opt")
        torch.save(self.policy.state_dict(),            filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(),  filename + "_actor_opt")
        torch.save(self.classifier.state_dict(),        filename + "_cls")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_cls_opt")
        torch.save({'lambda': self.lambda_val},         filename + "_dual")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_qr"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_qr_opt"))
        self.qc_func.load_state_dict(torch.load(filename + "_qc"))
        self.qc_optimizer.load_state_dict(torch.load(filename + "_qc_opt"))
        self.cost_net.load_state_dict(torch.load(filename + "_cost"))
        self.cost_optimizer.load_state_dict(torch.load(filename + "_cost_opt"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_opt"))
        self.classifier.load_state_dict(torch.load(filename + "_cls"))
        self.classifier_optimizer.load_state_dict(torch.load(filename + "_cls_opt"))
        dual = torch.load(filename + "_dual")
        self.lambda_val = dual['lambda'].to(self.device)
