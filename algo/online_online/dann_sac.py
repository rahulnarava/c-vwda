"""
DANN-SAC: Domain-Adversarial Neural Network + Soft Actor-Critic
for Online-Online Off-Dynamics RL

Motivation
----------
Both DARC and PAR correct for dynamics mismatch at the *reward level* -- they penalise
source transitions whose $(s,a,s')$ tuples look different from the target domain.
The Q-function and policy still receive *raw states*, which carry domain-specific
signatures (e.g. different joint velocities under friction changes).

This algorithm instead corrects at the *representation level*, borrowing the
Domain-Adversarial Neural Network (DANN) idea from computer vision
(Ganin et al., 2016, "Domain-Adversarial Training of Neural Networks").

Architecture
------------

    (s) ──► [State Encoder f_φ] ──► z_s ──► [Q-function / Policy]   (task stream)
                                       │
                                       └──► [Domain Discriminator D_ψ]  (domain stream)
                                                    ▲
                                            Gradient Reversal Layer (GRL)

Training Signal
---------------
* Task stream (Q / policy):  standard SAC on z_s, mixing src and tar batches.
* Domain stream (discriminator):  binary cross-entropy to classify z_s as src (0)
  or tar (1).
* GRL reversal:  during back-prop through the encoder, the discriminator gradient
  is *negated* (scaled by -λ).  This forces f_φ to produce representations from
  which the discriminator *cannot* tell the two domains apart, i.e. it learns
  domain-invariant features.

Intuition
---------
After training, z_s captures only task-relevant structure (joint angles, velocities,
contact forces that matter for locomotion) and discards domain-specific artifacts
(absolute friction / gravity signatures).  The Q-function trained on z_s therefore
generalises from source to target without needing any reward correction.

Hyperparameters (on top of SAC defaults)
-----------------------------------------
  grl_lambda      : gradient reversal strength λ (default 0.1).
                    Larger → stronger domain alignment but noisier task learning.
  zs_dim          : dimension of the shared latent state z_s (default 256).
  disc_hidden_size: hidden size of the domain discriminator (default 256).
  disc_lr         : learning rate for the discriminator (default 3e-4).
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


# ---------------------------------------------------------------------------
# Utility: TanhTransform  (identical to the rest of the codebase)
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


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """
    Forward pass: identity.
    Backward pass: negate the gradient and scale by lambda.

    This makes the layer upstream (the encoder) minimise the discriminator loss
    instead of maximising it, i.e. it learns to *fool* the discriminator.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        # negate gradient so encoder actively confuses the discriminator
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        lambda_tensor = torch.tensor(self.lambda_, dtype=x.dtype, device=x.device)
        return GradientReversalFunction.apply(x, lambda_tensor)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# ---------------------------------------------------------------------------
# Network Components
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


class StateEncoder(nn.Module):
    """
    Maps raw state s  ──►  domain-invariant latent z_s.

    The GRL is *not* baked into this module.  Instead the caller inserts it
    between encoder output and discriminator input so that only the adversarial
    path is reversed while the task path (Q / policy) receives normal gradients.
    """

    def __init__(self, state_dim, zs_dim=256, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, zs_dim),
            nn.LayerNorm(zs_dim),   # stabilises adversarial training
        )

    def forward(self, state):
        return self.net(state)


class DomainDiscriminator(nn.Module):
    """
    Binary classifier: given z_s, predict P(domain = target).
    Trained with standard cross-entropy, but the GRL ensures that the
    encoder produces features that maximally confuse this classifier.
    """

    def __init__(self, zs_dim=256, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),          # logits for [src, tar]
        )

    def forward(self, z):
        return self.net(z)


class Policy(nn.Module):
    """
    Squashed-Gaussian policy that operates on the latent z_s
    produced by StateEncoder.
    """

    def __init__(self, zs_dim, action_dim, max_action, hidden_size=256):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(zs_dim, action_dim * 2, hidden_size)

    def forward(self, z, get_logprob=False):
        mu_logstd = self.network(z)
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


class DoubleQFunc(nn.Module):
    """
    Twin Q-functions that operate on (z_s, a) instead of (s, a).
    """

    def __init__(self, zs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network1 = MLPNetwork(zs_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(zs_dim + action_dim, 1, hidden_size)

    def forward(self, z, action):
        x = torch.cat((z, action), dim=1)
        return self.network1(x), self.network2(x)


# ---------------------------------------------------------------------------
# DANN-SAC Algorithm
# ---------------------------------------------------------------------------

class DANN_SAC(object):
    """
    Domain-Adversarial Neural Network + SAC for the Online-Online
    off-dynamics RL setting.

    Key design choices
    ------------------
    1. A shared StateEncoder maps raw states to domain-invariant z_s.
    2. A DomainDiscriminator receives z_s through a GradientReversalLayer
       and tries to classify src vs tar.  The GRL ensures the encoder is
       trained to remove domain-specific information.
    3. The Q-function and policy are trained on z_s using standard SAC
       with both src and tar batches mixed (no reward penalty needed).
    4. The discriminator is trained with its own optimizer to stay a strong
       adversary; only the encoder sees the reversed gradient.
    5. grl_lambda can be annealed upward during training (progressive
       alignment), similar to the original DANN schedule.
    """

    def __init__(self, config, device, target_entropy=None):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']
        self.total_it = 0

        zs_dim = config.get('zs_dim', 256)
        disc_hidden = config.get('disc_hidden_size', 256)
        self.grl_lambda = config.get('grl_lambda', 0.1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = StateEncoder(
            config['state_dim'], zs_dim, config['hidden_sizes']
        ).to(device)
        self.encoder_target = copy.deepcopy(self.encoder)
        self.encoder_target.eval()
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        # ── Domain Discriminator + GRL ────────────────────────────────────────
        self.grl = GradientReversalLayer(lambda_=self.grl_lambda)
        self.discriminator = DomainDiscriminator(zs_dim, disc_hidden).to(device)

        # ── Actor / Critic ───────────────────────────────────────────────────
        self.q_funcs = DoubleQFunc(
            zs_dim, config['action_dim'], config['hidden_sizes']
        ).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.policy = Policy(
            zs_dim, config['action_dim'], config['max_action'],
            config['hidden_sizes']
        ).to(device)

        # ── Temperature ──────────────────────────────────────────────────────
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(device)

        # ── Optimisers ───────────────────────────────────────────────────────
        # Encoder + Q + policy share one "task" optimizer pathway.
        # The discriminator has its own optimizer so it stays a strong adversary.
        encoder_task_params = list(self.encoder.parameters())
        self.q_optimizer = torch.optim.Adam(
            list(self.q_funcs.parameters()) + encoder_task_params,
            lr=config['critic_lr']
        )
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + encoder_task_params,
            lr=config['actor_lr']
        )
        self.temp_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config['actor_lr']
        )
        # Discriminator optimizer: updates D_ψ only (not the encoder).
        # The encoder is updated adversarially via GRL through the task optimizers.
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('disc_lr', 3e-4)
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, test=True):
        with torch.no_grad():
            z = self.encoder(torch.FloatTensor(state).view(1, -1).to(self.device))
            action, _, mean = self.policy(z)
        return (mean if test else action).squeeze().cpu().numpy()

    def update_target(self):
        """EMA update for target Q-functions and target encoder."""
        with torch.no_grad():
            for tp, p in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(self.encoder_target.parameters(), self.encoder.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # -----------------------------------------------------------------------
    # Discriminator update  (step 1 in each train() call)
    # -----------------------------------------------------------------------

    def update_discriminator(self, src_state, tar_state, writer=None):
        """
        Train the domain discriminator D_ψ on encoder outputs *without*
        gradient reversal (the discriminator itself wants to discriminate;
        only the encoder should be confused, which happens via the task
        optimizers that see the GRL path).

        Loss: cross-entropy  with src→label 0, tar→label 1.
        """
        # Detach encoder output so discriminator gradients don't touch encoder here.
        # Encoder will be adversarially updated through the task optimizers.
        with torch.no_grad():
            z_src = self.encoder(src_state)
            z_tar = self.encoder(tar_state)

        z_all = torch.cat([z_src, z_tar], dim=0)
        labels = torch.cat([
            torch.zeros(src_state.size(0), dtype=torch.long),
            torch.ones(tar_state.size(0),  dtype=torch.long),
        ], dim=0).to(self.device)

        logits = self.discriminator(z_all)
        disc_loss = F.cross_entropy(logits, labels)

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            writer.add_scalar('train/disc_loss', disc_loss.item(), self.total_it)
            writer.add_scalar('train/disc_acc',  acc.item(),       self.total_it)

        return disc_loss.item()

    # -----------------------------------------------------------------------
    # Adversarial alignment loss (step 2 — injected into Q / policy updates)
    # -----------------------------------------------------------------------

    def adversarial_alignment_loss(self, src_state, tar_state):
        """
        Compute the GRL-based domain confusion loss.

        The encoder produces z_s; the GRL negates the gradient; the
        discriminator then tries to classify the domain.  Because the
        gradient is reversed, the encoder is nudged to produce
        representations that look the same for both domains.

        Returns scalar loss that must be added to the task loss.
        """
        z_src = self.encoder(src_state)
        z_tar = self.encoder(tar_state)
        z_all = torch.cat([z_src, z_tar], dim=0)

        # Pass through GRL — forward is identity, backward negates
        z_reversed = self.grl(z_all)
        logits = self.discriminator(z_reversed)

        labels = torch.cat([
            torch.zeros(src_state.size(0), dtype=torch.long),
            torch.ones(tar_state.size(0),  dtype=torch.long),
        ], dim=0).to(self.device)

        # Standard cross-entropy.  The GRL makes the encoder gradient go in
        # the direction that *increases* this loss → domain confusion.
        return F.cross_entropy(logits, labels)

    # -----------------------------------------------------------------------
    # Q-function update
    # -----------------------------------------------------------------------

    def update_q_functions(self, state_batch, action_batch, reward_batch,
                           nextstate_batch, not_done_batch,
                           src_state, tar_state, writer=None):
        """
        Standard twin-Q Bellman loss computed on z_s, plus the adversarial
        alignment term so that the encoder learns domain-invariant features
        during the Q update.
        """
        # Encode current and next states (with gradient for encoder)
        z = self.encoder(state_batch)
        with torch.no_grad():
            # Use target encoder for stable bootstrap targets
            z_next = self.encoder_target(nextstate_batch)
            next_action, logprobs, _ = self.policy(z_next, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(z_next, next_action)
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * (
                q_target - self.alpha * logprobs
            )

        q_1, q_2 = self.q_funcs(z, action_batch)
        task_loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)

        # Adversarial alignment: encoder should fool the discriminator
        adv_loss = self.adversarial_alignment_loss(src_state, tar_state)
        total_loss = task_loss + self.grl_lambda * adv_loss

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q_task_loss', task_loss.item(), self.total_it)
            writer.add_scalar('train/adv_loss',    adv_loss.item(),  self.total_it)
            writer.add_scalar('train/q1',          q_1.mean().item(), self.total_it)

        return total_loss

    # -----------------------------------------------------------------------
    # Policy & temperature update
    # -----------------------------------------------------------------------

    def update_policy_and_temp(self, z_batch):
        """
        Standard SAC policy update on latent z (no re-encoding needed; z
        was already computed in Q update but we re-compute here to get fresh
        gradients through the encoder).
        """
        action_batch, logprobs_batch, _ = self.policy(z_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(z_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    # -----------------------------------------------------------------------
    # Progressive GRL annealing (optional, call externally or in train)
    # -----------------------------------------------------------------------

    def anneal_grl_lambda(self, progress: float, lambda_max: float = 1.0):
        """
        Gradually increase lambda from 0 to lambda_max following the
        schedule in the original DANN paper:
            λ(p) = 2 / (1 + exp(−10·p)) − 1    where p ∈ [0, 1]

        Call this each step with progress = total_it / max_steps.
        """
        lambda_p = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        new_lambda = lambda_max * lambda_p
        self.grl_lambda = new_lambda
        self.grl.set_lambda(new_lambda)

    # -----------------------------------------------------------------------
    # Main training step
    # -----------------------------------------------------------------------

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        self.total_it += 1

        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        src_state, src_action, src_next_state, src_reward, src_not_done = \
            src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = \
            tar_replay_buffer.sample(batch_size)

        # ── Step 1: Update discriminator (D_ψ stays a strong adversary) ──────
        self.update_discriminator(src_state, tar_state, writer)

        # ── Step 2: Mix batches for task learning ─────────────────────────────
        state     = torch.cat([src_state,      tar_state],      dim=0)
        action    = torch.cat([src_action,     tar_action],     dim=0)
        next_state= torch.cat([src_next_state, tar_next_state], dim=0)
        reward    = torch.cat([src_reward,     tar_reward],     dim=0)
        not_done  = torch.cat([src_not_done,   tar_not_done],   dim=0)

        # ── Step 3: Q update with adversarial alignment ───────────────────────
        q_loss = self.update_q_functions(
            state, action, reward, next_state, not_done,
            src_state, tar_state, writer
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.update_target()

        # ── Step 4: Policy update ─────────────────────────────────────────────
        # Re-encode with fresh gradients for the policy path
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        z_for_policy = self.encoder(state)
        pi_loss, temp_loss = self.update_policy_and_temp(z_for_policy)

        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()

        if self.config['temperature_opt']:
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(),      filename + "_critic")
        torch.save(self.q_optimizer.state_dict(),  filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(),       filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.encoder.state_dict(),      filename + "_encoder")
        torch.save(self.discriminator.state_dict(),filename + "_discriminator")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.discriminator.load_state_dict(torch.load(filename + "_discriminator"))
