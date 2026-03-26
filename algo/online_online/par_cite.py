"""
PAR-CITE: Policy Adaptation via Representation (PAR) augmented with
          Contrastive Intrinsic Target Exploration (CITE)

CITE adds a gap-aware intrinsic exploration bonus to the *target* domain reward.
It drives the agent to explore target states where the source and target dynamics
diverge most, while avoiding re-visiting already-well-explored target regions.

Architecture:
  - DualQueueCITE: a MoCo-style transition encoder with two separate FIFO queues
        * src_queue  : running memory bank of source transition embeddings
        * tar_queue  : running memory bank of target transition embeddings
    The encoder is trained with an InfoNCE loss that:
        (a) PULLS together (src, tar) pairs that share the same action — they
            should look similar if dynamics were identical.
        (b) PUSHES apart all other cross-queue pairs as negatives.
    This shapes the latent space so that the cosine similarity between a new
    target transition and the src_queue measures "how source-like" it is, and
    similarity to the tar_queue measures "how familiar" it already is.

  - Intrinsic bonus (added to tar_reward every step):
        r_int = -sim_to_src - sim_to_tar
        High when the transition looks DIFFERENT from source (genuine dynamics gap)
        AND is NOT already familiar in the target buffer (novel exploration target).

  - Everything in PAR is preserved unchanged:
        * encoder (zs / zsa) trained on target data only → MSE loss
        * distance penalty on source reward: src_reward -= β * distance
        * double Q, policy, EMA encoder_target, temperature all identical

New hyper-parameters (added to config yaml):
    cite_z_dim        : embedding dimension of CITE encoder       (default 128)
    cite_queue_size   : number of slots per queue                 (default 4096)
    cite_momentum     : EMA momentum for key encoder              (default 0.995)
    cite_temperature  : InfoNCE temperature                       (default 0.07)
    cite_coeff        : intrinsic reward coefficient              (default 0.05)
    cite_update_freq  : how often to update CITE encoder          (default 1)
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility: TanhTransform  (identical to PAR)
# ─────────────────────────────────────────────────────────────────────────────

class TanhTransform(Transform):
    r"""Transform via the mapping y = tanh(x)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Shared networks  (identical to PAR)
# ─────────────────────────────────────────────────────────────────────────────

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


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class Encoder(nn.Module):
    """PAR's original dynamics representation encoder (unchanged)."""
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()
        self.activ = activ
        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


# ─────────────────────────────────────────────────────────────────────────────
# CITE: Contrastive Intrinsic Target Exploration
# ─────────────────────────────────────────────────────────────────────────────

class TransitionEncoder(nn.Module):
    """
    Encodes a transition (s, a, s') into a fixed-size embedding.
    Used by CITE's query and key encoders.
    Input  : concatenation of [state, action, next_state]  -> 2*state_dim + action_dim
    Output : L2-normalised embedding of dimension z_dim
    """
    def __init__(self, state_dim, action_dim, z_dim=128, hidden_size=256):
        super(TransitionEncoder, self).__init__()
        in_dim = 2 * state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, z_dim),
        )

    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class DualQueueCITE(nn.Module):
    """
    MoCo-style dual-queue contrastive module for gap-aware exploration.

    Two FIFO queues:
        src_queue [z_dim × queue_size] : recent source transition embeddings
        tar_queue [z_dim × queue_size] : recent target transition embeddings

    Key encoder (encoder_k) is updated by EMA of the query encoder (encoder_q).
    Only encoder_q is trained via gradient; encoder_k provides stable targets.

    InfoNCE training objective
    --------------------------
    Within a batch of B source and B target transitions (paired by batch-index,
    i.e., the i-th source and i-th target took the SAME action from a similar
    state position):

        Positive pair  : (src_q_i, tar_k_i)   — same action index
        Negative set   : all other tar_k_j  (j ≠ i) + entire tar_queue bank

    This pulls together transitions that share the same action effect and
    pushes apart transitions whose dynamics outcomes differ, learning a
    geometry where cross-domain similarity = dynamics compatibility.

    Intrinsic bonus
    ---------------
    For a new target transition (s, a, s'):

        z      = encoder_q(s, a, s')
        sim_src = max cosine similarity to src_queue   (how "source-like")
        sim_tar = max cosine similarity to tar_queue   (how "already explored")

        r_int  = -sim_src - sim_tar
               > 0 when: genuinely novel AND dynamically different from source
               < 0 when: similar to already-seen target data OR source-like (less informative)

    The bonus is clipped and normalised before being added to tar_reward.
    """

    def __init__(self, state_dim, action_dim,
                 z_dim=128, queue_size=4096,
                 momentum=0.995, temperature=0.07,
                 hidden_size=256, device='cuda'):
        super(DualQueueCITE, self).__init__()
        self.z_dim       = z_dim
        self.queue_size  = queue_size
        self.momentum    = momentum
        self.temperature = temperature
        self.device      = device

        # Query encoder  — trained with gradients
        self.encoder_q = TransitionEncoder(state_dim, action_dim, z_dim, hidden_size)
        # Key encoder    — updated by EMA only (no gradients)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        # Dual FIFO queues stored as columns: [z_dim, queue_size]
        self.register_buffer('src_queue',
                             F.normalize(torch.randn(z_dim, queue_size), dim=0))
        self.register_buffer('tar_queue',
                             F.normalize(torch.randn(z_dim, queue_size), dim=0))
        self.register_buffer('src_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('tar_queue_ptr', torch.zeros(1, dtype=torch.long))

    # ── EMA update of key encoder ──────────────────────────────────────────
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.encoder_q.parameters(),
                             self.encoder_k.parameters()):
            p_k.data.mul_(self.momentum).add_(p_q.data * (1.0 - self.momentum))

    # ── FIFO queue enqueue ──────────────────────────────────────────────────
    @torch.no_grad()
    def _dequeue_and_enqueue(self, src_keys, tar_keys):
        """Push a batch of key embeddings into the respective queues (FIFO)."""
        B = src_keys.shape[0]
        assert B <= self.queue_size, \
            f"Batch size {B} exceeds CITE queue_size {self.queue_size}"

        src_ptr = int(self.src_queue_ptr)
        tar_ptr = int(self.tar_queue_ptr)

        # Wrap-around FIFO
        end_src = src_ptr + B
        end_tar = tar_ptr + B

        if end_src <= self.queue_size:
            self.src_queue[:, src_ptr:end_src] = src_keys.T
        else:
            overflow = end_src - self.queue_size
            self.src_queue[:, src_ptr:] = src_keys[:(B - overflow)].T
            self.src_queue[:, :overflow] = src_keys[(B - overflow):].T

        if end_tar <= self.queue_size:
            self.tar_queue[:, tar_ptr:end_tar] = tar_keys.T
        else:
            overflow = end_tar - self.queue_size
            self.tar_queue[:, tar_ptr:] = tar_keys[:(B - overflow)].T
            self.tar_queue[:, :overflow] = tar_keys[(B - overflow):].T

        self.src_queue_ptr[0] = end_src % self.queue_size
        self.tar_queue_ptr[0] = end_tar % self.queue_size

    # ── InfoNCE contrastive loss ────────────────────────────────────────────
    def contrastive_loss(self, src_s, src_a, src_ns, tar_s, tar_a, tar_ns):
        """
        Compute InfoNCE loss pairing (src_i, tar_i) as positive.

        Query : src transitions encoded with encoder_q  [B, z_dim]
        Key   : tar transitions encoded with encoder_k  [B, z_dim]
        Memory: all entries in tar_queue               [z_dim, queue_size]

        Logits : [B, 1 + queue_size]
            col 0     : positive similarity  q_i · k_i
            col 1..Q  : negative similarities q_i · tar_queue_j
        """
        B = src_s.shape[0]

        # Query embeddings (with grad)
        q = self.encoder_q(src_s, src_a, src_ns)           # [B, z_dim]

        # Key embeddings (no grad) + EMA update
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(tar_s, tar_a, tar_ns)       # [B, z_dim]

        # Positive logits: [B, 1]
        l_pos = (q * k).sum(dim=-1, keepdim=True) / self.temperature

        # Negative logits from tar_queue: [B, queue_size]
        l_neg = (q @ self.tar_queue.clone().detach()) / self.temperature

        # Full logits: [B, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)
        # Label 0 = positive is always index 0
        labels = torch.zeros(B, dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits, labels)

        # Enqueue current keys into both queues for future negatives
        with torch.no_grad():
            src_k = self.encoder_k(src_s, src_a, src_ns)   # [B, z_dim]
            self._dequeue_and_enqueue(src_k, k)

        return loss

    # ── Intrinsic reward computation ────────────────────────────────────────
    @torch.no_grad()
    def intrinsic_reward(self, tar_s, tar_a, tar_ns):
        """
        Compute gap-aware intrinsic bonus for target transitions.

        r_int = -sim_to_src - sim_to_tar

        sim_to_src : max cosine similarity to src_queue  → measures "source-likeness"
                     HIGH  ⟹  dynamics similar to source  ⟹  less informative to explore
        sim_to_tar : max cosine similarity to tar_queue  → measures "target familiarity"
                     HIGH  ⟹  already well-explored region  ⟹  less useful

        The negation ensures the bonus is HIGHEST when:
            (1) the transition looks DIFFERENT from the source  (dynamics gap present)
            (2) the transition is NOVEL in the target buffer    (unexplored region)

        Returns: [B, 1] intrinsic bonus (un-scaled; caller applies cite_coeff)
        """
        z = self.encoder_q(tar_s, tar_a, tar_ns)           # [B, z_dim]

        # Max similarity over all queue entries
        sim_src = (z @ self.src_queue).max(dim=-1).values  # [B]
        sim_tar = (z @ self.tar_queue).max(dim=-1).values  # [B]

        bonus = -sim_src - sim_tar                         # [B]

        # Normalise within the batch to have zero mean and unit std
        # to keep the magnitude comparable across training phases
        bonus = (bonus - bonus.mean()) / (bonus.std() + 1e-8)

        return bonus.unsqueeze(-1)                         # [B, 1]


# ─────────────────────────────────────────────────────────────────────────────
# PAR-CITE  (main algorithm class)
# ─────────────────────────────────────────────────────────────────────────────

class PAR_CITE(object):
    """
    PAR-CITE: PAR + Contrastive Intrinsic Target Exploration.

    Differences from vanilla PAR
    ─────────────────────────────
    1. An additional DualQueueCITE module is created and optimised.
    2. After every `cite_update_freq` iterations the CITE encoder is updated
       via the InfoNCE loss on the current (src, tar) batch.
    3. An intrinsic bonus r_int is added to tar_reward BEFORE the Q-update:
            tar_reward += cite_coeff * r_int
       This steers the policy to explore target regions where dynamics diverge
       from the source, filling the target buffer with informative transitions.
    4. Everything else in PAR (encoder, distance, src reward penalty) is
       preserved 100% unchanged.
    """

    def __init__(self, config, device, target_entropy=None):
        self.config  = config
        self.device  = device
        self.discount = config['gamma']
        self.tau      = config['tau']
        self.target_entropy = (target_entropy if target_entropy
                               else -config['action_dim'])
        self.update_interval = config['update_interval']
        self.total_it = 0

        # ── PAR components (unchanged) ─────────────────────────────────────
        self.q_funcs = DoubleQFunc(
            config['state_dim'], config['action_dim'],
            hidden_size=config['hidden_sizes']
        ).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.policy = Policy(
            config['state_dim'], config['action_dim'],
            config['max_action'], hidden_size=config['hidden_sizes']
        ).to(self.device)

        # PAR encoder: learns target dynamics via predictive MSE objective
        self.encoder = Encoder(config['state_dim'], config['action_dim']).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder)
        self.encoder_target.eval()

        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        self.q_optimizer       = torch.optim.Adam(self.q_funcs.parameters(),
                                                   lr=config['critic_lr'])
        self.policy_optimizer  = torch.optim.Adam(self.policy.parameters(),
                                                   lr=config['actor_lr'])
        self.temp_optimizer    = torch.optim.Adam([self.log_alpha],
                                                   lr=config['actor_lr'])
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),
                                                   lr=config['actor_lr'])

        # ── CITE components (new) ─────────────────────────────────────────
        cite_z_dim      = config.get('cite_z_dim',      128)
        cite_queue_size = config.get('cite_queue_size', 4096)
        cite_momentum   = config.get('cite_momentum',   0.995)
        cite_temperature= config.get('cite_temperature',0.07)
        cite_hidden     = config.get('hidden_sizes',    256)

        self.cite_coeff      = config.get('cite_coeff',       0.05)
        self.cite_update_freq= config.get('cite_update_freq', 1)

        self.cite = DualQueueCITE(
            state_dim   = config['state_dim'],
            action_dim  = config['action_dim'],
            z_dim       = cite_z_dim,
            queue_size  = cite_queue_size,
            momentum    = cite_momentum,
            temperature = cite_temperature,
            hidden_size = cite_hidden,
            device      = device,
        ).to(self.device)

        self.cite_optimizer = torch.optim.Adam(
            self.cite.encoder_q.parameters(), lr=config['actor_lr']
        )

    # ── Inherited from PAR (unchanged) ────────────────────────────────────

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(
                torch.Tensor(state).view(1, -1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """Moving average update of Q and PAR encoder target networks."""
        with torch.no_grad():
            for tp, p in zip(self.target_q_funcs.parameters(),
                              self.q_funcs.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(self.encoder_target.parameters(),
                              self.encoder.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch,
                           nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(
                nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            q_target = torch.min(q_t1, q_t2)
            if self.config['entropy_backup']:
                value_target = (reward_batch
                                + not_done_batch * self.discount
                                * (q_target - self.alpha * logprobs_batch))
            else:
                value_target = (reward_batch
                                + not_done_batch * self.discount * q_target)

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss   = -self.alpha * (logprobs_batch.detach()
                                     + self.target_entropy).mean()
        return policy_loss, temp_loss

    def update_encoder(self, state_batch, action_batch, nextstate_batch,
                       writer=None):
        """PAR's MSE predictive encoder loss on target data."""
        with torch.no_grad():
            next_zs = self.encoder.zs(nextstate_batch)
        zs      = self.encoder.zs(state_batch)
        pred_zs = self.encoder.zsa(zs, action_batch)
        encoder_loss = F.mse_loss(pred_zs, next_zs)
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/encoder_loss', encoder_loss,
                              global_step=self.total_it)

    # ── CITE-specific update ───────────────────────────────────────────────

    def update_cite(self, src_s, src_a, src_ns, tar_s, tar_a, tar_ns,
                    writer=None):
        """
        Update the CITE encoder via InfoNCE and enqueue fresh key embeddings.

        Positive pair  : (src_i, tar_i)  — same batch index, same action taken
        Negatives      : entire tar_queue bank (O(queue_size) negatives)

        The loss pulls together src/tar transitions with compatible dynamics
        and pushes apart those with different dynamics outcomes.
        """
        cite_loss = self.cite.contrastive_loss(src_s, src_a, src_ns,
                                               tar_s, tar_a, tar_ns)
        self.cite_optimizer.zero_grad()
        cite_loss.backward()
        self.cite_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/cite_loss', cite_loss.item(),
                              global_step=self.total_it)
        return cite_loss.item()

    # ── Main training step ────────────────────────────────────────────────

    def train(self, src_replay_buffer, tar_replay_buffer,
              batch_size=128, writer=None):
        self.total_it += 1

        # Wait until both buffers have enough data
        if src_replay_buffer.size < 500 or tar_replay_buffer.size < 500:
            return

        src_state, src_action, src_next_state, src_reward, src_not_done = \
            src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = \
            tar_replay_buffer.sample(batch_size)

        # ── Step 1: Update PAR encoder (target dynamics, MSE) ─────────────
        # (every 200 steps, same cadence as original PAR)
        if self.total_it % 200 == 0:
            tar_s, tar_a, tar_ns, _, _ = tar_replay_buffer.sample(batch_size // 2)
            self.update_encoder(tar_s, tar_a, tar_ns, writer)

        # ── Step 2: Update CITE encoder (cross-domain contrastive) ────────
        # Runs at cite_update_freq (default every step) on the current batch.
        # Uses the SAME paired batch: src_i and tar_i took the same action
        # from comparably-positioned states — natural positive pair structure.
        if self.total_it % self.cite_update_freq == 0:
            self.update_cite(
                src_state, src_action, src_next_state,
                tar_state, tar_action, tar_next_state,
                writer
            )

        # ── Step 3: Compute PAR distance penalty (unchanged) ──────────────
        with torch.no_grad():
            next_src_zs  = self.encoder_target.zs(src_next_state)
            src_zs       = self.encoder_target.zs(src_state)
            pred_src_zs  = self.encoder_target.zsa(src_zs, src_action)
            distance     = ((pred_src_zs - next_src_zs) ** 2).mean(
                dim=-1, keepdim=True)

        src_reward -= self.config['beta'] * distance

        # ── Step 4: Compute CITE intrinsic bonus for target (new) ─────────
        # r_int = -sim_to_src - sim_to_tar
        #   Positive when: transition is NOVEL in target AND differs from source
        #   This is added to tar_reward so the Q-function propagates the
        #   exploration signal back through the Bellman backup.
        with torch.no_grad():
            r_int = self.cite.intrinsic_reward(
                tar_state, tar_action, tar_next_state)  # [B, 1]

        tar_reward = tar_reward + self.cite_coeff * r_int

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/par_distance',   distance.mean(),
                              self.total_it)
            writer.add_scalar('train/src_reward',     src_reward.mean(),
                              self.total_it)
            writer.add_scalar('train/cite_r_int_mean', r_int.mean(),
                              self.total_it)
            writer.add_scalar('train/cite_r_int_std',  r_int.std(),
                              self.total_it)
            writer.add_scalar('train/tar_reward_augmented', tar_reward.mean(),
                              self.total_it)

        # ── Step 5: Q-function update on mixed batch ───────────────────────
        state      = torch.cat([src_state,      tar_state],      0)
        action     = torch.cat([src_action,     tar_action],     0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward     = torch.cat([src_reward,     tar_reward],     0)
        not_done   = torch.cat([src_not_done,   tar_not_done],   0)

        q_loss_step = self.update_q_functions(
            state, action, reward, next_state, not_done, writer)
        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # ── Step 6: Policy and temperature update ─────────────────────────
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        pi_loss_step, a_loss_step = self.update_policy_and_temp(state)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

        if self.config['temperature_opt']:
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()
