import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from env_hybrid import EnvConfig, HybridFulfillmentEnv
from train_bc import BCActor
from train_rm import TrajectoryRewardModel


@dataclass
class PPOv2RLHFConfig:
    seed: int = 20240908
    total_updates: int = 2000
    rollout_steps: int = 48
    ppo_epochs: int = 4
    actor_lr: float = 5e-5
    actor_lr_shared_mult: float = 1.0
    actor_lr_price_mult: float = 2.0
    actor_lr_rebalance_mult: float = 1.0
    critic_warmup_updates: int = 0
    value_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02
    entropy_coef_min: float = 0.005
    entropy_decay_start: int = 500
    entropy_decay_rate: float = 0.999
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    kl_coef: float = 0.0
    target_kl: float = 100.0
    kl_adapt_rate: float = 1.5
    kl_coef_min: float = 1e-4
    kl_coef_max: float = 0.05
    rm_coef: float = 0.05
    rm_coef_start: float = 0.05
    rm_warmup_updates: int = 0
    rm_interval_steps: int = 6
    rm_credit_mode: str = "segment"  # segment: spread RM delta over interval; endpoint: old behavior
    rm_delta_clip: float = 5.0
    rm_terminal_coef: float = 0.0
    env_reward_weight: float = 1.0
    reward_scale: float = 10000.0
    rebalance_std: float = 2.0
    rebalance_std_final: float = 0.5
    rebalance_std_decay_start: int = 600
    rebalance_std_decay_end: int = 1200
    train_phase: str = "joint"
    save_every: int = 50


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.array([obs["slot"]], dtype=np.float32),
            obs["price_idx"].astype(np.float32),
            obs["av_units"].astype(np.float32),
            obs["cs_units"].astype(np.float32),
            obs["outstanding"].astype(np.float32),
            obs["capacity"].astype(np.float32),
            obs["workload"].astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((state - mean) / std).astype(np.float32)


def project_rebalance_action(env: HybridFulfillmentEnv, mat: np.ndarray) -> np.ndarray:
    """
    Project rebalance action to the same feasible region used by env step:
    non-negative, mask invalid OD pairs, and per-source max send ratio.
    """
    x = np.asarray(mat, dtype=np.float32).copy()
    x = np.maximum(x, 0.0)
    mask = getattr(env, "rebalance_mask", env.adj_mask)
    x = np.where(mask, x, 0.0)
    np.fill_diagonal(x, 0.0)
    max_ratio = float(getattr(env.config, "rebalance_max_send_ratio", 1.0))
    for i in range(env.n):
        row_sum = float(np.sum(x[i]))
        if row_sum <= 1e-8:
            continue
        avail = float(env.av_units[i])
        max_move = max(0.0, max_ratio) * avail
        if max_move <= 1e-8:
            x[i, :] = 0.0
            continue
        if row_sum > max_move:
            x[i, :] = x[i, :] * (max_move / row_sum)
    return x


class PPOActor(nn.Module):
    def __init__(self, base_actor: BCActor, rebalance_std: float = 0.10):
        super().__init__()
        self.base_actor = base_actor
        n = int(base_actor.n_region)
        self.log_std = nn.Parameter(
            torch.full((n, n), float(np.log(rebalance_std)), dtype=torch.float32)
        )

    def forward(self, states):
        return self.base_actor(states)

    def dists(self, states):
        price_logits, reb_mean = self.forward(states)  # [B,N,P], [B,N,N]
        p_dist = Categorical(logits=price_logits)
        r_std = torch.exp(self.log_std).clamp(min=0.01, max=5.0)
        r_dist = Normal(reb_mean, r_std)
        return p_dist, r_dist, reb_mean

    @torch.no_grad()
    def set_rebalance_std(self, std: float):
        std = float(max(1e-4, std))
        self.log_std.fill_(float(np.log(std)))

    def sample_actions(self, states, deterministic=False):
        p_dist, r_dist, reb_mean = self.dists(states)
        if deterministic:
            price_a = torch.argmax(p_dist.logits, dim=-1)
            reb_a = reb_mean
        else:
            price_a = p_dist.sample()
            reb_a = r_dist.sample()
        reb_a = F.relu(reb_a)
        return price_a, reb_a

    def logprob_of_actions(self, states, price_actions, reb_actions, reb_dim_mask=None):
        p_dist, r_dist, _ = self.dists(states)
        
        # [CRITICAL FIX]: PPO requires Joint Log Probability (SUM), not Mean!
        p_lp = p_dist.log_prob(price_actions).sum(dim=-1)
        
        r_lp_all = r_dist.log_prob(reb_actions)
        ent_r_all = r_dist.entropy()
        if reb_dim_mask is not None:
            mask = reb_dim_mask.to(r_lp_all.device, dtype=r_lp_all.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            denom = mask.sum(dim=(-1, -2)).clamp(min=1.0)
            
            # [CRITICAL FIX]: Use sum() instead of mean()
            r_lp = (r_lp_all * mask).sum(dim=(-1, -2))
            # Keep entropy on mean scale for entropy coefficient compatibility.
            entropy_r = (ent_r_all * mask).sum(dim=(-1, -2)) / denom
        else:
            # [CRITICAL FIX]: Use sum()
            r_lp = r_lp_all.sum(dim=(-1, -2))
            entropy_r = ent_r_all.mean(dim=(-1, -2))
        entropy_p = p_dist.entropy().mean(dim=-1)
        return p_lp, r_lp, entropy_p, entropy_r


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, states):
        return self.net(states).squeeze(-1)


def build_rm_from_ckpt(rm_ckpt_path: str, device: torch.device):
    ckpt = torch.load(rm_ckpt_path, map_location=device)
    rm = TrajectoryRewardModel(
        state_dim=int(ckpt["state_dim"]),
        n_region=int(ckpt["n_region"]),
        n_price_levels=int(ckpt["n_price_levels"]),
        hidden_dim=int(ckpt["config"]["hidden_dim"]),
    ).to(device)
    rm.load_state_dict(ckpt["model_state_dict"])
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    rm_mean = np.asarray(ckpt.get("state_mean", None), dtype=np.float32).reshape(-1) if ckpt.get("state_mean", None) is not None else None
    rm_std = np.asarray(ckpt.get("state_std", None), dtype=np.float32).reshape(-1) if ckpt.get("state_std", None) is not None else None
    if rm_mean is not None and rm_std is not None:
        rm_std = np.where(rm_std < 1e-6, 1.0, rm_std)
    return rm, ckpt, rm_mean, rm_std


def build_actor_from_ckpt(actor_ckpt_path: str, device: torch.device):
    ckpt = torch.load(actor_ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        state_dict = ckpt.get("actor_state_dict", None)
    if state_dict is None:
        raise KeyError(
            f"{actor_ckpt_path} does not contain 'model_state_dict' or 'actor_state_dict'."
        )

    # PPO checkpoints save actor params with "base_actor." prefix.
    if any(k.startswith("base_actor.") for k in state_dict.keys()):
        state_dict = {
            (k[len("base_actor."):] if k.startswith("base_actor.") else k): v
            for k, v in state_dict.items()
        }
    # PPO checkpoints may contain PPO-only parameter "log_std".
    # Drop it when reconstructing a pure BCActor.
    if "log_std" in state_dict:
        state_dict.pop("log_std")

    def _cfg_get(name, default=None):
        cfg = ckpt.get("config", {})
        if isinstance(cfg, dict) and name in cfg:
            return cfg[name]
        return default

    # Prefer explicit metadata; fall back to cfg / tensor-shape inference for legacy ckpts.
    state_dim = ckpt.get("state_dim", _cfg_get("state_dim"))
    n_region = ckpt.get("n_region", _cfg_get("n_region"))
    n_price_levels = ckpt.get("n_price_levels", _cfg_get("n_price_levels"))
    hidden_dim = _cfg_get("hidden_dim")

    if state_dim is None and "shared_net.0.weight" in state_dict:
        state_dim = int(state_dict["shared_net.0.weight"].shape[1])
    if hidden_dim is None and "shared_net.0.weight" in state_dict:
        hidden_dim = int(state_dict["shared_net.0.weight"].shape[0])
    if n_region is None and "rebalance_head.2.bias" in state_dict:
        reb_out = int(state_dict["rebalance_head.2.bias"].numel())
        n_region = int(round(np.sqrt(reb_out)))
    if (n_price_levels is None) and (n_region is not None) and ("price_head.2.bias" in state_dict):
        price_out = int(state_dict["price_head.2.bias"].numel())
        if n_region > 0 and price_out % n_region == 0:
            n_price_levels = int(price_out // n_region)

    missing = [
        k for k, v in {
            "state_dim": state_dim,
            "n_region": n_region,
            "n_price_levels": n_price_levels,
            "hidden_dim": hidden_dim,
        }.items() if v is None
    ]
    if missing:
        raise KeyError(
            f"Cannot infer {missing} from checkpoint: {actor_ckpt_path}. "
            "Please provide a BC checkpoint with metadata."
        )

    use_region_attn = bool(_cfg_get("use_region_attn", True))
    actor = BCActor(
        state_dim=int(state_dim),
        n_region=int(n_region),
        n_price_levels=int(n_price_levels),
        hidden_dim=int(hidden_dim),
        use_region_attn=use_region_attn,
    ).to(device)
    actor.load_state_dict(state_dict, strict=True)

    # Legacy compatibility: if stats are absent, default to identity normalization.
    if ("state_mean" in ckpt) and ("state_std" in ckpt):
        mean = np.asarray(ckpt["state_mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(ckpt["state_std"], dtype=np.float32).reshape(-1)
    else:
        mean = np.zeros((int(state_dim),), dtype=np.float32)
        std = np.ones((int(state_dim),), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return actor, ckpt, mean, std


def reset_module_parameters(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def apply_half_cold_start(base_actor: BCActor, init_reb_bias: float = 1.0):
    # Keep backbone representation, reset only output heads.
    reset_module_parameters(base_actor.price_head)
    reset_module_parameters(base_actor.rebalance_head)
    # Force early dispatch exploration: shift rebalance head output upward.
    if isinstance(base_actor.rebalance_head, nn.Sequential) and len(base_actor.rebalance_head) > 0:
        last = base_actor.rebalance_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, float(init_reb_bias))


def set_module_requires_grad(module: nn.Module, enabled: bool):
    for p in module.parameters():
        p.requires_grad_(enabled)


def apply_train_phase(actor: PPOActor, phase: str):
    base = actor.base_actor
    phase = str(phase).strip().lower()
    if phase == "joint":
        set_module_requires_grad(base.shared_net, True)
        set_module_requires_grad(base.price_head, True)
        set_module_requires_grad(base.rebalance_head, True)
        actor.log_std.requires_grad_(True)
        return
    if phase == "price":
        # Price-only: keep representation/rebalance fixed, train only pricing head.
        set_module_requires_grad(base.shared_net, False)
        set_module_requires_grad(base.price_head, True)
        set_module_requires_grad(base.rebalance_head, False)
        actor.log_std.requires_grad_(False)
        return
    if phase == "rebalance":
        # Rebalance-only: keep representation/pricing fixed, train only rebalance branch.
        set_module_requires_grad(base.shared_net, False)
        set_module_requires_grad(base.price_head, False)
        set_module_requires_grad(base.rebalance_head, True)
        actor.log_std.requires_grad_(True)
        return
    raise ValueError(f"Unsupported --train-phase: {phase}. Choose from joint/price/rebalance.")


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + values
    return adv, ret


def scheduled_rebalance_std(cfg: PPOv2RLHFConfig, upd: int) -> float:
    # High exploration first, then anneal to a stable execution std.
    if upd <= cfg.rebalance_std_decay_start:
        return float(cfg.rebalance_std)
    if upd >= cfg.rebalance_std_decay_end:
        return float(cfg.rebalance_std_final)
    span = max(1, cfg.rebalance_std_decay_end - cfg.rebalance_std_decay_start)
    ratio = (upd - cfg.rebalance_std_decay_start) / span
    return float(cfg.rebalance_std + ratio * (cfg.rebalance_std_final - cfg.rebalance_std))


def rollout(env, actor: PPOActor, actor_ref: PPOActor, value_net: ValueNetwork, rm_model: TrajectoryRewardModel, rm_n_price_levels: int,
            state_mean: np.ndarray, state_std: np.ndarray, rm_state_mean: np.ndarray, rm_state_std: np.ndarray,
            cfg: PPOv2RLHFConfig, device: torch.device, rm_coef: float):
    obs = env.reset(seed=env.config.random_seed + np.random.randint(0, 10_000))
    buffers = {
        "states": [],
        "price_actions": [],
        "reb_actions": [],
        "mask_price": [],
        "mask_reb": [],
        "old_logp": [],
        "ref_logp": [],
        "values": [],
        "env_rewards": [],
        "platform_revenue": [],
        "arrivals": [],
        "timeouts": [],
        "empty_cost": [],
        "rebalance_km": [],
        "dones": [],
    }
    seq_states_rm = []
    seq_price_rm = []
    seq_reb_rm = []
    reb_dim_mask = torch.from_numpy(getattr(env, "rebalance_mask", env.adj_mask).astype(np.float32)).to(device)

    for _ in range(cfg.rollout_steps):
        s_raw = flatten_obs(obs)
        s_np = normalize_state(s_raw, state_mean, state_std)
        s = torch.from_numpy(s_np).unsqueeze(0).to(device)
        is_price = 1.0 if (env.t % env.config.pricing_interval_steps == 0) else 0.0
        is_reb = 1.0 if (env.t % env.config.rebalance_interval_steps == 0) else 0.0
        # Phase-aware masks: avoid penalizing/fitting frozen action branches.
        if cfg.train_phase == "price":
            is_reb = 0.0
        elif cfg.train_phase == "rebalance":
            is_price = 0.0

        with torch.no_grad():
            p_a, r_a = actor.sample_actions(s, deterministic=False)
            v = value_net(s)

        r_exec_np = project_rebalance_action(env, r_a.squeeze(0).cpu().numpy())
        r_exec = torch.from_numpy(r_exec_np).unsqueeze(0).to(device)

        with torch.no_grad():
            # PPO probability ratio / KL must be computed on the sampled action from policy
            # distribution (pre-projection). Environment execution still uses projected action.
            p_lp, r_lp, _, _ = actor.logprob_of_actions(s, p_a, r_a, reb_dim_mask=reb_dim_mask)
            p_lp_ref, r_lp_ref, _, _ = actor_ref.logprob_of_actions(s, p_a, r_a, reb_dim_mask=reb_dim_mask)

        action = {}
        if is_price > 0.5:
            action["price"] = p_a.squeeze(0).cpu().numpy().astype(int)
        if is_reb > 0.5:
            action["rebalance"] = r_exec_np.astype(np.float32)

        next_obs, _, done, info = env.step(action)

        # for RM sequence: price at this step from action if updated, else current obs price_idx
        if is_price > 0.5:
            p_rm = action["price"]
        else:
            p_rm = obs["price_idx"]
        p_rm = np.clip(np.asarray(p_rm, dtype=np.int64), 0, rm_n_price_levels - 1)
        r_rm = action.get("rebalance", np.zeros((env.n, env.n), dtype=np.float32))

        if rm_state_mean is not None and rm_state_std is not None:
            s_rm_np = normalize_state(s_raw, rm_state_mean, rm_state_std)
        else:
            # Fallback for backward compatibility if RM ckpt has no normalization stats.
            s_rm_np = s_np
        seq_states_rm.append(torch.from_numpy(s_rm_np).to(device))
        seq_price_rm.append(torch.from_numpy(p_rm).to(device))
        seq_reb_rm.append(torch.from_numpy(np.asarray(r_rm, dtype=np.float32)).to(device))

        old_lp = is_price * p_lp.squeeze(0) + is_reb * r_lp.squeeze(0)
        ref_lp = is_price * p_lp_ref.squeeze(0) + is_reb * r_lp_ref.squeeze(0)

        buffers["states"].append(s.squeeze(0))
        buffers["price_actions"].append(p_a.squeeze(0))
        buffers["reb_actions"].append(r_a.squeeze(0))
        buffers["mask_price"].append(torch.tensor(is_price, device=device))
        buffers["mask_reb"].append(torch.tensor(is_reb, device=device))
        buffers["old_logp"].append(old_lp)
        buffers["ref_logp"].append(ref_lp)
        buffers["values"].append(v.squeeze(0))
        buffers["env_rewards"].append(torch.tensor(float(info["reward"]), device=device))
        buffers["platform_revenue"].append(torch.tensor(float(info.get("platform_revenue", info.get("revenue", 0.0))), device=device))
        buffers["arrivals"].append(torch.tensor(float(info.get("arrivals", 0.0)), device=device))
        buffers["timeouts"].append(torch.tensor(float(info.get("timeout", 0.0)), device=device))
        buffers["empty_cost"].append(torch.tensor(float(info.get("empty_cost", 0.0)), device=device))
        buffers["rebalance_km"].append(torch.tensor(float(info.get("rebalance_km", 0.0)), device=device))
        buffers["dones"].append(torch.tensor(1.0 if done else 0.0, device=device))

        obs = next_obs
        if done:
            break

    T = len(buffers["states"])
    for k in buffers:
        buffers[k] = torch.stack(buffers[k], dim=0)
        
    # 1) Compute joint KL, then convert to mean-KL for coefficient compatibility.
    n_price_dims = buffers["price_actions"].shape[-1]
    n_reb_dims = (
        reb_dim_mask.sum().item()
        if reb_dim_mask is not None
        else (buffers["reb_actions"].shape[-1] * buffers["reb_actions"].shape[-2])
    )
    total_dims = max(1.0, float(n_price_dims + n_reb_dims))

    kl_joint = buffers["old_logp"] - buffers["ref_logp"]
    kl_mean = kl_joint / total_dims
    scaled_env_rewards = buffers["env_rewards"] / cfg.reward_scale
    rewards = cfg.env_reward_weight * scaled_env_rewards - cfg.kl_coef * kl_mean
    # RM interval delta shaping (credit assignment every rm_interval_steps).
    with torch.no_grad():
        s_rm = torch.stack(seq_states_rm, dim=0).unsqueeze(0)  # [1,T,D]
        p_rm = torch.stack(seq_price_rm, dim=0).unsqueeze(0)   # [1,T,N]
        r_rm = torch.stack(seq_reb_rm, dim=0).unsqueeze(0)     # [1,T,N,N]
        interval = max(1, int(cfg.rm_interval_steps))
        rm_prev = torch.tensor(0.0, device=device)
        rm_score = torch.tensor(0.0, device=device)
        rm_delta_sum = torch.tensor(0.0, device=device)
        last_end = 0
        for end in range(interval, T + 1, interval):
            m_rm = torch.ones((1, end), device=device)
            rm_cur = rm_model(s_rm[:, :end], p_rm[:, :end], r_rm[:, :end], m_rm).squeeze(0)
            delta = rm_cur - rm_prev
            if cfg.rm_delta_clip > 0:
                delta = torch.clamp(delta, -cfg.rm_delta_clip, cfg.rm_delta_clip)
            seg_len = max(1, end - last_end)
            if cfg.rm_credit_mode == "segment":
                rewards[last_end:end] = rewards[last_end:end] + (rm_coef * delta / seg_len)
            else:
                rewards[end - 1] = rewards[end - 1] + rm_coef * delta
            rm_delta_sum += delta
            rm_prev = rm_cur
            rm_score = rm_cur
            last_end = end
        if last_end < T:
            m_rm = torch.ones((1, T), device=device)
            rm_cur = rm_model(s_rm, p_rm, r_rm, m_rm).squeeze(0)
            delta = rm_cur - rm_prev
            if cfg.rm_delta_clip > 0:
                delta = torch.clamp(delta, -cfg.rm_delta_clip, cfg.rm_delta_clip)
            seg_len = max(1, T - last_end)
            if cfg.rm_credit_mode == "segment":
                rewards[last_end:T] = rewards[last_end:T] + (rm_coef * delta / seg_len)
            else:
                rewards[-1] = rewards[-1] + rm_coef * delta
            rm_delta_sum += delta
            rm_score = rm_cur

        # Optional small terminal anchor.
        if cfg.rm_terminal_coef > 0:
            rewards[-1] = rewards[-1] + cfg.rm_terminal_coef * rm_coef * rm_score

    adv, ret = compute_gae(
        rewards=rewards,
        values=buffers["values"].detach(),
        dones=buffers["dones"],
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
    )
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # Prevent a few extreme steps from dominating PPO updates.
    adv = torch.clamp(adv, -4.0, 4.0)

    rollout_info = {
        "T": T,
        "reward_sum": float(rewards.sum().item()),
        "scaled_env_reward_sum": float(scaled_env_rewards.sum().item()),
        "env_reward_weight": float(cfg.env_reward_weight),
        "env_reward_sum": float(buffers["env_rewards"].sum().item()),
        "platform_revenue_sum": float(buffers["platform_revenue"].sum().item()),
        "arrivals_sum": float(buffers["arrivals"].sum().item()),
        "timeout_sum": float(buffers["timeouts"].sum().item()),
        "timeout_rate": float(
            buffers["timeouts"].sum().item() / max(buffers["arrivals"].sum().item(), 1e-8)
        ),
        "empty_cost_sum": float(buffers["empty_cost"].sum().item()),
        "rebalance_km_sum": float(buffers["rebalance_km"].sum().item()),
        "rm_score": float(rm_score.item()),
        "rm_delta_sum": float(rm_delta_sum.item()),
        "rm_coef": float(rm_coef),
        "rm_term_abs": float(abs(rm_coef * rm_delta_sum).item() if torch.is_tensor(rm_delta_sum) else abs(rm_coef * rm_delta_sum)),
        "rm_to_env_abs_ratio": float(
            (abs(rm_coef * rm_delta_sum).item() if torch.is_tensor(rm_delta_sum) else abs(rm_coef * rm_delta_sum))
            / (abs(float(scaled_env_rewards.sum().item())) + 1e-8)
        ),
        "kl_mean": float(kl_mean.mean().item()),
        "reward_scale": float(cfg.reward_scale),
    }
    batch = {
        **buffers,
        "rewards": rewards.detach(),
        "advantages": adv.detach(),
        "returns": ret.detach(),
        "reb_dim_mask": reb_dim_mask.detach(),
    }
    return batch, rollout_info


def ppo_update(batch, actor: PPOActor, value_net: ValueNetwork, optim_actor, optim_value, cfg: PPOv2RLHFConfig, device, update_actor: bool = True):
    states = batch["states"].to(device)
    p_act = batch["price_actions"].to(device).long()
    r_act = batch["reb_actions"].to(device)
    old_logp = batch["old_logp"].to(device)
    adv = batch["advantages"].to(device)
    ret = batch["returns"].to(device)
    m_p = batch["mask_price"].to(device)
    m_r = batch["mask_reb"].to(device)
    reb_dim_mask = batch["reb_dim_mask"].to(device)

    actor_loss_last = 0.0
    value_loss_last = 0.0
    entropy_last = 0.0

    for ep in range(cfg.ppo_epochs):
        p_lp_new, r_lp_new, ent_p, ent_r = actor.logprob_of_actions(
            states, p_act, r_act, reb_dim_mask=reb_dim_mask
        )
        logp_new = m_p * p_lp_new + m_r * r_lp_new
        ratio = torch.exp(logp_new - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv

        # Only steps with actual decisions should contribute to policy and entropy objectives.
        valid_mask = (m_p + m_r > 0).float()
        valid_steps = valid_mask.sum().clamp(min=1.0)
        # KL fuse at PPO-epoch granularity.
        if update_actor and cfg.target_kl > 0:
            n_price_dims = p_act.shape[-1]
            n_reb_dims = (
                reb_dim_mask.sum().item()
                if reb_dim_mask is not None
                else (r_act.shape[-1] * r_act.shape[-2])
            )
            total_dims = max(1.0, float(n_price_dims + n_reb_dims))
            
            # Convert joint KL to mean-KL for comparison with target_kl.
            approx_kl_joint = ((old_logp - logp_new) * valid_mask).sum() / valid_steps
            approx_kl_mean = approx_kl_joint / total_dims
            
            if float(approx_kl_mean.item()) > (1.5 * float(cfg.target_kl)):
                print(
                    f"[PPO] Early stopping at epoch {ep + 1}/{cfg.ppo_epochs} due to KL "
                    f"({float(approx_kl_mean.item()):.4f} > {1.5 * float(cfg.target_kl):.4f})."
                )
                break

        policy_loss = -(torch.min(surr1, surr2) * valid_mask).sum() / valid_steps
        entropy_bonus = (m_p * ent_p + m_r * ent_r).sum() / valid_steps
        actor_loss = policy_loss - cfg.entropy_coef * entropy_bonus

        v = value_net(states)
        # 1. Standard MSE value loss.
        v_loss_unclipped = (v - ret) ** 2
        # 2. Clipped value prediction around old values.
        old_v = batch["values"].to(device)
        v_clipped = old_v + torch.clamp(v - old_v, -cfg.clip_ratio, cfg.clip_ratio)
        v_loss_clipped = (v_clipped - ret) ** 2
        # 3. Pessimistic estimate to avoid overly large value updates.
        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        if update_actor:
            optim_actor.zero_grad()
            actor_loss.backward()
            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            optim_actor.step()

        optim_value.zero_grad()
        (cfg.value_coef * value_loss).backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(value_net.parameters(), cfg.max_grad_norm)
        optim_value.step()

        actor_loss_last = float(actor_loss.item())
        value_loss_last = float(value_loss.item())
        entropy_last = float(entropy_bonus.item())

    return actor_loss_last, value_loss_last, entropy_last


def build_actor_optimizer(actor: PPOActor, cfg: PPOv2RLHFConfig):
    # Separate LR by head: price head learns faster, rebalance head slower.
    base = actor.base_actor
    param_groups = []
    shared_params = [p for p in base.shared_net.parameters() if p.requires_grad]
    price_params = [p for p in base.price_head.parameters() if p.requires_grad]
    reb_params = [p for p in base.rebalance_head.parameters() if p.requires_grad]
    log_std_params = [actor.log_std] if actor.log_std.requires_grad else []
    if shared_params:
        param_groups.append({"params": shared_params, "lr": cfg.actor_lr * cfg.actor_lr_shared_mult})
    if price_params:
        param_groups.append({"params": price_params, "lr": cfg.actor_lr * cfg.actor_lr_price_mult})
    if reb_params:
        param_groups.append({"params": reb_params, "lr": cfg.actor_lr * cfg.actor_lr_rebalance_mult})
    if log_std_params:
        param_groups.append({"params": log_std_params, "lr": cfg.actor_lr * cfg.actor_lr_rebalance_mult})
    if len(param_groups) == 0:
        raise ValueError("No trainable actor parameters after applying train-phase/freeze settings.")
    return torch.optim.Adam(param_groups)


def main():
    parser = argparse.ArgumentParser(description="Train PPOv2-style RLHF policy for hybrid fulfillment.")
    parser.add_argument("--dataset", type=str, default="allarea_set_hybrid7_p12.pkl")
    parser.add_argument("--pi-init", type=str, default="checkpoints_rlhf/pi_init.pt")
    parser.add_argument("--pi-ref", type=str, default="checkpoints_rlhf/pi_ref.pt")
    parser.add_argument("--rm-ckpt", type=str, default="checkpoints_rm/rm_best.pt")
    parser.add_argument("--save-dir", type=str, default="checkpoints_ppo_rlhf")
    parser.add_argument("--total-updates", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=5e-5)
    parser.add_argument("--actor-lr-shared-mult", type=float, default=1.0)
    parser.add_argument("--actor-lr-price-mult", type=float, default=2.0)
    parser.add_argument("--actor-lr-rebalance-mult", type=float, default=1.0)
    parser.add_argument("--critic-warmup-updates", type=int, default=0, help="Only update value net in first N updates.")
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef-min", type=float, default=0.005)
    parser.add_argument("--entropy-decay-start", type=int, default=500)
    parser.add_argument("--entropy-decay-rate", type=float, default=0.999)
    parser.add_argument("--rebalance-std", type=float, default=2.0)
    parser.add_argument("--rebalance-std-final", type=float, default=0.5)
    parser.add_argument("--rebalance-std-decay-start", type=int, default=600)
    parser.add_argument("--rebalance-std-decay-end", type=int, default=1200)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=100.0, help="0 disables adaptive KL.")
    parser.add_argument("--kl-adapt-rate", type=float, default=1.5, help="Multiplier for adaptive KL.")
    parser.add_argument("--kl-coef-min", type=float, default=1e-4)
    parser.add_argument("--kl-coef-max", type=float, default=0.05)
    parser.add_argument("--rm-coef", type=float, default=0.05)
    parser.add_argument("--rm-coef-start", type=float, default=0.05, help="RM coef at update=1.")
    parser.add_argument("--rm-warmup-updates", type=int, default=0, help="Linear warmup steps from rm-coef-start to rm-coef.")
    parser.add_argument("--rm-interval-steps", type=int, default=6, help="Interval steps for RM delta shaping.")
    parser.add_argument("--rm-credit-mode", type=str, default="segment", choices=["segment", "endpoint"])
    parser.add_argument("--rm-delta-clip", type=float, default=5.0, help="Clip each RM delta for stability; <=0 disables clip.")
    parser.add_argument("--rm-terminal-coef", type=float, default=0.0, help="Optional terminal RM anchor coefficient.")
    parser.add_argument("--env-reward-weight", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument(
        "--init-mode",
        type=str,
        default="full",
        choices=["full", "half"],
        help="Actor init mode: full=load BC fully, half=keep backbone and reset policy heads.",
    )
    parser.add_argument(
        "--init-reb-bias",
        type=float,
        default=1.0,
        help="Initial bias for rebalance head last linear layer (positive => force early dispatch).",
    )
    parser.add_argument(
        "--train-phase",
        type=str,
        default="joint",
        choices=["joint", "price", "rebalance"],
        help="joint=update all heads; price=update pricing head only; rebalance=update rebalance branch only.",
    )
    parser.add_argument("--seed", type=int, default=20240908)
    parser.add_argument("--max-steps", type=int, default=288)
    args = parser.parse_args()

    cfg = PPOv2RLHFConfig(
        seed=args.seed,
        total_updates=args.total_updates,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        actor_lr=args.actor_lr,
        actor_lr_shared_mult=args.actor_lr_shared_mult,
        actor_lr_price_mult=args.actor_lr_price_mult,
        actor_lr_rebalance_mult=args.actor_lr_rebalance_mult,
        critic_warmup_updates=args.critic_warmup_updates,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
        entropy_coef_min=args.entropy_coef_min,
        entropy_decay_start=args.entropy_decay_start,
        entropy_decay_rate=args.entropy_decay_rate,
        rebalance_std=args.rebalance_std,
        rebalance_std_final=args.rebalance_std_final,
        rebalance_std_decay_start=args.rebalance_std_decay_start,
        rebalance_std_decay_end=args.rebalance_std_decay_end,
        train_phase=args.train_phase,
        kl_coef=args.kl_coef,
        target_kl=args.target_kl,
        kl_adapt_rate=args.kl_adapt_rate,
        kl_coef_min=args.kl_coef_min,
        kl_coef_max=args.kl_coef_max,
        rm_coef=args.rm_coef,
        rm_coef_start=args.rm_coef_start,
        rm_warmup_updates=args.rm_warmup_updates,
        rm_interval_steps=args.rm_interval_steps,
        rm_credit_mode=args.rm_credit_mode,
        rm_delta_clip=args.rm_delta_clip,
        rm_terminal_coef=args.rm_terminal_coef,
        env_reward_weight=args.env_reward_weight,
        reward_scale=args.reward_scale,
        clip_ratio=args.clip_ratio,
    )
    if cfg.rm_coef < 0 or cfg.rm_coef_start < 0:
        raise ValueError("rm-coef and rm-coef-start must be non-negative.")
    if cfg.rm_coef == 0.0 and cfg.rm_coef_start == 0.0:
        print("[Warn] RM coefficient is zero. Training will ignore RM and run as pure PPO.")
    # Keep exploration schedule consistent with annealing (high -> low std).
    if cfg.rebalance_std_final > cfg.rebalance_std:
        print(
            f"[Warn] rebalance-std-final ({cfg.rebalance_std_final:.4f}) > rebalance-std ({cfg.rebalance_std:.4f}). "
            "Clamping final std to start std to avoid increasing exploration later in training."
        )
        cfg.rebalance_std_final = cfg.rebalance_std
    set_seed(cfg.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env
    env_cfg = EnvConfig(max_steps=args.max_steps, random_seed=args.seed)
    env = HybridFulfillmentEnv(args.dataset, env_cfg)

    # actor init / ref
    base_init, init_ckpt, s_mean, s_std = build_actor_from_ckpt(args.pi_init, device)
    n_share_levels = len(getattr(env.config, "cs_share_levels", ()))
    actor_price_levels = int(getattr(base_init, "n_price_levels", 0))
    if n_share_levels > 0 and actor_price_levels != n_share_levels:
        raise ValueError(
            f"Actor n_price_levels={actor_price_levels} != env cs_share_levels={n_share_levels}. "
            "Please retrain BC/SFT or align env share levels."
        )
    if args.init_mode == "half":
        apply_half_cold_start(base_init, init_reb_bias=args.init_reb_bias)
    base_ref, _, _, _ = build_actor_from_ckpt(args.pi_ref, device)
    actor = PPOActor(base_init, rebalance_std=cfg.rebalance_std).to(device)
    actor_ref = PPOActor(base_ref, rebalance_std=cfg.rebalance_std).to(device)
    actor_ref.eval()
    for p in actor_ref.parameters():
        p.requires_grad_(False)
    apply_train_phase(actor, cfg.train_phase)

    # rm
    rm_model, rm_ckpt, rm_state_mean, rm_state_std = build_rm_from_ckpt(args.rm_ckpt, device)
    rm_n_price_levels = int(rm_ckpt["n_price_levels"])

    # value
    state_dim = init_ckpt.get("state_dim", None)
    if state_dim is None:
        if isinstance(s_mean, np.ndarray) and s_mean.ndim == 1:
            state_dim = int(s_mean.shape[0])
        else:
            state_dim = int(base_init.shared_net[0].in_features)
    state_dim = int(state_dim)
    value_net = ValueNetwork(state_dim=state_dim, hidden_dim=256).to(device)

    optim_actor = build_actor_optimizer(actor, cfg)
    optim_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)
    from torch.optim.lr_scheduler import LinearLR
    scheduler_actor = LinearLR(optim_actor, start_factor=1.0, end_factor=0.0, total_iters=cfg.total_updates)
    scheduler_value = LinearLR(optim_value, start_factor=1.0, end_factor=0.0, total_iters=cfg.total_updates)

    print(
        f"[Init] init_mode={args.init_mode} train_phase={cfg.train_phase} kl_coef={cfg.kl_coef:.5f} target_kl={cfg.target_kl:.3f} "
        f"rm_coef={cfg.rm_coef:.3f} rm_coef_start={cfg.rm_coef_start:.3f} rm_warmup_updates={cfg.rm_warmup_updates} "
        f"reward_scale={cfg.reward_scale:.1f} "
        f"actor_lr={cfg.actor_lr:.2e} "
        f"(shared x{cfg.actor_lr_shared_mult:.2f}, price x{cfg.actor_lr_price_mult:.2f}, reb x{cfg.actor_lr_rebalance_mult:.2f})"
    )

    best_env_reward = -1e18
    for upd in range(1, cfg.total_updates + 1):
        # Schedules: force early exploration, then gradually stabilize.
        cur_reb_std = scheduled_rebalance_std(cfg, upd)
        actor.set_rebalance_std(cur_reb_std)
        # Keep reference policy std on the same scale; otherwise KL is dominated
        # by distribution temperature mismatch instead of policy drift.
        actor_ref.set_rebalance_std(cur_reb_std)

        if upd > cfg.entropy_decay_start:
            cfg.entropy_coef = max(cfg.entropy_coef_min, cfg.entropy_coef * cfg.entropy_decay_rate)

        if cfg.rm_warmup_updates > 0 and upd <= cfg.rm_warmup_updates:
            frac = upd / max(1, cfg.rm_warmup_updates)
            cur_rm_coef = cfg.rm_coef_start + frac * (cfg.rm_coef - cfg.rm_coef_start)
        else:
            cur_rm_coef = cfg.rm_coef

        batch, info_roll = rollout(
            env=env,
            actor=actor,
            actor_ref=actor_ref,
            value_net=value_net,
            rm_model=rm_model,
            rm_n_price_levels=rm_n_price_levels,
            state_mean=s_mean,
            state_std=s_std,
            rm_state_mean=rm_state_mean,
            rm_state_std=rm_state_std,
            cfg=cfg,
            device=device,
            rm_coef=cur_rm_coef,
        )
        actor_loss, value_loss, ent = ppo_update(
            batch=batch,
            actor=actor,
            value_net=value_net,
            optim_actor=optim_actor,
            optim_value=optim_value,
            cfg=cfg,
            device=device,
            update_actor=(upd > cfg.critic_warmup_updates),
        )

        # Optional adaptive KL control to avoid both policy collapse and over-constraint.
        if cfg.target_kl > 0 and cfg.kl_coef > 0:
            kl_now = abs(info_roll["kl_mean"])
            if kl_now > cfg.target_kl * 1.5:
                cfg.kl_coef *= cfg.kl_adapt_rate
            elif kl_now < cfg.target_kl / 1.5:
                cfg.kl_coef /= cfg.kl_adapt_rate
            cfg.kl_coef = float(np.clip(cfg.kl_coef, cfg.kl_coef_min, cfg.kl_coef_max))

        print(
            f"[Update {upd:04d}] T={info_roll['T']} "
            f"reward_sum={info_roll['reward_sum']:.2f} "
            f"env_sum_raw={info_roll['env_reward_sum']:.2f} env_sum_scaled={info_roll['scaled_env_reward_sum']:.4f} "
            f"platform_rev={info_roll['platform_revenue_sum']:.2f} "
            f"timeout_rate={info_roll['timeout_rate']:.4f} "
            f"empty_cost_sum={info_roll['empty_cost_sum']:.2f} rebalance_km_sum={info_roll['rebalance_km_sum']:.2f} "
            f"rm={info_roll['rm_score']:.3f} rm_delta_sum={info_roll['rm_delta_sum']:.3f} rm_coef={cur_rm_coef:.3f} "
            f"rm/env_abs={info_roll['rm_to_env_abs_ratio']:.3f} "
            f"kl={info_roll['kl_mean']:.4f} kl_coef={cfg.kl_coef:.5f} "
            f"actor_loss={actor_loss:.4f} value_loss={value_loss:.4f} entropy={ent:.4f} "
            f"entropy_coef={cfg.entropy_coef:.5f} reb_std={cur_reb_std:.4f} "
            f"actor_update={int(upd > cfg.critic_warmup_updates)}"
        )
        scheduler_actor.step()
        scheduler_value.step()

        # save periodic + best
        ckpt = {
            "actor_state_dict": actor.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "state_mean": s_mean,
            "state_std": s_std,
            "config": vars(cfg),
            "update": upd,
            "rollout_info": info_roll,
        }
        if info_roll["env_reward_sum"] > best_env_reward:
            best_env_reward = info_roll["env_reward_sum"]
            torch.save(ckpt, os.path.join(args.save_dir, "ppo_rlhf_best.pt"))
        if upd % cfg.save_every == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f"ppo_rlhf_{upd:04d}.pt"))

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "state_mean": s_mean,
            "state_std": s_std,
            "config": vars(cfg),
            "update": cfg.total_updates,
        },
        os.path.join(args.save_dir, "ppo_rlhf_last.pt"),
    )
    print("Training finished.")
    print(f"Best rollout env_reward_sum={best_env_reward:.2f}")


if __name__ == "__main__":
    main()

