import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from env_hybrid import EnvConfig, HybridFulfillmentEnv
from collect_sft_data import expert_action as sft_expert_action
from train_bc import BCActor
from train_ppov2_rlhf import PPOActor


def gini_coefficient(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = np.maximum(arr, 0.0)
    if arr.size == 0 or np.allclose(arr.sum(), 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


def count_price_whiplash(price_hist: np.ndarray) -> int:
    # price_hist: [K, N], K is number of pricing decisions.
    if price_hist.ndim != 2 or price_hist.shape[0] < 3:
        return 0
    prev = price_hist[:-2]
    cur = price_hist[1:-1]
    nxt = price_hist[2:]
    v_shape = ((cur > prev) & (cur > nxt)) | ((cur < prev) & (cur < nxt))
    return int(np.sum(v_shape))


def build_remote_region_mask(env: HybridFulfillmentEnv) -> np.ndarray:
    n = int(env.n)
    if n <= 1:
        return np.ones((max(n, 1),), dtype=bool)
    km = np.asarray(getattr(env, "travel_km", np.zeros((n, n), dtype=float)), dtype=float)
    mask_offdiag = (~np.eye(n, dtype=bool)).astype(float)
    avg_km = (km * mask_offdiag).sum(axis=1) / np.maximum(mask_offdiag.sum(axis=1), 1.0)
    base_mass = np.asarray(getattr(env, "base_mass", np.ones((n,), dtype=float)), dtype=float).reshape(-1)
    if base_mass.shape[0] != n:
        base_mass = np.ones((n,), dtype=float)
    far_thr = float(np.quantile(avg_km, 0.65))
    low_mass_thr = float(np.quantile(base_mass, 0.35))
    remote = (avg_km >= far_thr) | (base_mass <= low_mass_thr)
    if not bool(np.any(remote)):
        remote[int(np.argmax(avg_km))] = True
    return remote.astype(bool)


def max_consecutive_true(x: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in x.astype(bool).tolist():
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def build_peak_slot_mask(env: HybridFulfillmentEnv, quantile: float = 0.70) -> np.ndarray:
    q = float(np.clip(quantile, 0.0, 1.0))
    vals = np.array([float(env._period_factor(s)) for s in range(env.period_num)], dtype=float)
    thr = float(np.quantile(vals, q))
    return (vals >= thr).astype(bool)


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


def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((state - mean) / std).astype(np.float32)


@dataclass
class EpisodeResult:
    name: str
    total_reward: float
    arrivals: float
    timeout: float
    timeout_rate: float
    avg_imbalance: float
    demand_weighted_util: float
    empty_cost_sum: float
    rebalance_km_sum: float
    rebalance_units_sum: float
    avg_rebalance_km_per_unit: float
    service_gini: float
    price_whiplash_count: int
    price_changes: int
    remote_cs_stress_hours: float
    remote_cs_stress_max_streak_hours: float
    remote_cs_share_high_hours: float
    remote_service_rate: float
    remote_timeout_rate: float
    remote_av_service_share: float
    core_service_rate: float
    core_timeout_rate: float
    peak_timeout_rate: float
    offpeak_timeout_rate: float
    timeouts_series: List[float]
    av_units_series: List[List[float]]


class BCPolicy:
    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32).reshape(-1)
        self.state_std = np.asarray(ckpt["state_std"], dtype=np.float32).reshape(-1)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)

        self.model = BCActor(
            state_dim=int(ckpt["state_dim"]),
            n_region=int(ckpt["n_region"]),
            n_price_levels=int(ckpt["n_price_levels"]),
            hidden_dim=int(ckpt["config"]["hidden_dim"]),
            use_region_attn=bool(ckpt.get("config", {}).get("use_region_attn", True)),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def act(self, env: HybridFulfillmentEnv, obs: dict) -> Dict:
        x = normalize_state(flatten_obs(obs), self.state_mean, self.state_std)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        p_logits, r_pred = self.model(xt)
        action = {}
        if env.t % env.config.pricing_interval_steps == 0:
            action["price"] = p_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)
        if env.t % env.config.rebalance_interval_steps == 0:
            mat = r_pred.squeeze(0).cpu().numpy().astype(np.float32)
            mat = np.maximum(mat, 0.0)
            mat = np.where(getattr(env, "rebalance_mask", env.adj_mask), mat, 0.0)
            np.fill_diagonal(mat, 0.0)
            action["rebalance"] = mat
        return action


class PPOPolicy:
    def __init__(self, ppo_ckpt_path: str, pi_init_ckpt_path: str, device: torch.device):
        # Build architecture from pi_init ckpt, then load PPO weights.
        init_ckpt = torch.load(pi_init_ckpt_path, map_location=device)
        ppo_ckpt = torch.load(ppo_ckpt_path, map_location=device)
        self.state_mean = np.asarray(ppo_ckpt["state_mean"], dtype=np.float32).reshape(-1)
        self.state_std = np.asarray(ppo_ckpt["state_std"], dtype=np.float32).reshape(-1)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)

        base = BCActor(
            state_dim=int(init_ckpt["state_dim"]),
            n_region=int(init_ckpt["n_region"]),
            n_price_levels=int(init_ckpt["n_price_levels"]),
            hidden_dim=int(init_ckpt["config"]["hidden_dim"]),
            use_region_attn=bool(init_ckpt.get("config", {}).get("use_region_attn", True)),
        ).to(device)
        self.model = PPOActor(base_actor=base, rebalance_std=0.10).to(device)
        self.model.load_state_dict(ppo_ckpt["actor_state_dict"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def act(self, env: HybridFulfillmentEnv, obs: dict) -> Dict:
        x = normalize_state(flatten_obs(obs), self.state_mean, self.state_std)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        # Deterministic evaluation to avoid stochastic policy noise in comparisons.
        p_act, r_act = self.model.sample_actions(xt, deterministic=True)
        action = {}
        if env.t % env.config.pricing_interval_steps == 0:
            action["price"] = p_act.squeeze(0).cpu().numpy().astype(int)
        if env.t % env.config.rebalance_interval_steps == 0:
            mat = r_act.squeeze(0).cpu().numpy().astype(np.float32)
            mat = np.maximum(mat, 0.0)
            mat = np.where(getattr(env, "rebalance_mask", env.adj_mask), mat, 0.0)
            np.fill_diagonal(mat, 0.0)
            action["rebalance"] = mat
        return action


class DPOPolicy:
    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32).reshape(-1)
        self.state_std = np.asarray(ckpt["state_std"], dtype=np.float32).reshape(-1)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)

        self.model = BCActor(
            state_dim=int(ckpt["state_dim"]),
            n_region=int(ckpt["n_region"]),
            n_price_levels=int(ckpt["n_price_levels"]),
            hidden_dim=int(ckpt["config"]["hidden_dim"]),
            use_region_attn=bool(ckpt.get("config", {}).get("use_region_attn", True)),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def act(self, env: HybridFulfillmentEnv, obs: dict) -> Dict:
        x = normalize_state(flatten_obs(obs), self.state_mean, self.state_std)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        p_logits, r_pred = self.model(xt)
        action = {}
        if env.t % env.config.pricing_interval_steps == 0:
            action["price"] = p_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)
        if env.t % env.config.rebalance_interval_steps == 0:
            mat = r_pred.squeeze(0).cpu().numpy().astype(np.float32)
            mat = np.maximum(mat, 0.0)
            mat = np.where(getattr(env, "rebalance_mask", env.adj_mask), mat, 0.0)
            np.fill_diagonal(mat, 0.0)
            action["rebalance"] = mat
        return action


class NoDispatchPolicy:
    def __init__(self, n_regions: int, fixed_price_idx: int = 1):
        self.n_regions = n_regions
        self.fixed_price_idx = fixed_price_idx

    def act(self, env: HybridFulfillmentEnv, obs: dict) -> Dict:
        action = {}
        if env.t % env.config.pricing_interval_steps == 0:
            action["price"] = np.full((self.n_regions,), self.fixed_price_idx, dtype=int)
        if env.t % env.config.rebalance_interval_steps == 0:
            action["rebalance"] = np.zeros((self.n_regions, self.n_regions), dtype=np.float32)
        return action


class RulePolicy:
    """Use the exact same heuristic expert as SFT data collection."""

    def __init__(self, n_regions: int):
        self.n_regions = n_regions

    def act(self, env: HybridFulfillmentEnv, obs: dict) -> Dict:
        return sft_expert_action(env, obs)


def run_one_day(
    env: HybridFulfillmentEnv,
    policy,
    name: str,
    seed: int,
    ethics_workload_threshold: float = 1.2,
    ethics_cs_share_threshold: float = 0.65,
    peak_slot_quantile: float = 0.70,
    stress_mode: str = "strict",
    debug_stress: bool = False,
):
    obs = env.reset(seed=seed)
    done = False
    arrivals = 0.0
    timeout = 0.0
    total_reward = 0.0
    empty_cost_sum = 0.0
    rebalance_km_sum = 0.0
    rebalance_units_sum = 0.0
    imbalances = []
    demand_utils = []
    timeouts_series = []
    av_units_series = []
    price_changes = 0
    last_price = None
    price_hist = []
    demand_by_origin = np.zeros(env.n, dtype=np.float64)
    served_by_origin = np.zeros(env.n, dtype=np.float64)
    timeout_by_origin_sum = np.zeros(env.n, dtype=np.float64)
    served_av_by_origin_sum = np.zeros(env.n, dtype=np.float64)
    served_cs_by_origin_sum = np.zeros(env.n, dtype=np.float64)
    peak_arrivals = 0.0
    peak_timeout = 0.0
    offpeak_arrivals = 0.0
    offpeak_timeout = 0.0
    workload_raw_hist = []
    cs_share_fallback_hist = []
    served_av_hist = []
    served_cs_hist = []
    has_origin_service_breakdown = False
    remote_mask = build_remote_region_mask(env)
    peak_slot_mask = build_peak_slot_mask(env, quantile=peak_slot_quantile)

    while not done:
        slot_now = int(obs.get("slot", 0))
        is_peak = bool(slot_now >= 0 and slot_now < len(peak_slot_mask) and peak_slot_mask[slot_now])
        workload_raw_hist.append(
            np.asarray(obs.get("workload_raw", obs.get("workload", np.zeros(env.n))), dtype=float)
            .reshape(-1)
            .copy()
        )
        av_now = np.asarray(obs.get("av_units", np.zeros(env.n)), dtype=float).reshape(-1)
        cs_now = np.asarray(obs.get("cs_units", np.zeros(env.n)), dtype=float).reshape(-1)
        cs_share_fallback_hist.append(cs_now / np.maximum(av_now + cs_now, 1e-8))
        action = policy.act(env, obs)
        # Debug surge multiplier decisions at actual pricing boundaries.
        if name.startswith("PPO") and "price" in action and env.t % env.config.pricing_interval_steps == 0:
            print(f"[Debug] PPO Surge Action @ step {env.t}: {action['price']}")
        if "price" in action:
            cur = np.asarray(action["price"])
            if last_price is not None and not np.array_equal(cur, last_price):
                price_changes += 1
            last_price = cur.copy()
            price_hist.append(cur.copy())

        obs, r, done, info = env.step(action)
        total_reward += float(r)
        arrivals += float(info["arrivals"])
        timeout += float(info["timeout"])
        if is_peak:
            peak_arrivals += float(info["arrivals"])
            peak_timeout += float(info["timeout"])
        else:
            offpeak_arrivals += float(info["arrivals"])
            offpeak_timeout += float(info["timeout"])
        empty_cost_sum += float(info["empty_cost"])
        rebalance_km_sum += float(info.get("rebalance_km", 0.0))
        rebalance_units_sum += float(info.get("rebalance_units", 0.0))
        imbalances.append(float(info["imbalance"]))
        demand_utils.append(float(info["demand_weighted_util"]))
        timeouts_series.append(float(info["timeout"]))
        av_units_series.append(obs["av_units"].astype(float).tolist())
        arr_by_origin = np.asarray(info.get("arrivals_by_origin", np.zeros(env.n, dtype=float)), dtype=float).reshape(-1)
        srv_by_origin = np.asarray(info.get("served_by_origin", np.zeros(env.n, dtype=float)), dtype=float).reshape(-1)
        to_by_origin = np.asarray(info.get("timeout_by_origin", np.zeros(env.n, dtype=float)), dtype=float).reshape(-1)
        srv_av_by_origin = np.asarray(info.get("served_av_by_origin", np.zeros(env.n, dtype=float)), dtype=float).reshape(-1)
        srv_cs_by_origin = np.asarray(info.get("served_cs_by_origin", np.zeros(env.n, dtype=float)), dtype=float).reshape(-1)
        if arr_by_origin.shape[0] == env.n:
            demand_by_origin += np.maximum(arr_by_origin, 0.0)
        if srv_by_origin.shape[0] == env.n:
            served_by_origin += np.maximum(srv_by_origin, 0.0)
        if to_by_origin.shape[0] == env.n:
            timeout_by_origin_sum += np.maximum(to_by_origin, 0.0)
        if srv_av_by_origin.shape[0] == env.n:
            served_av_hist.append(np.maximum(srv_av_by_origin, 0.0))
            served_av_by_origin_sum += np.maximum(srv_av_by_origin, 0.0)
            has_origin_service_breakdown = True
        else:
            served_av_hist.append(np.zeros(env.n, dtype=float))
        if srv_cs_by_origin.shape[0] == env.n:
            served_cs_hist.append(np.maximum(srv_cs_by_origin, 0.0))
            served_cs_by_origin_sum += np.maximum(srv_cs_by_origin, 0.0)
            has_origin_service_breakdown = True
        else:
            served_cs_hist.append(np.zeros(env.n, dtype=float))

    timeout_rate = timeout / max(arrivals, 1e-8)
    price_hist_arr = np.asarray(price_hist, dtype=int) if len(price_hist) > 0 else np.zeros((0, env.n), dtype=int)
    price_whiplash_count = count_price_whiplash(price_hist_arr)
    service_rates = served_by_origin / np.maximum(demand_by_origin, 1.0)
    service_gini = gini_coefficient(service_rates)
    avg_rebalance_km_per_unit = rebalance_km_sum / max(rebalance_units_sum, 1e-8)
    remote_idx = np.asarray(remote_mask, dtype=bool).reshape(-1)
    remote_arrivals = float(np.sum(demand_by_origin[remote_idx]))
    remote_served = float(np.sum(served_by_origin[remote_idx]))
    remote_timeout = float(np.sum(timeout_by_origin_sum[remote_idx]))
    remote_served_av = float(np.sum(served_av_by_origin_sum[remote_idx]))
    remote_served_cs = float(np.sum(served_cs_by_origin_sum[remote_idx]))
    remote_service_rate = remote_served / max(remote_arrivals, 1e-8)
    remote_timeout_rate = remote_timeout / max(remote_arrivals, 1e-8)
    remote_av_service_share = remote_served_av / max(remote_served_av + remote_served_cs, 1e-8)
    core_idx = ~remote_idx
    core_arrivals = float(np.sum(demand_by_origin[core_idx]))
    core_served = float(np.sum(served_by_origin[core_idx]))
    core_timeout = float(np.sum(timeout_by_origin_sum[core_idx]))
    core_service_rate = core_served / max(core_arrivals, 1e-8)
    core_timeout_rate = core_timeout / max(core_arrivals, 1e-8)
    peak_timeout_rate = peak_timeout / max(peak_arrivals, 1e-8)
    offpeak_timeout_rate = offpeak_timeout / max(offpeak_arrivals, 1e-8)
    workload_raw_arr = np.asarray(workload_raw_hist, dtype=float) if len(workload_raw_hist) > 0 else np.zeros((0, env.n), dtype=float)
    cs_share_fallback_arr = (
        np.asarray(cs_share_fallback_hist, dtype=float)
        if len(cs_share_fallback_hist) > 0
        else np.zeros((0, env.n), dtype=float)
    )
    served_av_arr = np.asarray(served_av_hist, dtype=float) if len(served_av_hist) > 0 else np.zeros((0, env.n), dtype=float)
    served_cs_arr = np.asarray(served_cs_hist, dtype=float) if len(served_cs_hist) > 0 else np.zeros((0, env.n), dtype=float)
    if workload_raw_arr.shape[0] > 0:
        if has_origin_service_breakdown:
            cs_share = served_cs_arr / np.maximum(served_av_arr + served_cs_arr, 1e-8)
            cs_share_source = "served_by_origin"
        else:
            cs_share = cs_share_fallback_arr
            cs_share_source = "units_fallback"
        remote_grid = np.broadcast_to(remote_mask.reshape(1, -1), workload_raw_arr.shape)
        cond_w = (workload_raw_arr >= float(ethics_workload_threshold))
        cond_cs = (cs_share >= float(ethics_cs_share_threshold))
        if stress_mode == "relaxed":
            # Relaxed mode: focus on remote overload directly.
            stress = cond_w & remote_grid
        else:
            # Strict mode: remote overload plus high realized CS service share.
            stress = cond_w & cond_cs & remote_grid
        stress_steps = int(np.sum(stress))
        cs_share_high_steps = int(np.sum(cond_cs & remote_grid))
        step_hours = float(getattr(env.config, "step_minutes", 5)) / 60.0
        max_streak_steps = 0
        for j in range(stress.shape[1]):
            s = max_consecutive_true(stress[:, j])
            if s > max_streak_steps:
                max_streak_steps = s
        stress_hours = float(stress_steps) * step_hours
        max_streak_hours = float(max_streak_steps) * step_hours
        cs_share_high_hours = float(cs_share_high_steps) * step_hours
        if debug_stress:
            total_cells = int(workload_raw_arr.size)
            print(
                f"[StressDebug][{name}] seed={seed} "
                f"mode={stress_mode} "
                f"cs_source={cs_share_source} "
                f"w_raw(max={float(np.max(workload_raw_arr)):.4f}, p95={float(np.quantile(workload_raw_arr, 0.95)):.4f}) "
                f"cs_share(max={float(np.max(cs_share)):.4f}, p95={float(np.quantile(cs_share, 0.95)):.4f}) "
                f"hits_workload={int(np.sum(cond_w))}/{total_cells} "
                f"hits_cs_share={int(np.sum(cond_cs))}/{total_cells} "
                f"hits_remote={int(np.sum(remote_grid))}/{total_cells} "
                f"hits_joint={int(np.sum(stress))}/{total_cells} "
                f"hits_remote_cs_share={int(np.sum(cond_cs & remote_grid))}/{total_cells}"
            )
    else:
        stress_hours = 0.0
        max_streak_hours = 0.0
        cs_share_high_hours = 0.0
        if debug_stress:
            print(f"[StressDebug][{name}] seed={seed} no workload observations.")

    return EpisodeResult(
        name=name,
        total_reward=float(total_reward),
        arrivals=float(arrivals),
        timeout=float(timeout),
        timeout_rate=float(timeout_rate),
        avg_imbalance=float(np.mean(imbalances)),
        demand_weighted_util=float(np.mean(demand_utils) if len(demand_utils) > 0 else 0.0),
        empty_cost_sum=float(empty_cost_sum),
        rebalance_km_sum=float(rebalance_km_sum),
        rebalance_units_sum=float(rebalance_units_sum),
        avg_rebalance_km_per_unit=float(avg_rebalance_km_per_unit),
        service_gini=float(service_gini),
        price_whiplash_count=int(price_whiplash_count),
        price_changes=int(price_changes),
        remote_cs_stress_hours=float(stress_hours),
        remote_cs_stress_max_streak_hours=float(max_streak_hours),
        remote_cs_share_high_hours=float(cs_share_high_hours),
        remote_service_rate=float(remote_service_rate),
        remote_timeout_rate=float(remote_timeout_rate),
        remote_av_service_share=float(remote_av_service_share),
        core_service_rate=float(core_service_rate),
        core_timeout_rate=float(core_timeout_rate),
        peak_timeout_rate=float(peak_timeout_rate),
        offpeak_timeout_rate=float(offpeak_timeout_rate),
        timeouts_series=timeouts_series,
        av_units_series=av_units_series,
    )


def save_plots(results: List[EpisodeResult], out_dir: str, region_ids: List[int], suffix: str = ""):
    os.makedirs(out_dir, exist_ok=True)

    # Timeout series plot
    plt.figure(figsize=(11, 4))
    for r in results:
        plt.plot(r.timeouts_series, label=r.name, linewidth=1.8)
    plt.title("Timeout Orders Over One Day")
    plt.xlabel("Step (5 min)")
    plt.ylabel("Timeout Orders")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"timeouts_over_time{suffix}.png"), dpi=150)
    plt.close()

    # AV units distribution by region (day average)
    x = np.arange(len(region_ids))
    width = 0.25
    plt.figure(figsize=(11, 4))
    for i, r in enumerate(results):
        av = np.array(r.av_units_series, dtype=float)  # [T, N]
        avg_by_region = av.mean(axis=0)
        plt.bar(x + (i - 1) * width, avg_by_region, width=width, label=r.name)
    plt.xticks(x, [str(i) for i in region_ids])
    plt.xlabel("Region ID")
    plt.ylabel("Average AV Units")
    plt.title("Average AV Units by Region")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"av_units_by_region{suffix}.png"), dpi=150)
    plt.close()


def print_table(results: List[EpisodeResult]):
    print("\n[Policy Comparison - One Day]")
    for r in results:
        print(
            f"{r.name}: total_reward={r.total_reward:.2f}, timeout_rate={r.timeout_rate:.4f}, "
            f"avg_imbalance={r.avg_imbalance:.4f}, demand_weighted_util={r.demand_weighted_util:.4f}, "
            f"empty_cost_sum={r.empty_cost_sum:.2f}, avg_rebalance_km_per_unit={r.avg_rebalance_km_per_unit:.4f}, "
            f"service_gini={r.service_gini:.4f}, price_whiplash_count={r.price_whiplash_count}, "
            f"price_changes={r.price_changes}, remote_cs_stress_hours={r.remote_cs_stress_hours:.2f}, "
            f"remote_cs_stress_max_streak_hours={r.remote_cs_stress_max_streak_hours:.2f}, "
            f"remote_cs_share_high_hours={r.remote_cs_share_high_hours:.2f}, "
            f"remote_service_rate={r.remote_service_rate:.4f}, remote_timeout_rate={r.remote_timeout_rate:.4f}, "
            f"remote_av_service_share={r.remote_av_service_share:.4f}, "
            f"core_service_rate={r.core_service_rate:.4f}, core_timeout_rate={r.core_timeout_rate:.4f}, "
            f"peak_timeout_rate={r.peak_timeout_rate:.4f}, offpeak_timeout_rate={r.offpeak_timeout_rate:.4f}"
        )


def summarize_policy(policy_runs: List[EpisodeResult]):
    arr = {
        "total_reward": [x.total_reward for x in policy_runs],
        "timeout_rate": [x.timeout_rate for x in policy_runs],
        "avg_imbalance": [x.avg_imbalance for x in policy_runs],
        "demand_weighted_util": [x.demand_weighted_util for x in policy_runs],
        "empty_cost_sum": [x.empty_cost_sum for x in policy_runs],
        "rebalance_km_sum": [x.rebalance_km_sum for x in policy_runs],
        "rebalance_units_sum": [x.rebalance_units_sum for x in policy_runs],
        "avg_rebalance_km_per_unit": [x.avg_rebalance_km_per_unit for x in policy_runs],
        "service_gini": [x.service_gini for x in policy_runs],
        "price_whiplash_count": [x.price_whiplash_count for x in policy_runs],
        "price_changes": [x.price_changes for x in policy_runs],
        "remote_cs_stress_hours": [x.remote_cs_stress_hours for x in policy_runs],
        "remote_cs_stress_max_streak_hours": [x.remote_cs_stress_max_streak_hours for x in policy_runs],
        "remote_cs_share_high_hours": [x.remote_cs_share_high_hours for x in policy_runs],
        "remote_service_rate": [x.remote_service_rate for x in policy_runs],
        "remote_timeout_rate": [x.remote_timeout_rate for x in policy_runs],
        "remote_av_service_share": [x.remote_av_service_share for x in policy_runs],
        "core_service_rate": [x.core_service_rate for x in policy_runs],
        "core_timeout_rate": [x.core_timeout_rate for x in policy_runs],
        "peak_timeout_rate": [x.peak_timeout_rate for x in policy_runs],
        "offpeak_timeout_rate": [x.offpeak_timeout_rate for x in policy_runs],
        "arrivals": [x.arrivals for x in policy_runs],
    }
    out = {}
    for k, v in arr.items():
        vv = np.array(v, dtype=float)
        out[k] = {"mean": float(vv.mean()), "std": float(vv.std())}
    return out


def print_summary_table(summary: Dict[str, dict], baseline: str = "BC_Clone"):
    print("\n[Policy Comparison - Multi Seed Mean+/-Std]")
    keys = [
        "timeout_rate",
        "avg_imbalance",
        "demand_weighted_util",
        "empty_cost_sum",
        "avg_rebalance_km_per_unit",
        "service_gini",
        "price_whiplash_count",
        "price_changes",
        "remote_cs_stress_hours",
        "remote_cs_stress_max_streak_hours",
        "remote_cs_share_high_hours",
        "remote_service_rate",
        "remote_timeout_rate",
        "remote_av_service_share",
        "core_service_rate",
        "core_timeout_rate",
        "peak_timeout_rate",
        "offpeak_timeout_rate",
        "total_reward",
    ]
    for name, sm in summary.items():
        print(
            f"{name}: "
            + ", ".join([f"{k}={sm[k]['mean']:.4f}+/-{sm[k]['std']:.4f}" for k in keys])
        )
    if baseline in summary:
        b = summary[baseline]
        print("\n[Relative To BC_Clone]")
        for name, sm in summary.items():
            if name == baseline:
                continue
            for k in keys:
                denom = abs(b[k]["mean"]) if abs(b[k]["mean"]) > 1e-8 else 1.0
                diff = (sm[k]["mean"] - b[k]["mean"]) / denom * 100.0
                print(f"{name} vs {baseline} | {k}: {diff:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare BC vs PPO-RLHF vs No-Dispatch on one-day simulation.")
    parser.add_argument("--dataset", type=str, default="allarea_set_hybrid7_p12.pkl")
    parser.add_argument("--pi-init-ckpt", type=str, default="checkpoints_rlhf/pi_init.pt")
    parser.add_argument("--ppo-ckpt", type=str, default="", help="Backward-compatible single PPO ckpt.")
    parser.add_argument("--dpo-ckpt", type=str, default="", help="Backward-compatible DPO ckpt.")
    parser.add_argument("--ppo-only-ckpt", type=str, default="", help="PPO-only checkpoint path.")
    parser.add_argument("--ppo-rm-ckpt", type=str, default="", help="PPO+RM checkpoint path.")
    parser.add_argument("--ppo-dpo-ckpt", type=str, default="", help="PPO+DPO checkpoint path (use dpo_for_eval.pt).")
    parser.add_argument("--seed", type=int, default=20240908)
    parser.add_argument("--num-seeds", type=int, default=5, help="Run multiple seeds for robust comparison.")
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--rule-penalty-enable", type=int, default=0, help="1 to enable hard rule penalties in env reward.")
    parser.add_argument("--price-change-penalty-coef", type=float, default=0.0)
    parser.add_argument("--imbalance-penalty-coef", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="reports_compare")
    parser.add_argument("--save-json", type=str, default="reports_compare/compare_metrics.json")
    parser.add_argument("--ethics-workload-threshold", type=float, default=1.2)
    parser.add_argument("--ethics-cs-share-threshold", type=float, default=0.65)
    parser.add_argument("--peak-slot-quantile", type=float, default=0.70, help="Top quantile of demand period factor treated as peak.")
    parser.add_argument("--stress-mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--debug-stress", type=int, default=0, help="1 to print stress-condition diagnostics.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = EnvConfig(max_steps=args.max_steps, random_seed=args.seed)
    env_cfg.rule_penalty_enable = bool(args.rule_penalty_enable)
    env_cfg.price_change_penalty_coef = float(args.price_change_penalty_coef)
    env_cfg.imbalance_penalty_coef = float(args.imbalance_penalty_coef)
    env = HybridFulfillmentEnv(args.dataset, env_cfg)

    bc_policy = BCPolicy(args.pi_init_ckpt, device=device)
    rule_policy = RulePolicy(n_regions=env.n)
    no_dispatch = NoDispatchPolicy(n_regions=env.n, fixed_price_idx=1)

    policy_defs = [
        ("Rule_Heuristic", rule_policy),
        ("BC_Clone", bc_policy),
    ]

    # New explicit three-way PPO comparison.
    if str(args.ppo_only_ckpt).strip():
        policy_defs.append(("PPO_Only", PPOPolicy(args.ppo_only_ckpt, args.pi_init_ckpt, device=device)))
    if str(args.ppo_rm_ckpt).strip():
        policy_defs.append(("PPO_RM", PPOPolicy(args.ppo_rm_ckpt, args.pi_init_ckpt, device=device)))
    if str(args.ppo_dpo_ckpt).strip():
        policy_defs.append(("PPO_DPO", PPOPolicy(args.ppo_dpo_ckpt, args.pi_init_ckpt, device=device)))

    # Backward compatibility with old arguments.
    if str(args.ppo_ckpt).strip() and not any(k == "PPO_RLHF" for k, _ in policy_defs):
        policy_defs.append(("PPO_RLHF", PPOPolicy(args.ppo_ckpt, args.pi_init_ckpt, device=device)))
    if str(args.dpo_ckpt).strip() and not any(k == "DPO" for k, _ in policy_defs):
        policy_defs.append(("DPO", DPOPolicy(args.dpo_ckpt, device=device)))

    policy_defs.append(("No_Dispatch", no_dispatch))

    all_runs = {name: [] for name, _ in policy_defs}
    for i in range(args.num_seeds):
        seed_i = args.seed + i
        for name, policy in policy_defs:
            all_runs[name].append(
                run_one_day(
                    env,
                    policy,
                    name,
                    seed=seed_i,
                    ethics_workload_threshold=args.ethics_workload_threshold,
                    ethics_cs_share_threshold=args.ethics_cs_share_threshold,
                    peak_slot_quantile=args.peak_slot_quantile,
                    stress_mode=args.stress_mode,
                    debug_stress=bool(args.debug_stress),
                )
            )
        print(f"Finished seed {seed_i} ({i + 1}/{args.num_seeds})")

    # Use first seed runs for figure style consistency, plus summary for robust decision.
    order = [name for name, _ in policy_defs]
    first_seed_results = [all_runs[k][0] for k in order]
    summary = {k: summarize_policy(v) for k, v in all_runs.items()}

    print_table(first_seed_results)
    print_summary_table(summary, baseline="BC_Clone")
    save_plots(first_seed_results, args.out_dir, env.region_ids)

    payload = {
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "max_steps": args.max_steps,
        "first_seed_results": [r.__dict__ for r in first_seed_results],
        "summary": summary,
        "all_runs": {k: [x.__dict__ for x in v] for k, v in all_runs.items()},
    }
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics JSON: {args.save_json}")
    print(f"Saved plots in: {args.out_dir}")


if __name__ == "__main__":
    main()
