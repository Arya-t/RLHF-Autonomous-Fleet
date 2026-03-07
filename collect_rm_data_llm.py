import argparse
import copy
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from env_hybrid import EnvConfig, HybridFulfillmentEnv
from train_bc import BCActor


@dataclass
class CollectConfig:
    pairs_to_collect: int = 500
    rollout_steps: int = 48
    save_path: str = "rm_dataset.pkl"
    pref_jsonl_path: str = "preference_pairs.jsonl"
    noise_a: float = 0.05
    noise_b: float = 0.15
    min_reward_diff_ratio: float = 0.005
    min_timeout_diff: float = 0.02
    min_demand_util_diff: float = 0.01
    min_km_per_unit_diff: float = 0.20
    warmup_max_steps: int = 60
    max_attempts: int = 0
    log_every: int = 10
    mode: str = "rule"
    qwen_model: str = "qwen-plus"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    reward_margin_ratio: float = 0.02
    timeout_margin_pp: float = 0.2
    timeout_filter_pp: float = 1.0
    gini_margin: float = 0.02
    km_margin: float = 0.20
    util_margin: float = 0.01
    label_objective: str = "residual"  # residual: ignore total_reward, focus on soft preferences.
    seed: int = 20240908
    hard_neg_sampling_prob: float = 0.35
    hard_neg_noise: float = 0.30
    hard_neg_timeout_delta: float = 0.5
    hard_neg_gini_delta: float = 0.01
    hard_neg_whiplash_delta: float = 1.0
    hard_neg_km_delta: float = 0.20
    hard_neg_reward_margin_ratio: float = 0.03
    rebuild_hard_neg_target_ratio: float = 0.40
    ethics_workload_threshold: float = 1.2
    ethics_cs_share_threshold: float = 0.65
    ethics_max_streak_steps: int = 48  # 4 hours with 5-min steps
    ethics_hours_threshold: float = 6.0
    gray_reward_margin_ratio: float = 0.03
    gray_timeout_margin_pp: float = 0.5


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


def normalize_state(state: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray) -> np.ndarray:
    return ((state - state_mean) / state_std).astype(np.float32)


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
    # price_hist: [K, N], K is number of surge pricing decisions.
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


def has_ethics_risk(summary: dict, cfg: CollectConfig) -> bool:
    return bool(
        float(summary.get("remote_cs_stress_hours", 0.0)) >= float(cfg.ethics_hours_threshold)
        or int(summary.get("remote_cs_stress_max_streak_steps", 0)) >= int(cfg.ethics_max_streak_steps)
    )


class BCPolicy:
    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32).reshape(-1)
        self.state_std = np.asarray(ckpt["state_std"], dtype=np.float32).reshape(-1)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)
        self.device = device
        state_dict = ckpt.get("model_state_dict", None)
        if state_dict is None:
            state_dict = ckpt.get("actor_state_dict", None)
        if state_dict is None:
            raise KeyError(
                f"{ckpt_path} does not contain 'model_state_dict' or 'actor_state_dict'."
            )

        # PPO checkpoints save actor params with "base_actor." prefix and include "log_std".
        if any(k.startswith("base_actor.") for k in state_dict.keys()):
            state_dict = {
                (k[len("base_actor."):] if k.startswith("base_actor.") else k): v
                for k, v in state_dict.items()
            }
        if "log_std" in state_dict:
            state_dict.pop("log_std")

        def _cfg_get(name, default=None):
            cfg = ckpt.get("config", {})
            if isinstance(cfg, dict) and name in cfg:
                return cfg[name]
            return default

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
                f"Cannot infer {missing} from checkpoint: {ckpt_path}. "
                "Please provide a BC/PPO actor checkpoint with metadata."
            )

        self.n_region = int(n_region)
        self.n_price_levels = int(n_price_levels)
        self.model = BCActor(
            state_dim=int(state_dim),
            n_region=self.n_region,
            n_price_levels=self.n_price_levels,
            hidden_dim=int(hidden_dim),
            use_region_attn=bool(_cfg_get("use_region_attn", True)),
        ).to(device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def act(self, env: HybridFulfillmentEnv, obs: dict, noise_scale: float = 0.1) -> Dict:
        state = flatten_obs(obs)
        state_n = normalize_state(state, self.state_mean, self.state_std)
        xt = torch.from_numpy(state_n).unsqueeze(0).to(self.device)
        p_logits, reb_pred = self.model(xt)

        action = {}
        if env.t % env.config.pricing_interval_steps == 0:
            if env.rng.random() < noise_scale:
                p_action = env.rng.integers(0, self.n_price_levels, size=(env.n,), dtype=int)
            else:
                p_action = p_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)
            action["price"] = p_action

        if env.t % env.config.rebalance_interval_steps == 0:
            r_action = reb_pred.squeeze(0).cpu().numpy().astype(np.float32)
            r_action += env.rng.normal(0.0, noise_scale * 5.0, size=r_action.shape).astype(np.float32)
            r_action = np.maximum(r_action, 0.0)
            r_action = np.where(getattr(env, "rebalance_mask", env.adj_mask), r_action, 0.0)
            np.fill_diagonal(r_action, 0.0)
            action["rebalance"] = r_action
        return action


def query_qwen_preference(
    client,
    model: str,
    metrics_a: dict,
    metrics_b: dict,
    reward_margin_ratio: float = 0.02,
    timeout_margin_pp: float = 0.2,
    gini_margin: float = 0.02,
    km_margin: float = 0.20,
    util_margin: float = 0.01,
    label_objective: str = "residual",
):
    prompt = f"""你是大型出行/外卖平台的“信任与安全委员会”主席。
现在有两个策略 A 和 B，它们已经通过基础门槛筛选（利润和超时差距都不大）。
请你从【用户信任度】和【众包骑手福祉】角度，选择更优策略。

重点审查（按优先级）：
1) 抵制价格刺客：price_whiplash_count 越低越好（价格不应剧烈锯齿波动）。
2) 防止偏远区域 CS 长时间硬扛：remote_cs_stress_max_streak_hours、remote_cs_stress_hours 越低越好。
3) 若上述体感与福祉指标接近，再参考 timeout_rate（越低越好）和 total_reward（越高越好）。

请输出严格 JSON：
{{"thought_process":"...","preferred_trajectory":"A或B"}}

策略A（结构化指标）:
total_reward={metrics_a.get('total_reward', 0.0):.2f}
timeout_rate={metrics_a.get('timeout_rate', 0.0):.2f}%
price_whiplash_count={metrics_a.get('price_whiplash_count', 0)}
remote_cs_stress_hours={metrics_a.get('remote_cs_stress_hours', 0.0):.2f}
remote_cs_stress_max_streak_hours={metrics_a.get('remote_cs_stress_max_streak_hours', 0.0):.2f}

策略B（结构化指标）:
total_reward={metrics_b.get('total_reward', 0.0):.2f}
timeout_rate={metrics_b.get('timeout_rate', 0.0):.2f}%
price_whiplash_count={metrics_b.get('price_whiplash_count', 0)}
remote_cs_stress_hours={metrics_b.get('remote_cs_stress_hours', 0.0):.2f}
remote_cs_stress_max_streak_hours={metrics_b.get('remote_cs_stress_max_streak_hours', 0.0):.2f}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格输出 JSON 的助手。"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    pref = str(obj.get("preferred_trajectory", "A")).strip().upper()
    if pref not in ("A", "B"):
        pref = "A"
    return pref, obj


def rule_preference(
    metrics_a: dict,
    metrics_b: dict,
    reward_margin_ratio: float = 0.02,
    timeout_margin_pp: float = 0.2,
    gini_margin: float = 0.02,
    km_margin: float = 0.20,
    util_margin: float = 0.01,
    label_objective: str = "residual",
):
    ra, rb = float(metrics_a.get("total_reward", 0.0)), float(metrics_b.get("total_reward", 0.0))
    if str(label_objective).lower() == "reward_first":
        reward_margin = reward_margin_ratio * max(abs(ra), abs(rb), 1.0)
        if abs(ra - rb) > reward_margin:
            pref = "A" if ra > rb else "B"
            return pref, {
                "thought_process": f"reward-priority(|delta_reward|={abs(ra-rb):.3f}>{reward_margin:.3f})",
                "preferred_trajectory": pref,
            }

    # Residual objective: prioritize soft constraints absent from env reward.
    if (ra < -1000.0) and (rb < -1000.0):
        pref = "A" if ra > rb else "B"
        return pref, {"thought_process": "survival_check", "preferred_trajectory": pref}

    sa = float(metrics_a.get("remote_cs_stress_max_streak_hours", 0.0))
    sb = float(metrics_b.get("remote_cs_stress_max_streak_hours", 0.0))
    if abs(sa - sb) >= 0.5:
        pref = "A" if sa < sb else "B"
        return pref, {"thought_process": "remote-cs-stress-priority", "preferred_trajectory": pref}

    wa, wb = float(metrics_a.get("price_whiplash_count", 0.0)), float(metrics_b.get("price_whiplash_count", 0.0))
    if abs(wa - wb) >= 1:
        pref = "A" if wa < wb else "B"
        return pref, {"thought_process": "whiplash-priority", "preferred_trajectory": pref}

    ga, gb = float(metrics_a.get("service_gini", 0.0)), float(metrics_b.get("service_gini", 0.0))
    if abs(ga - gb) > gini_margin:
        pref = "A" if ga < gb else "B"
        return pref, {"thought_process": f"fairness-priority(|diff|={abs(ga-gb):.3f})", "preferred_trajectory": pref}

    ka = float(metrics_a.get("avg_rebalance_km_per_unit", 0.0))
    kb = float(metrics_b.get("avg_rebalance_km_per_unit", 0.0))
    if abs(ka - kb) > km_margin:
        pref = "A" if ka < kb else "B"
        return pref, {"thought_process": "rebalance-efficiency-priority", "preferred_trajectory": pref}

    ta, tb = float(metrics_a["timeout_rate"]), float(metrics_b["timeout_rate"])
    dt = ta - tb
    if abs(dt) > timeout_margin_pp:
        pref = "A" if ta < tb else "B"
        return pref, {
            "thought_process": f"timeout-secondary(|delta_timeout|={abs(dt):.3f}pp>{timeout_margin_pp:.3f}pp)",
            "preferred_trajectory": pref,
        }

    ua, ub = float(metrics_a.get("demand_weighted_util", 0.0)), float(metrics_b.get("demand_weighted_util", 0.0))
    if abs(ua - ub) > util_margin:
        pref = "A" if ua > ub else "B"
        return pref, {"thought_process": "util-secondary", "preferred_trajectory": pref}

    # Final fallback: allow total_reward to break near ties.
    pref = "A" if ra >= rb else "B"
    return pref, {"thought_process": "near-tie fallback by total_reward", "preferred_trajectory": pref}


def rollout_trajectory(env: HybridFulfillmentEnv, policy: BCPolicy, cfg: CollectConfig, steps: int = 24, noise_scale: float = 0.1):
    traj = {"states": [], "prices": [], "rebalances": [], "price_mask": [], "rebalance_mask": [], "infos": []}
    metrics = {
        "arrivals": 0.0,
        "timeout": 0.0,
        "platform_revenue": 0.0,
        "demand_util_sum": 0.0,
        "empty_cost_sum": 0.0,
        "service_cost_sum": 0.0,
        "rebalance_km_sum": 0.0,
        "rebalance_units_sum": 0.0,
        "total_reward": 0.0,
    }
    price_history = []
    served_by_origin = np.zeros(env.n, dtype=np.float64)
    demand_by_origin = np.zeros(env.n, dtype=np.float64)
    remote_mask = build_remote_region_mask(env)
    workload_raw_hist = []
    av_hist = []
    cs_hist = []
    obs = env._get_obs()

    actual_steps = 0
    for _ in range(steps):
        workload_raw_hist.append(np.asarray(obs.get("workload_raw", obs.get("workload", np.zeros(env.n))), dtype=float))
        av_hist.append(np.asarray(obs.get("av_units", np.zeros(env.n)), dtype=float))
        cs_hist.append(np.asarray(obs.get("cs_units", np.zeros(env.n)), dtype=float))
        state_vec = flatten_obs(obs)
        traj["states"].append(state_vec.tolist())

        action = policy.act(env, obs, noise_scale=noise_scale)
        p_action = action.get("price", np.full(env.n, -1, dtype=int))
        r_action = action.get("rebalance", np.zeros((env.n, env.n), dtype=np.float32))
        traj["prices"].append(np.asarray(p_action).tolist())
        traj["rebalances"].append(np.asarray(r_action).tolist())
        traj["price_mask"].append(1.0 if "price" in action else 0.0)
        traj["rebalance_mask"].append(1.0 if "rebalance" in action else 0.0)

        if "price" in action:
            price_history.append(np.array(p_action, copy=True))

        obs, rew, done, info = env.step(action)

        traj["infos"].append(
            {
                "arrivals": float(info["arrivals"]),
                "timeout": float(info["timeout"]),
                "platform_revenue": float(info.get("platform_revenue", info.get("revenue", 0.0))),
                "demand_weighted_util": float(info["demand_weighted_util"]),
                "empty_cost": float(info.get("empty_cost", 0.0)),
                "rebalance_km": float(info.get("rebalance_km", 0.0)),
                "rebalance_units": float(info.get("rebalance_units", 0.0)),
                "service_gini": float(info.get("service_gini", 0.0)) if "service_gini" in info else 0.0,
            }
        )
        metrics["arrivals"] += float(info["arrivals"])
        metrics["timeout"] += float(info["timeout"])
        metrics["platform_revenue"] += float(info.get("platform_revenue", info.get("revenue", 0.0)))
        metrics["demand_util_sum"] += float(info["demand_weighted_util"])
        metrics["empty_cost_sum"] += float(info.get("empty_cost", 0.0))
        metrics["service_cost_sum"] += float(info.get("service_cost", 0.0))
        metrics["rebalance_km_sum"] += float(info.get("rebalance_km", 0.0))
        metrics["rebalance_units_sum"] += float(info.get("rebalance_units", 0.0))
        metrics["total_reward"] += float(rew)
        arr_by_origin = np.asarray(info.get("arrivals_by_origin", np.zeros(env.n, dtype=float)), dtype=float)
        srv_by_origin = np.asarray(info.get("served_by_origin", np.zeros(env.n, dtype=float)), dtype=float)
        if arr_by_origin.shape == (env.n,):
            demand_by_origin += np.maximum(arr_by_origin, 0.0)
        if srv_by_origin.shape == (env.n,):
            served_by_origin += np.maximum(srv_by_origin, 0.0)
        actual_steps += 1
        if done:
            break

    price_hist = np.asarray(price_history, dtype=int) if len(price_history) > 0 else np.zeros((0, env.n), dtype=int)
    workload_raw_arr = np.asarray(workload_raw_hist, dtype=float) if len(workload_raw_hist) > 0 else np.zeros((0, env.n), dtype=float)
    av_arr = np.asarray(av_hist, dtype=float) if len(av_hist) > 0 else np.zeros((0, env.n), dtype=float)
    cs_arr = np.asarray(cs_hist, dtype=float) if len(cs_hist) > 0 else np.zeros((0, env.n), dtype=float)
    if workload_raw_arr.shape[0] > 0:
        cs_share = cs_arr / np.maximum(av_arr + cs_arr, 1e-8)
        stress = (
            (workload_raw_arr >= float(env.config.workload_cap_floor))
            & (workload_raw_arr >= float(cfg.ethics_workload_threshold))
            & (cs_share >= float(cfg.ethics_cs_share_threshold))
        )
        stress = stress & remote_mask.reshape(1, -1)
        stress_steps = int(np.sum(stress))
        step_hours = float(getattr(env.config, "step_minutes", 5)) / 60.0
        max_streak_steps = 0
        for j in range(stress.shape[1]):
            s = max_consecutive_true(stress[:, j])
            if s > max_streak_steps:
                max_streak_steps = s
        stress_hours = float(stress_steps) * step_hours
        max_streak_hours = float(max_streak_steps) * step_hours
    else:
        stress_hours = 0.0
        max_streak_steps = 0
        max_streak_hours = 0.0

    summary = {
        "timeout_rate": (metrics["timeout"] / max(metrics["arrivals"], 1e-8)) * 100.0,
        "price_whiplash_count": int(count_price_whiplash(price_hist)),
        "demand_weighted_util": float(metrics["demand_util_sum"] / max(actual_steps, 1)),
        "platform_revenue": float(metrics["platform_revenue"]),
        "empty_cost_sum": float(metrics["empty_cost_sum"]),
        "service_cost_sum": float(metrics["service_cost_sum"]),
        "rebalance_km_sum": float(metrics["rebalance_km_sum"]),
        "avg_rebalance_km_per_unit": float(metrics["rebalance_km_sum"] / max(metrics["rebalance_units_sum"], 1e-8)),
        "empty_cost_ratio": float(metrics["empty_cost_sum"] / max(metrics["empty_cost_sum"] + metrics["service_cost_sum"], 1e-8)),
        "service_gini": float(
            gini_coefficient(served_by_origin / np.maximum(demand_by_origin, 1.0))
            if env.n > 0
            else 0.0
        ),
        "total_reward": float(metrics["total_reward"]),
        "arrivals": metrics["arrivals"],
        "timeout": metrics["timeout"],
        "steps": actual_steps,
        "remote_cs_stress_hours": stress_hours,
        "remote_cs_stress_max_streak_steps": int(max_streak_steps),
        "remote_cs_stress_max_streak_hours": max_streak_hours,
    }
    return traj, summary


def safe_clone_env(env: HybridFulfillmentEnv) -> HybridFulfillmentEnv:
    try:
        return copy.deepcopy(env)
    except Exception:
        new_env = HybridFulfillmentEnv.__new__(HybridFulfillmentEnv)
        new_env.__dict__ = copy.deepcopy(env.__dict__)
        return new_env


def _pair_reward_margin(a: dict, b: dict, ratio: float) -> float:
    ra = float(a.get("total_reward", 0.0))
    rb = float(b.get("total_reward", 0.0))
    return float(ratio * max(abs(ra), abs(rb), 1.0))


def is_hard_negative_pair(summary_a: dict, summary_b: dict, cfg: CollectConfig) -> bool:
    ra = float(summary_a.get("total_reward", 0.0))
    rb = float(summary_b.get("total_reward", 0.0))
    reward_gap = abs(ra - rb)
    reward_margin = _pair_reward_margin(summary_a, summary_b, cfg.hard_neg_reward_margin_ratio)
    if reward_gap < reward_margin:
        return False

    if ra > rb:
        hi, lo = summary_a, summary_b
    else:
        hi, lo = summary_b, summary_a

    timeout_worse = float(hi.get("timeout_rate", 0.0)) - float(lo.get("timeout_rate", 0.0)) >= cfg.hard_neg_timeout_delta
    gini_worse = float(hi.get("service_gini", 0.0)) - float(lo.get("service_gini", 0.0)) >= cfg.hard_neg_gini_delta
    whip_worse = float(hi.get("price_whiplash_count", 0.0)) - float(lo.get("price_whiplash_count", 0.0)) >= cfg.hard_neg_whiplash_delta
    km_worse = float(hi.get("avg_rebalance_km_per_unit", 0.0)) - float(lo.get("avg_rebalance_km_per_unit", 0.0)) >= cfg.hard_neg_km_delta

    # High-reward but at least one soft metric clearly worse => hard negative candidate.
    return bool(timeout_worse or gini_worse or whip_worse or km_worse)


def rebuild_dataset_with_hard_negatives(dataset: list, cfg: CollectConfig):
    if len(dataset) == 0:
        return dataset, {"hard_neg_before": 0, "hard_neg_after": 0}

    hard = [x for x in dataset if bool(x.get("is_hard_negative", False))]
    normal = [x for x in dataset if not bool(x.get("is_hard_negative", False))]
    target_hard = int(round(cfg.rebuild_hard_neg_target_ratio * len(dataset)))
    target_hard = max(0, min(target_hard, len(dataset)))

    keep_hard = hard[: min(len(hard), target_hard)]
    need_normal = len(dataset) - len(keep_hard)
    keep_normal = normal[: min(len(normal), need_normal)]

    # If hard samples are insufficient, backfill with remaining normal samples.
    if len(keep_hard) + len(keep_normal) < len(dataset):
        remain = len(dataset) - (len(keep_hard) + len(keep_normal))
        extra_normal = normal[len(keep_normal) : len(keep_normal) + remain]
        keep_normal.extend(extra_normal)

    rebuilt = keep_hard + keep_normal
    np.random.shuffle(rebuilt)
    return rebuilt, {
        "hard_neg_before": len(hard),
        "hard_neg_after": sum(1 for x in rebuilt if bool(x.get("is_hard_negative", False))),
    }


def collect_rm_dataset(env_base: HybridFulfillmentEnv, policy: BCPolicy, cfg: CollectConfig):
    dataset = []
    rebuild_stat = {"hard_neg_before": 0, "hard_neg_after": 0}
    np.random.seed(cfg.seed)
    attempts = 0
    filtered = 0
    api_errors = 0
    chosen_a = 0
    chosen_b = 0
    timeout_gap_gt_05 = 0
    timeout_gap_total = 0
    swap_count = 0
    hard_neg_accepted = 0

    client = None
    if cfg.mode == "qwen":
        from openai import OpenAI

        if not cfg.api_key:
            raise ValueError("Qwen mode requires API key.")
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.api_base)

    pref_jsonl = open(cfg.pref_jsonl_path, "w", encoding="utf-8")
    try:
        while len(dataset) < cfg.pairs_to_collect:
            attempts += 1
            if cfg.max_attempts > 0 and attempts > cfg.max_attempts:
                print(f"Stop at max_attempts={cfg.max_attempts}. accepted={len(dataset)}")
                break

            env_base.reset(seed=cfg.seed + len(dataset))
            warm_steps = np.random.randint(0, min(cfg.warmup_max_steps, env_base.config.max_steps))
            for _ in range(warm_steps):
                _, _, done, _ = env_base.step({})
                if done:
                    break

            env_a = safe_clone_env(env_base)
            env_b = safe_clone_env(env_base)
            env_a.rng = np.random.default_rng(cfg.seed + attempts * 17 + 1)
            env_b.rng = np.random.default_rng(cfg.seed + attempts * 17 + 2)

            # Adversarial sampling mode: enlarge one branch's exploration to trigger
            # "high-reward-but-soft-metrics-worse" cases.
            if bool(np.random.rand() < cfg.hard_neg_sampling_prob):
                if bool(np.random.rand() < 0.5):
                    noise_for_a, noise_for_b = cfg.noise_a, cfg.hard_neg_noise
                else:
                    noise_for_a, noise_for_b = cfg.hard_neg_noise, cfg.noise_a
            else:
                # Standard symmetric pairing.
                if bool(np.random.rand() < 0.5):
                    noise_for_a, noise_for_b = cfg.noise_a, cfg.noise_b
                else:
                    noise_for_a, noise_for_b = cfg.noise_b, cfg.noise_a

            traj_a, summary_a = rollout_trajectory(env_a, policy, cfg=cfg, steps=cfg.rollout_steps, noise_scale=noise_for_a)
            traj_b, summary_b = rollout_trajectory(env_b, policy, cfg=cfg, steps=cfg.rollout_steps, noise_scale=noise_for_b)

            whiplash_diff = abs(float(summary_a.get("price_whiplash_count", 0.0)) - float(summary_b.get("price_whiplash_count", 0.0)))
            gini_diff = abs(float(summary_a.get("service_gini", 0.0)) - float(summary_b.get("service_gini", 0.0)))
            timeout_diff_pp = abs(float(summary_a.get("timeout_rate", 0.0)) - float(summary_b.get("timeout_rate", 0.0)))
            km_diff = abs(float(summary_a.get("avg_rebalance_km_per_unit", 0.0)) - float(summary_b.get("avg_rebalance_km_per_unit", 0.0)))
            hard_negative_flag = is_hard_negative_pair(summary_a, summary_b, cfg)
            soft_gate = (
                whiplash_diff >= 1.0
                or gini_diff > cfg.gini_margin
                or timeout_diff_pp > cfg.timeout_filter_pp
                or km_diff > cfg.km_margin
            )
            if not (soft_gate or hard_negative_flag):
                filtered += 1
                if attempts % max(1, cfg.log_every) == 0:
                    print(
                        f"Progress attempts={attempts} accepted={len(dataset)} filtered={filtered} api_errors={api_errors}"
                    )
                continue

            try:
                # Randomly swap displayed A/B order to reduce position bias in preference labeling.
                swap_ab = bool(np.random.rand() < 0.5)
                if swap_ab:
                    swap_count += 1
                    shown_traj_a, shown_summary_a = traj_b, summary_b
                    shown_traj_b, shown_summary_b = traj_a, summary_a
                else:
                    shown_traj_a, shown_summary_a = traj_a, summary_a
                    shown_traj_b, shown_summary_b = traj_b, summary_b

                reward_gap = abs(float(summary_a.get("total_reward", 0.0)) - float(summary_b.get("total_reward", 0.0)))
                reward_hard_margin = _pair_reward_margin(summary_a, summary_b, cfg.reward_margin_ratio)
                hard_reward = reward_gap > reward_hard_margin
                hard_timeout = timeout_diff_pp > float(cfg.timeout_margin_pp)
                hard_rule = bool(hard_reward or hard_timeout)

                reward_close = reward_gap <= _pair_reward_margin(summary_a, summary_b, cfg.gray_reward_margin_ratio)
                timeout_close = timeout_diff_pp <= float(cfg.gray_timeout_margin_pp)
                ethics_gray = has_ethics_risk(summary_a, cfg) or has_ethics_risk(summary_b, cfg)
                use_llm = bool(cfg.mode == "qwen" and (not hard_rule) and (ethics_gray or (reward_close and timeout_close)))

                if hard_reward:
                    pref = "A" if float(summary_a.get("total_reward", 0.0)) >= float(summary_b.get("total_reward", 0.0)) else "B"
                    llm_meta = {
                        "thought_process": f"hard-rule-reward(|delta|={reward_gap:.3f}>{reward_hard_margin:.3f})",
                        "preferred_trajectory": pref,
                        "llm_bypassed": True,
                    }
                elif hard_timeout:
                    pref = "A" if float(summary_a.get("timeout_rate", 0.0)) <= float(summary_b.get("timeout_rate", 0.0)) else "B"
                    llm_meta = {
                        "thought_process": f"hard-rule-timeout(|delta|={timeout_diff_pp:.3f}pp>{float(cfg.timeout_margin_pp):.3f}pp)",
                        "preferred_trajectory": pref,
                        "llm_bypassed": True,
                    }
                elif use_llm:
                    pref, llm_meta = query_qwen_preference(
                        client,
                        cfg.qwen_model,
                        shown_summary_a,
                        shown_summary_b,
                        reward_margin_ratio=cfg.reward_margin_ratio,
                        timeout_margin_pp=cfg.timeout_margin_pp,
                        gini_margin=cfg.gini_margin,
                        km_margin=cfg.km_margin,
                        util_margin=cfg.util_margin,
                        label_objective=cfg.label_objective,
                    )
                else:
                    pref, llm_meta = rule_preference(
                        shown_summary_a,
                        shown_summary_b,
                        reward_margin_ratio=cfg.reward_margin_ratio,
                        timeout_margin_pp=cfg.timeout_margin_pp,
                        gini_margin=cfg.gini_margin,
                        km_margin=cfg.km_margin,
                        util_margin=cfg.util_margin,
                        label_objective=cfg.label_objective,
                    )
                    if cfg.mode == "qwen":
                        llm_meta = {
                            **llm_meta,
                            "llm_bypassed": True,
                            "llm_bypass_reason": (
                                "not_gray_zone_or_no_ethics_trigger"
                                if not ethics_gray
                                else "rule_override"
                            ),
                        }

                # Map displayed preference back to original trajectory order.
                if swap_ab:
                    # displayed A==orig B, displayed B==orig A
                    label = 0 if pref == "A" else 1
                else:
                    label = 1 if pref == "A" else 0
                if label == 1:
                    chosen_a += 1
                else:
                    chosen_b += 1
                timeout_gap_total += 1
                if abs(float(summary_a["timeout_rate"]) - float(summary_b["timeout_rate"])) > 0.5:
                    timeout_gap_gt_05 += 1

                print(
                    f"Pair {len(dataset)+1}: choose {'A' if label == 1 else 'B'} | "
                    f"A_reward={summary_a['total_reward']:.1f} vs B_reward={summary_b['total_reward']:.1f} | "
                    f"A_timeout={summary_a['timeout_rate']:.2f}% vs B_timeout={summary_b['timeout_rate']:.2f}% | "
                    f"A_util={summary_a.get('demand_weighted_util', 0.0):.4f} vs B_util={summary_b.get('demand_weighted_util', 0.0):.4f} | "
                    f"A_gini={summary_a.get('service_gini', 0.0):.4f} vs B_gini={summary_b.get('service_gini', 0.0):.4f} | "
                    f"A_km_per_unit={summary_a.get('avg_rebalance_km_per_unit', 0.0):.4f} vs B_km_per_unit={summary_b.get('avg_rebalance_km_per_unit', 0.0):.4f} | "
                    f"A_empty_ratio={summary_a.get('empty_cost_ratio', 0.0):.4f} vs B_empty_ratio={summary_b.get('empty_cost_ratio', 0.0):.4f}"
                )

                item = {
                    "traj_A": traj_a,
                    "traj_B": traj_b,
                    "preference_label": label,
                    "summary_A": summary_a,
                    "summary_B": summary_b,
                    "llm_meta": llm_meta,
                    "noise_A": float(noise_for_a),
                    "noise_B": float(noise_for_b),
                    "is_hard_negative": bool(hard_negative_flag),
                }
                dataset.append(item)
                if hard_negative_flag:
                    hard_neg_accepted += 1

                chosen = "A" if label == 1 else "B"
                rejected = "B" if chosen == "A" else "A"
                pref_jsonl.write(
                    json.dumps(
                        {
                            "pair_id": len(dataset) - 1,
                            "prompt": (
                                "Hard rules first: large reward gap => higher reward; large timeout gap => lower timeout. "
                                "Otherwise, gray-zone preference uses business-expert criteria: "
                                "price_whiplash_count and remote_cs_stress_* first, then timeout/reward tie-break."
                            ),
                            "chosen": f"{chosen}: {item['summary_A'] if chosen == 'A' else item['summary_B']}",
                            "rejected": f"{rejected}: {item['summary_B'] if chosen == 'A' else item['summary_A']}",
                            "summary_A": summary_a,
                            "summary_B": summary_b,
                            "noise_A": float(noise_for_a),
                            "noise_B": float(noise_for_b),
                            "label_meta": llm_meta,
                            "priority_config": {
                                "reward_margin_ratio": cfg.reward_margin_ratio,
                                "timeout_margin_pp": cfg.timeout_margin_pp,
                                "util_margin": cfg.util_margin,
                                "label_objective": cfg.label_objective,
                                "gray_reward_margin_ratio": cfg.gray_reward_margin_ratio,
                                "gray_timeout_margin_pp": cfg.gray_timeout_margin_pp,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if len(dataset) % 50 == 0:
                    with open(cfg.save_path, "wb") as f:
                        pickle.dump(dataset, f)
                    print(f"Checkpoint saved: {len(dataset)} pairs")
            except Exception as e:
                api_errors += 1
                print(f"Labeling error: {e}")

                if attempts % max(1, cfg.log_every) == 0:
                    ab_total = max(chosen_a + chosen_b, 1)
                    timeout_gap_ratio = timeout_gap_gt_05 / max(timeout_gap_total, 1)
                    print(
                        f"Progress attempts={attempts} accepted={len(dataset)} filtered={filtered} api_errors={api_errors} "
                        f"| timeout_gap>0.5pp_ratio={timeout_gap_ratio:.3f} "
                        f"| chooseA={chosen_a/ab_total:.3f} chooseB={chosen_b/ab_total:.3f} "
                        f"| swap_ratio={swap_count/max(timeout_gap_total,1):.3f} "
                        f"| hard_neg_ratio={hard_neg_accepted/max(len(dataset),1):.3f}"
                    )

        # Rebuild dataset to enforce hard-negative mix for RM robustness.
        dataset, rebuild_stat = rebuild_dataset_with_hard_negatives(dataset, cfg)
        with open(cfg.save_path, "wb") as f:
            pickle.dump(dataset, f)
    finally:
        pref_jsonl.close()
    ab_total = max(chosen_a + chosen_b, 1)
    timeout_gap_ratio = timeout_gap_gt_05 / max(timeout_gap_total, 1)
    print(
        f"Data collection complete! saved={cfg.save_path}, pref_jsonl={cfg.pref_jsonl_path}, "
        f"pairs={len(dataset)}, attempts={attempts}, filtered={filtered}, api_errors={api_errors}, "
        f"timeout_gap>0.5pp_ratio={timeout_gap_ratio:.3f}, chooseA={chosen_a/ab_total:.3f}, chooseB={chosen_b/ab_total:.3f}, "
        f"swap_ratio={swap_count/max(timeout_gap_total,1):.3f}, "
        f"hard_neg_before={rebuild_stat.get('hard_neg_before', 0)}, hard_neg_after={rebuild_stat.get('hard_neg_after', 0)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Collect RM preference data with BC policy + Qwen/rule judge.")
    parser.add_argument("--dataset", type=str, default="allarea_set_hybrid7_p12.pkl")
    parser.add_argument(
        "--bc-ckpt",
        type=str,
        default="checkpoints_bc/bc_actor_best.pt",
        help="Policy checkpoint path. Supports BC ckpt (model_state_dict) or PPO ckpt (actor_state_dict).",
    )
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=48, help="48 steps = 4 hours for 5-min step env.")
    parser.add_argument("--noise-a", type=float, default=0.05, help="Exploration noise for trajectory A.")
    parser.add_argument("--noise-b", type=float, default=0.15, help="Exploration noise for trajectory B.")
    parser.add_argument(
        "--min-reward-diff-ratio",
        type=float,
        default=0.005,
        help="Minimum |A-B total_reward| ratio to keep pair when timeout/util diffs are also small.",
    )
    parser.add_argument("--min-timeout-diff", type=float, default=0.02, help="Min timeout_rate diff (percentage points).")
    parser.add_argument("--min-demand-util-diff", type=float, default=0.01, help="Min demand_weighted_util diff.")
    parser.add_argument("--min-km-per-unit-diff", type=float, default=0.20, help="Min avg_rebalance_km_per_unit diff.")
    parser.add_argument("--warmup-max-steps", type=int, default=60, help="Max random warmup before branch.")
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--log-every", type=int, default=10, help="Progress logging interval by attempts.")
    parser.add_argument("--save-path", type=str, default="rm_dataset.pkl")
    parser.add_argument("--pref-jsonl", type=str, default="preference_pairs.jsonl")
    parser.add_argument("--mode", type=str, choices=["qwen", "rule"], default="rule")
    parser.add_argument(
        "--reward-margin-ratio",
        type=float,
        default=0.02,
        help="If |A-B total_reward| exceeds this ratio * max(|A|,|B|,1), pick higher-reward one directly.",
    )
    parser.add_argument(
        "--timeout-margin-pp",
        type=float,
        default=0.2,
        help="When rewards are close, if |A-B timeout_rate| exceeds this (percentage points), pick lower-timeout one.",
    )
    parser.add_argument(
        "--timeout-filter-pp",
        type=float,
        default=1.0,
        help="Hard pre-filter threshold on timeout_rate diff (percentage points) before LLM judging.",
    )
    parser.add_argument(
        "--gini-margin",
        type=float,
        default=0.02,
        help="If |A-B service_gini| exceeds this, pick lower-gini one.",
    )
    parser.add_argument(
        "--km-margin",
        type=float,
        default=0.20,
        help="If |A-B avg_rebalance_km_per_unit| exceeds this, pick lower one.",
    )
    parser.add_argument(
        "--util-margin",
        type=float,
        default=0.01,
        help="When reward and timeout are close, if |A-B demand_weighted_util| exceeds this, pick higher-util one.",
    )
    parser.add_argument(
        "--label-objective",
        type=str,
        choices=["residual", "reward_first"],
        default="residual",
        help="residual: RM focuses on soft preferences and avoids double-counting env reward.",
    )
    parser.add_argument("--qwen-model", type=str, default="qwen-plus")
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=20240908)
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--hard-neg-sampling-prob", type=float, default=0.35, help="Probability of adversarial/high-noise pair generation.")
    parser.add_argument("--hard-neg-noise", type=float, default=0.30, help="Exploration noise used in adversarial branch.")
    parser.add_argument("--hard-neg-timeout-delta", type=float, default=0.5, help="Hard-negative threshold on timeout_rate gap (percentage points).")
    parser.add_argument("--hard-neg-gini-delta", type=float, default=0.01, help="Hard-negative threshold on service_gini gap.")
    parser.add_argument("--hard-neg-whiplash-delta", type=float, default=1.0, help="Hard-negative threshold on whiplash gap.")
    parser.add_argument("--hard-neg-km-delta", type=float, default=0.20, help="Hard-negative threshold on avg_rebalance_km_per_unit gap.")
    parser.add_argument("--hard-neg-reward-margin-ratio", type=float, default=0.03, help="Minimum reward gap ratio for high-reward candidate in hard-negative detection.")
    parser.add_argument("--rebuild-hard-neg-target-ratio", type=float, default=0.40, help="Target hard-negative ratio after reconstruction.")
    parser.add_argument("--ethics-workload-threshold", type=float, default=1.2)
    parser.add_argument("--ethics-cs-share-threshold", type=float, default=0.65)
    parser.add_argument("--ethics-max-streak-steps", type=int, default=48, help="Consecutive stress steps threshold.")
    parser.add_argument("--ethics-hours-threshold", type=float, default=6.0, help="Total remote CS stress hours threshold.")
    parser.add_argument("--gray-reward-margin-ratio", type=float, default=0.03, help="LLM is used when reward gap is within this ratio.")
    parser.add_argument("--gray-timeout-margin-pp", type=float, default=0.5, help="LLM is used when timeout gap is within this margin.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EnvConfig(max_steps=args.max_steps, random_seed=args.seed)
    env = HybridFulfillmentEnv(args.dataset, cfg)
    policy = BCPolicy(args.bc_ckpt, device=device)

    collect_cfg = CollectConfig(
        pairs_to_collect=args.pairs,
        rollout_steps=args.rollout_steps,
        save_path=args.save_path,
        pref_jsonl_path=args.pref_jsonl,
        noise_a=args.noise_a,
        noise_b=args.noise_b,
        min_reward_diff_ratio=args.min_reward_diff_ratio,
        min_timeout_diff=args.min_timeout_diff,
        min_demand_util_diff=args.min_demand_util_diff,
        min_km_per_unit_diff=args.min_km_per_unit_diff,
        warmup_max_steps=args.warmup_max_steps,
        max_attempts=args.max_attempts,
        log_every=args.log_every,
        mode=args.mode,
        qwen_model=args.qwen_model,
        api_base=args.api_base,
        api_key=args.api_key,
        reward_margin_ratio=args.reward_margin_ratio,
        timeout_margin_pp=args.timeout_margin_pp,
        timeout_filter_pp=args.timeout_filter_pp,
        gini_margin=args.gini_margin,
        km_margin=args.km_margin,
        util_margin=args.util_margin,
        label_objective=args.label_objective,
        seed=args.seed,
        hard_neg_sampling_prob=args.hard_neg_sampling_prob,
        hard_neg_noise=args.hard_neg_noise,
        hard_neg_timeout_delta=args.hard_neg_timeout_delta,
        hard_neg_gini_delta=args.hard_neg_gini_delta,
        hard_neg_whiplash_delta=args.hard_neg_whiplash_delta,
        hard_neg_km_delta=args.hard_neg_km_delta,
        hard_neg_reward_margin_ratio=args.hard_neg_reward_margin_ratio,
        rebuild_hard_neg_target_ratio=args.rebuild_hard_neg_target_ratio,
        ethics_workload_threshold=args.ethics_workload_threshold,
        ethics_cs_share_threshold=args.ethics_cs_share_threshold,
        ethics_max_streak_steps=args.ethics_max_streak_steps,
        ethics_hours_threshold=args.ethics_hours_threshold,
        gray_reward_margin_ratio=args.gray_reward_margin_ratio,
        gray_timeout_margin_pp=args.gray_timeout_margin_pp,
    )
    collect_rm_dataset(env_base=env, policy=policy, cfg=collect_cfg)


if __name__ == "__main__":
    main()


