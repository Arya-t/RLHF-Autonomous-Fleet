import argparse
import pickle

import numpy as np

from env_hybrid import EnvConfig, HybridFulfillmentEnv


def expert_action(env: HybridFulfillmentEnv, obs: dict):
    """
    Heuristic expert used for BC/SFT data collection.
    price action semantics: surge price-multiplier level index.
    Higher level => stronger demand suppression and stronger CS supply response.
    """
    n = env.n
    action = {}

    workload = np.asarray(obs["workload"], dtype=float)
    av_units = np.asarray(obs["av_units"], dtype=float)
    cs_units = np.asarray(obs["cs_units"], dtype=float)

    # 1) Surge multiplier (slow clock, action key kept as "price" for compatibility)
    if env.t % env.config.pricing_interval_steps == 0:
        surge_action = np.zeros(n, dtype=int)
        max_idx = len(env.config.cs_share_levels) - 1
        base_idx = env.config.cs_share_levels.index(1.00)
        for i in range(n):
            wi = float(workload[i])
            av_tight = float(av_units[i]) < (0.75 * float(np.mean(av_units)) if np.mean(av_units) > 1e-6 else 0.0)
            cs_weak = float(cs_units[i]) < (0.75 * float(np.mean(cs_units)) if np.mean(cs_units) > 1e-6 else 0.0)

            # Smooth mapping from workload to surge level.
            if wi <= 0.2:
                idx = base_idx - 1
            elif wi <= 0.5:
                idx = base_idx
            elif wi <= 0.8:
                idx = base_idx + 1
            elif wi <= 1.2:
                idx = base_idx + 2
            else:
                idx = max_idx

            # If both AV and CS are weak locally, force one extra level up.
            if av_tight and cs_weak:
                idx = min(max_idx, idx + 1)

            surge_action[i] = max(0, min(max_idx, idx))

        action["price"] = surge_action

    # 2) AV rebalance (medium clock): move AV from low-pressure to high-pressure regions.
    if env.t % env.config.rebalance_interval_steps == 0:
        mat = np.zeros((n, n), dtype=float)

        # Supply pressure proxy: high workload + low AV stock => higher deficit.
        av_mean = float(np.mean(av_units)) if np.mean(av_units) > 1e-6 else 1.0
        deficit = workload + np.maximum(0.0, (av_mean - av_units) / max(av_mean, 1e-6))
        # Distance-normalized score to avoid over-preferring far destinations with marginal deficit gain.
        km = np.asarray(getattr(env, "travel_km", np.zeros((n, n))), dtype=float)
        nz = km[km > 0.0]
        km_scale = float(np.median(nz)) if nz.size > 0 else 1.0
        dist_penalty = 0.12

        for i in np.argsort(-av_units):
            avail = float(av_units[i])
            if avail <= 2.0:
                continue

            all_dst = [j for j in range(n) if getattr(env, "rebalance_mask", env.adj_mask)[i, j]]
            if not all_dst:
                continue

            # Source should have meaningful slack; allow earlier intervention at mid pressure.
            if deficit[i] > 1.15:
                continue

            # Prefer adjacent destinations; allow non-adjacent only under larger pressure gap.
            near_dst = [j for j in all_dst if bool(env.adj_mask[i, j])]
            cand_dst = near_dst if len(near_dst) > 0 else all_dst

            # Push to destination with highest (deficit - distance penalty).
            def _score(j):
                dnorm = float(km[i, j] / max(km_scale, 1e-6))
                return float(deficit[j]) - dist_penalty * dnorm

            best_dst = max(cand_dst, key=_score)
            gap = float(deficit[best_dst] - deficit[i])
            score_gap = float(_score(best_dst) - float(deficit[i]))
            if gap <= 0.08 or score_gap <= 0.04:
                continue

            # If picking a non-adjacent destination, require clearly larger pressure gap.
            if (not bool(env.adj_mask[i, best_dst])) and gap <= 0.20:
                continue

            # Keep base stock locally; scale dispatch aggressiveness by pressure gap.
            min_keep = max(3.0, 0.40 * avail)
            if gap < 0.20:
                send_ratio = 0.12
            elif gap < 0.40:
                send_ratio = 0.18
            else:
                send_ratio = 0.22
            send = min(send_ratio * avail, max(0.0, avail - min_keep))
            if send <= 0:
                continue
            mat[i, best_dst] += send

        action["rebalance"] = mat
    return action


def flatten_obs(obs: dict):
    """Flatten observation into a fixed-size float32 vector."""
    state = np.concatenate(
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
    )
    return state


def maybe_warm_start(env: HybridFulfillmentEnv, obs: dict, rng: np.random.Generator, start_min: int, start_max: int):
    done = False
    start_t = int(rng.integers(start_min, start_max + 1))
    start_t = max(0, min(start_t, env.config.max_steps - 1))
    for _ in range(start_t):
        obs, _, done, _ = env.step({})
        if done:
            break
    return obs, done, start_t


def collect_sft_data(
    env: HybridFulfillmentEnv,
    episodes=50,
    save_path="sft_dataset.pkl",
    hard_start_prob: float = 0.5,
    hard_start_min: int = 96,
    hard_start_max: int = 216,
):
    dataset = []
    rng = np.random.default_rng(env.config.random_seed)

    for ep in range(episodes):
        obs = env.reset(seed=env.config.random_seed + ep)
        done = False
        ep_steps = 0
        warm_start_t = -1
        # Hard-state oversampling: probabilistically warm-start from mid/high-load periods.
        if hard_start_prob > 0 and hard_start_min >= 0 and hard_start_max >= hard_start_min:
            if bool(rng.random() < hard_start_prob):
                obs, done, warm_start_t = maybe_warm_start(env, obs, rng, hard_start_min, hard_start_max)

        while not done:
            action = expert_action(env, obs)
            state_vector = flatten_obs(obs)

            # Decision masks: useful for multi-timescale policy training.
            price_mask = int(env.t % env.config.pricing_interval_steps == 0)
            rebalance_mask = int(env.t % env.config.rebalance_interval_steps == 0)

            transition = {
                "state": state_vector,
                "price_action": action.get("price", np.zeros(env.n, dtype=int)),
                "rebalance_action": action.get("rebalance", np.zeros((env.n, env.n), dtype=np.float32)),
                "price_mask": price_mask,
                "rebalance_mask": rebalance_mask,
                "t": int(env.t),
            }
            dataset.append(transition)

            obs, _, done, _ = env.step(action)
            ep_steps += 1

        print(f"Collected episode {ep + 1}/{episodes}, warm_start_t={warm_start_t}, steps={ep_steps}")

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} transitions to {save_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Collect BC/SFT dataset using expert heuristic.")
    parser.add_argument("--dataset", type=str, default="allarea_set_hybrid7_p12.pkl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=20240908)
    parser.add_argument("--save-path", type=str, default="sft_dataset.pkl")
    parser.add_argument(
        "--peak-start-min",
        type=int,
        default=-1,
        help="If >=0, warmup start time lower bound (inclusive) before collecting each episode.",
    )
    parser.add_argument(
        "--peak-start-max",
        type=int,
        default=-1,
        help="If >=0, warmup start time upper bound (inclusive) before collecting each episode.",
    )
    parser.add_argument(
        "--hard-start-prob",
        type=float,
        default=0.5,
        help="Probability of warm-starting an episode from mid/high-load periods.",
    )
    parser.add_argument(
        "--hard-start-min",
        type=int,
        default=96,
        help="Warm-start lower bound (inclusive) when hard-start triggers.",
    )
    parser.add_argument(
        "--hard-start-max",
        type=int,
        default=216,
        help="Warm-start upper bound (inclusive) when hard-start triggers.",
    )
    args = parser.parse_args()

    cfg = EnvConfig(max_steps=args.max_steps, random_seed=args.seed)
    env = HybridFulfillmentEnv(args.dataset, cfg)

    # Optional peak-start warmup: bias data toward high-load periods.
    if args.peak_start_min >= 0 and args.peak_start_max >= 0:
        if args.peak_start_max < args.peak_start_min:
            raise ValueError("--peak-start-max must be >= --peak-start-min")
        rng = np.random.default_rng(args.seed)
        dataset = []
        for ep in range(args.episodes):
            obs = env.reset(seed=env.config.random_seed + ep)
            done = False
            ep_steps = 0

            start_t = int(rng.integers(args.peak_start_min, args.peak_start_max + 1))
            start_t = max(0, min(start_t, env.config.max_steps - 1))
            for _ in range(start_t):
                obs, _, done, _ = env.step({})
                if done:
                    break

            while not done:
                action = expert_action(env, obs)
                state_vector = flatten_obs(obs)
                price_mask = int(env.t % env.config.pricing_interval_steps == 0)
                rebalance_mask = int(env.t % env.config.rebalance_interval_steps == 0)

                transition = {
                    "state": state_vector,
                    "price_action": action.get("price", np.zeros(env.n, dtype=int)),
                    "rebalance_action": action.get("rebalance", np.zeros((env.n, env.n), dtype=np.float32)),
                    "price_mask": price_mask,
                    "rebalance_mask": rebalance_mask,
                    "t": int(env.t),
                }
                dataset.append(transition)
                obs, _, done, _ = env.step(action)
                ep_steps += 1

            print(
                f"Collected episode {ep + 1}/{args.episodes}, warm_start_t={start_t}, steps={ep_steps}"
            )

        with open(args.save_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved {len(dataset)} transitions to {args.save_path}")
    else:
        collect_sft_data(
            env,
            episodes=args.episodes,
            save_path=args.save_path,
            hard_start_prob=args.hard_start_prob,
            hard_start_min=args.hard_start_min,
            hard_start_max=args.hard_start_max,
        )


if __name__ == "__main__":
    main()
