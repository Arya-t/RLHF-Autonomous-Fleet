import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, Dataset

from train_bc import BCActor


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 1e-5
    weight_decay: float = 1e-6
    batch_size: int = 16
    epochs: int = 10
    grad_clip_norm: float = 1.0
    rebalance_std: float = 1.0
    rebal_logprob_weight: float = 1.0
    length_normalize: bool = True
    val_ratio: float = 0.1
    seed: int = 20240908


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    return idx[n_val:], idx[:n_val]


class DPODataset(Dataset):
    """
    Build DPO pairs from collect_rm_data_llm.py output:
    item = {
      "traj_A": {"states","prices","rebalances",...},
      "traj_B": {...},
      "preference_label": 1 if A else 0
    }
    """

    def __init__(self, data_path: str, state_mean: np.ndarray, state_std: np.ndarray):
        with open(data_path, "rb") as f:
            raw = pickle.load(f)
        self.state_mean = state_mean.astype(np.float32).reshape(-1)
        self.state_std = state_std.astype(np.float32).reshape(-1)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)

        self.pairs: List[Dict] = []
        for it in raw:
            if "preference_label" not in it:
                continue
            label = int(it["preference_label"])
            traj_w = it["traj_A"] if label == 1 else it["traj_B"]
            traj_l = it["traj_B"] if label == 1 else it["traj_A"]

            sw = np.asarray(traj_w["states"], dtype=np.float32)
            sl = np.asarray(traj_l["states"], dtype=np.float32)
            pw = np.asarray(traj_w["prices"], dtype=np.int64)
            pl = np.asarray(traj_l["prices"], dtype=np.int64)
            rw = np.asarray(traj_w["rebalances"], dtype=np.float32)
            rl = np.asarray(traj_l["rebalances"], dtype=np.float32)
            mpw = np.asarray(traj_w.get("price_mask", []), dtype=np.float32).reshape(-1)
            mpl = np.asarray(traj_l.get("price_mask", []), dtype=np.float32).reshape(-1)
            mrw = np.asarray(traj_w.get("rebalance_mask", []), dtype=np.float32).reshape(-1)
            mrl = np.asarray(traj_l.get("rebalance_mask", []), dtype=np.float32).reshape(-1)

            if sw.ndim != 2 or sl.ndim != 2:
                continue
            if pw.ndim != 2 or pl.ndim != 2:
                continue
            if rw.ndim != 3 or rl.ndim != 3:
                continue
            if len(sw) == 0 or len(sl) == 0:
                continue

            # Normalize states using BC ckpt stats.
            sw = (sw - self.state_mean) / self.state_std
            sl = (sl - self.state_mean) / self.state_std

            # Backward compatibility: if masks absent, infer with old heuristics.
            if mpw.shape[0] != sw.shape[0]:
                mpw = (pw >= 0).all(axis=1).astype(np.float32)
            if mpl.shape[0] != sl.shape[0]:
                mpl = (pl >= 0).all(axis=1).astype(np.float32)
            if mrw.shape[0] != sw.shape[0]:
                mrw = (np.abs(rw).sum(axis=(1, 2)) > 1e-8).astype(np.float32)
            if mrl.shape[0] != sl.shape[0]:
                mrl = (np.abs(rl).sum(axis=(1, 2)) > 1e-8).astype(np.float32)

            self.pairs.append(
                {
                    "s_w": sw.astype(np.float32),
                    "p_w": pw.astype(np.int64),
                    "r_w": rw.astype(np.float32),
                    "mp_w": mpw.astype(np.float32),
                    "mr_w": mrw.astype(np.float32),
                    "s_l": sl.astype(np.float32),
                    "p_l": pl.astype(np.int64),
                    "r_l": rl.astype(np.float32),
                    "mp_l": mpl.astype(np.float32),
                    "mr_l": mrl.astype(np.float32),
                }
            )

        if len(self.pairs) == 0:
            raise ValueError("No valid DPO pairs found in dataset.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def dpo_collate(batch: List[Dict]):
    def pad_side(key_s, key_p, key_r, key_mp, key_mr):
        lengths = [x[key_s].shape[0] for x in batch]
        bsz = len(batch)
        tmax = max(lengths)
        sd = batch[0][key_s].shape[1]
        nr = batch[0][key_p].shape[1]
        s = np.zeros((bsz, tmax, sd), dtype=np.float32)
        p = np.zeros((bsz, tmax, nr), dtype=np.int64)
        r = np.zeros((bsz, tmax, nr, nr), dtype=np.float32)
        m_t = np.zeros((bsz, tmax), dtype=np.float32)
        m_p = np.zeros((bsz, tmax), dtype=np.float32)
        m_r = np.zeros((bsz, tmax), dtype=np.float32)

        for i, x in enumerate(batch):
            t = x[key_s].shape[0]
            s[i, :t] = x[key_s]
            p_i = x[key_p]
            # price action -1 means no price decision at this step in collected data
            p_mask = np.asarray(x[key_mp], dtype=np.float32).reshape(-1)
            if p_mask.shape[0] != t:
                p_mask = (p_i >= 0).all(axis=1).astype(np.float32)
            p_clamped = np.maximum(p_i, 0)
            p[i, :t] = p_clamped
            r_i = x[key_r]
            r[i, :t] = r_i
            m_t[i, :t] = 1.0
            m_p[i, :t] = p_mask
            r_mask = np.asarray(x[key_mr], dtype=np.float32).reshape(-1)
            if r_mask.shape[0] != t:
                r_mask = (np.abs(r_i).sum(axis=(1, 2)) > 1e-8).astype(np.float32)
            m_r[i, :t] = r_mask

        return {
            "states": torch.from_numpy(s),
            "price": torch.from_numpy(p),
            "reb": torch.from_numpy(r),
            "mask_t": torch.from_numpy(m_t),
            "mask_p": torch.from_numpy(m_p),
            "mask_r": torch.from_numpy(m_r),
        }

    win = pad_side("s_w", "p_w", "r_w", "mp_w", "mr_w")
    lose = pad_side("s_l", "p_l", "r_l", "mp_l", "mr_l")
    return {"win": win, "lose": lose}


def sequence_logprob(
    actor: BCActor,
    states: torch.Tensor,   # [B,T,D]
    price: torch.Tensor,    # [B,T,N]
    reb: torch.Tensor,      # [B,T,N,N]
    mask_t: torch.Tensor,   # [B,T]
    mask_p: torch.Tensor,   # [B,T]
    mask_r: torch.Tensor,   # [B,T]
    rebalance_std: float,
    rebal_logprob_weight: float,
    length_normalize: bool,
):
    bsz, tmax, d = states.shape
    n_region = price.shape[-1]

    flat_s = states.reshape(bsz * tmax, d)
    logits, reb_mean = actor(flat_s)  # [B*T,N,P], [B*T,N,N]
    p_classes = logits.shape[-1]

    flat_p = price.reshape(bsz * tmax, n_region).long()
    flat_r = reb.reshape(bsz * tmax, n_region, n_region)
    flat_mt = mask_t.reshape(bsz * tmax)
    flat_mp = mask_p.reshape(bsz * tmax)
    flat_mr = mask_r.reshape(bsz * tmax)

    # Discrete price log-prob, masked by price-decision steps.
    flat_p = torch.clamp(flat_p, 0, p_classes - 1)
    p_dist = Categorical(logits=logits)
    # Use mean over regions to keep price/rebalance scales comparable.
    p_lp = p_dist.log_prob(flat_p).mean(dim=-1) * flat_mp

    # Continuous rebalance log-prob, masked by rebalance-decision steps.
    std = torch.tensor(rebalance_std, device=states.device, dtype=states.dtype)
    r_dist = Normal(reb_mean, std)
    # Use mean over matrix dims to avoid continuous branch dominating logits scale.
    r_lp = r_dist.log_prob(flat_r).mean(dim=(-1, -2)) * flat_mr * rebal_logprob_weight

    step_lp = (p_lp + r_lp) * flat_mt
    traj_lp = step_lp.view(bsz, tmax).sum(dim=1)  # [B]
    if length_normalize:
        step_mask = ((mask_p + mask_r) > 0).float() * mask_t
        denom = step_mask.sum(dim=1).clamp(min=1.0)
        traj_lp = traj_lp / denom
    return traj_lp


def run_epoch(
    policy: BCActor,
    ref: BCActor,
    loader: DataLoader,
    optimizer,
    cfg: DPOConfig,
    device: torch.device,
    train: bool = True,
):
    if train:
        policy.train()
    else:
        policy.eval()
    ref.eval()

    total_loss = 0.0
    total_samples = 0
    total_pref_acc = 0.0
    total_margin = 0.0

    for batch in loader:
        win = {k: v.to(device) for k, v in batch["win"].items()}
        lose = {k: v.to(device) for k, v in batch["lose"].items()}
        bsz = win["states"].shape[0]

        with torch.set_grad_enabled(train):
            pol_w = sequence_logprob(
                policy,
                win["states"],
                win["price"],
                win["reb"],
                win["mask_t"],
                win["mask_p"],
                win["mask_r"],
                cfg.rebalance_std,
                cfg.rebal_logprob_weight,
                cfg.length_normalize,
            )
            pol_l = sequence_logprob(
                policy,
                lose["states"],
                lose["price"],
                lose["reb"],
                lose["mask_t"],
                lose["mask_p"],
                lose["mask_r"],
                cfg.rebalance_std,
                cfg.rebal_logprob_weight,
                cfg.length_normalize,
            )
            with torch.no_grad():
                ref_w = sequence_logprob(
                    ref,
                    win["states"],
                    win["price"],
                    win["reb"],
                    win["mask_t"],
                    win["mask_p"],
                    win["mask_r"],
                    cfg.rebalance_std,
                    cfg.rebal_logprob_weight,
                    cfg.length_normalize,
                )
                ref_l = sequence_logprob(
                    ref,
                    lose["states"],
                    lose["price"],
                    lose["reb"],
                    lose["mask_t"],
                    lose["mask_p"],
                    lose["mask_r"],
                    cfg.rebalance_std,
                    cfg.rebal_logprob_weight,
                    cfg.length_normalize,
                )

            logits = (pol_w - ref_w) - (pol_l - ref_l)
            loss = -F.logsigmoid(cfg.beta * logits).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        with torch.no_grad():
            pref_acc = (logits > 0).float().mean().item()
            margin = logits.mean().item()

        total_loss += float(loss.item()) * bsz
        total_pref_acc += float(pref_acc) * bsz
        total_margin += float(margin) * bsz
        total_samples += bsz

    if total_samples == 0:
        return {"loss": 0.0, "pref_acc": 0.0, "margin": 0.0}
    return {
        "loss": total_loss / total_samples,
        "pref_acc": total_pref_acc / total_samples,
        "margin": total_margin / total_samples,
    }


def build_actor_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        state_dict = ckpt.get("actor_state_dict", None)
    if state_dict is None:
        raise KeyError(
            f"{ckpt_path} does not contain 'model_state_dict' or 'actor_state_dict'."
        )
    # PPO actor checkpoints use base_actor.* keys and include log_std.
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
            "Please provide BC/PPO actor checkpoint with metadata."
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
    # Backfill inferred metadata so downstream save logic works for PPO init ckpts.
    ckpt["state_dim"] = int(state_dim)
    ckpt["n_region"] = int(n_region)
    ckpt["n_price_levels"] = int(n_price_levels)
    if not isinstance(ckpt.get("config", None), dict):
        ckpt["config"] = {}
    ckpt["config"]["hidden_dim"] = int(hidden_dim)
    ckpt["config"]["use_region_attn"] = bool(use_region_attn)
    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32).reshape(-1)
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32).reshape(-1)
    state_std = np.where(state_std < 1e-6, 1.0, state_std)
    return actor, ckpt, state_mean, state_std


def main():
    parser = argparse.ArgumentParser(description="Train DPO policy on trajectory preference pairs.")
    parser.add_argument("--data-path", type=str, default="rm_dataset.pkl")
    parser.add_argument("--pi-init", type=str, default="checkpoints_rlhf/pi_init.pt")
    parser.add_argument("--pi-ref", type=str, default="", help="default=pi-init")
    parser.add_argument("--save-dir", type=str, default="checkpoints_dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebalance-std", type=float, default=1.0)
    parser.add_argument("--rebal-logprob-weight", type=float, default=1.0)
    parser.add_argument("--length-normalize", type=int, default=1, help="1: normalize trajectory logprob by valid decision steps.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20240908)
    args = parser.parse_args()

    cfg = DPOConfig(
        beta=args.beta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        rebalance_std=args.rebalance_std,
        rebal_logprob_weight=args.rebal_logprob_weight,
        length_normalize=bool(args.length_normalize),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pi_ref_path = args.pi_ref if args.pi_ref else args.pi_init
    policy, init_ckpt, state_mean, state_std = build_actor_from_ckpt(args.pi_init, device)
    ref, _, _, _ = build_actor_from_ckpt(pi_ref_path, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    dataset = DPODataset(args.data_path, state_mean=state_mean, state_std=state_std)
    n_total = len(dataset)
    tr_idx, va_idx = split_indices(n_total, cfg.val_ratio, cfg.seed)
    tr_ds = torch.utils.data.Subset(dataset, tr_idx.tolist())
    va_ds = torch.utils.data.Subset(dataset, va_idx.tolist())
    if len(va_ds) == 0:
        raise ValueError("Validation split is empty. Increase data size or val-ratio.")

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=dpo_collate)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=dpo_collate)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "dpo_actor_best.pt")
    last_path = os.path.join(args.save_dir, "dpo_actor_last.pt")

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(policy, ref, tr_loader, optimizer, cfg, device, train=True)
        va = run_epoch(policy, ref, va_loader, optimizer, cfg, device, train=False)
        print(
            f"[Epoch {ep:03d}] train_loss={tr['loss']:.6f} train_pref_acc={tr['pref_acc']:.4f} train_margin={tr['margin']:.4f} "
            f"val_loss={va['loss']:.6f} val_pref_acc={va['pref_acc']:.4f} val_margin={va['margin']:.4f}"
        )

        ckpt = {
            "model_state_dict": policy.state_dict(),
            "state_dim": int(init_ckpt["state_dim"]),
            "n_region": int(init_ckpt["n_region"]),
            "n_price_levels": int(init_ckpt["n_price_levels"]),
            "state_mean": state_mean,
            "state_std": state_std,
            "config": {
                "hidden_dim": int(init_ckpt["config"]["hidden_dim"]),
                "use_region_attn": bool(init_ckpt.get("config", {}).get("use_region_attn", True)),
                "dpo_beta": cfg.beta,
                "dpo_rebalance_std": cfg.rebalance_std,
                "dpo_rebal_logprob_weight": cfg.rebal_logprob_weight,
                "dpo_lr": cfg.lr,
                "dpo_weight_decay": cfg.weight_decay,
                "dpo_seed": cfg.seed,
            },
            "epoch": ep,
            "val_loss": va["loss"],
            "train_pref_acc": tr["pref_acc"],
            "val_pref_acc": va["pref_acc"],
            "ref_path": pi_ref_path,
            "init_path": args.pi_init,
        }
        torch.save(ckpt, last_path)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, best_path)
            print(f"  -> saved best checkpoint: {best_path}")

    print("DPO training finished.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
