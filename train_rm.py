import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class RMConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    hidden_dim: int = 256
    val_ratio: float = 0.1
    seed: int = 20240908
    grad_clip_norm: float = 1.0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def infer_dims(sample_item):
    # state: [T, D], rebalance: [T, N, N]
    s = np.asarray(sample_item["traj_A"]["states"], dtype=np.float32)
    r = np.asarray(sample_item["traj_A"]["rebalances"], dtype=np.float32)
    if s.ndim != 2 or r.ndim != 3:
        raise ValueError("Unexpected trajectory shapes in rm_dataset.pkl")
    state_dim = s.shape[1]
    n_region = r.shape[1]
    return state_dim, n_region


def extract_price_from_state(states: np.ndarray, n_region: int):
    # flatten_obs layout: [slot, price_idx(n_region), av(n_region), cs(n_region), outstanding(n_region), capacity(n_region), workload(n_region)]
    price = states[:, 1 : 1 + n_region]
    price = np.rint(price).astype(np.int64)
    return price


class PairRMDataset(Dataset):
    def __init__(self, items: List[dict], n_region: int, n_price_levels: int, state_mean=None, state_std=None):
        self.items = items
        self.n_region = n_region
        self.n_price_levels = n_price_levels

        # Build normalization stats from provided split only.
        all_states = []
        for it in items:
            all_states.append(np.asarray(it["traj_A"]["states"], dtype=np.float32))
            all_states.append(np.asarray(it["traj_B"]["states"], dtype=np.float32))
        cat = np.concatenate(all_states, axis=0)
        if state_mean is None or state_std is None:
            state_mean = cat.mean(axis=0, keepdims=True)
            state_std = cat.std(axis=0, keepdims=True)
        state_std = np.where(state_std < 1e-6, 1.0, state_std)
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        sA = np.asarray(it["traj_A"]["states"], dtype=np.float32)
        sB = np.asarray(it["traj_B"]["states"], dtype=np.float32)
        rA = np.asarray(it["traj_A"]["rebalances"], dtype=np.float32)
        rB = np.asarray(it["traj_B"]["rebalances"], dtype=np.float32)
        y = float(it["preference_label"])  # 1 if A preferred, else 0

        # Normalize states
        sA = (sA - self.state_mean) / self.state_std
        sB = (sB - self.state_mean) / self.state_std

        # Price actions reconstructed from state to avoid invalid -1 placeholders.
        pA = extract_price_from_state(np.asarray(it["traj_A"]["states"], dtype=np.float32), self.n_region)
        pB = extract_price_from_state(np.asarray(it["traj_B"]["states"], dtype=np.float32), self.n_region)
        pA = np.clip(pA, 0, self.n_price_levels - 1)
        pB = np.clip(pB, 0, self.n_price_levels - 1)

        return {
            "state_A": torch.from_numpy(sA),
            "price_A": torch.from_numpy(pA),
            "rebal_A": torch.from_numpy(rA),
            "state_B": torch.from_numpy(sB),
            "price_B": torch.from_numpy(pB),
            "rebal_B": torch.from_numpy(rB),
            "label": torch.tensor(y, dtype=torch.float32),
        }


def pad_sequence_3d(tensors, pad_value=0.0):
    # list of [T, D] -> [B, Tmax, D], mask [B, Tmax]
    b = len(tensors)
    tmax = max(t.shape[0] for t in tensors)
    d = tensors[0].shape[1]
    out = torch.full((b, tmax, d), pad_value, dtype=tensors[0].dtype)
    mask = torch.zeros((b, tmax), dtype=torch.float32)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L] = t
        mask[i, :L] = 1.0
    return out, mask


def pad_sequence_4d(tensors, pad_value=0.0):
    # list of [T, N, N] -> [B, Tmax, N, N]
    b = len(tensors)
    tmax = max(t.shape[0] for t in tensors)
    n1, n2 = tensors[0].shape[1], tensors[0].shape[2]
    out = torch.full((b, tmax, n1, n2), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L] = t
    return out


def pad_sequence_price(tensors, pad_value=0):
    # list of [T, N] -> [B, Tmax, N]
    b = len(tensors)
    tmax = max(t.shape[0] for t in tensors)
    n = tensors[0].shape[1]
    out = torch.full((b, tmax, n), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L] = t
    return out


def collate_fn(batch):
    sA, pA, rA = [x["state_A"] for x in batch], [x["price_A"] for x in batch], [x["rebal_A"] for x in batch]
    sB, pB, rB = [x["state_B"] for x in batch], [x["price_B"] for x in batch], [x["rebal_B"] for x in batch]
    y = torch.stack([x["label"] for x in batch], dim=0)

    sA_pad, mA = pad_sequence_3d(sA)
    sB_pad, mB = pad_sequence_3d(sB)
    pA_pad = pad_sequence_price(pA, pad_value=0)
    pB_pad = pad_sequence_price(pB, pad_value=0)
    rA_pad = pad_sequence_4d(rA)
    rB_pad = pad_sequence_4d(rB)

    return {
        "state_A": sA_pad,
        "price_A": pA_pad,
        "rebal_A": rA_pad,
        "mask_A": mA,
        "state_B": sB_pad,
        "price_B": pB_pad,
        "rebal_B": rB_pad,
        "mask_B": mB,
        "label": y,
    }


class TrajectoryRewardModel(nn.Module):
    def __init__(self, state_dim=43, n_region=7, n_price_levels=4, hidden_dim=256):
        super().__init__()
        self.n_region = n_region
        self.n_price_levels = n_price_levels
        action_dim = (n_region * n_price_levels) + (n_region * n_region)
        input_dim = state_dim + action_dim

        self.step_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        self.reward_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, states, price_actions, rebal_actions, step_mask):
        # states: [B,T,D], price_actions: [B,T,N], rebal_actions:[B,T,N,N], step_mask:[B,T]
        B, T, _ = states.shape
        p_oh = F.one_hot(price_actions.long(), num_classes=self.n_price_levels).float()  # [B,T,N,P]
        p_flat = p_oh.view(B, T, -1)
        r_flat = rebal_actions.view(B, T, -1)
        x = torch.cat([states, p_flat, r_flat], dim=-1)
        h = self.step_net(x)  # [B,T,H]

        m = step_mask.unsqueeze(-1)  # [B,T,1]
        h_sum = (h * m).sum(dim=1)
        h_cnt = m.sum(dim=1).clamp_min(1e-6)
        traj_h = h_sum / h_cnt
        reward = self.reward_head(traj_h).squeeze(-1)
        return reward


def run_epoch(model, loader, optimizer, device, cfg: RMConfig, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        sA = batch["state_A"].to(device)
        pA = batch["price_A"].to(device)
        rA = batch["rebal_A"].to(device)
        mA = batch["mask_A"].to(device)
        sB = batch["state_B"].to(device)
        pB = batch["price_B"].to(device)
        rB = batch["rebal_B"].to(device)
        mB = batch["mask_B"].to(device)
        y = batch["label"].to(device)  # 1 if A preferred

        with torch.set_grad_enabled(train):
            RA = model(sA, pA, rA, mA)
            RB = model(sB, pB, rB, mB)
            diff = torch.where(y > 0.5, RA - RB, RB - RA)
            loss = -F.logsigmoid(diff).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        pred = (RA > RB).float()
        correct = (pred == y).sum().item()
        bs = y.shape[0]
        total += bs
        total_correct += correct
        total_loss += loss.item() * bs

    if total == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / total, "acc": total_correct / total}


def main():
    parser = argparse.ArgumentParser(description="Train trajectory-level reward model from rm_dataset.pkl")
    parser.add_argument("--data", type=str, default="rm_dataset.pkl")
    parser.add_argument("--save-dir", type=str, default="checkpoints_rm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20240908)
    args = parser.parse_args()

    cfg = RMConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    with open(args.data, "rb") as f:
        items = pickle.load(f)
    if len(items) < 20:
        raise ValueError(f"Too few pairs ({len(items)}). Recommend >= 50.")

    state_dim, n_region = infer_dims(items[0])
    # infer max price index from states' price segment
    max_price = 0
    for it in items:
        sA = np.asarray(it["traj_A"]["states"], dtype=np.float32)
        sB = np.asarray(it["traj_B"]["states"], dtype=np.float32)
        pA = extract_price_from_state(sA, n_region)
        pB = extract_price_from_state(sB, n_region)
        max_price = max(max_price, int(np.max(pA)), int(np.max(pB)))
    n_price_levels = max(2, max_price + 1)

    train_idx, val_idx = split_indices(len(items), cfg.val_ratio, cfg.seed)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    if len(val_items) == 0:
        raise ValueError("Validation split empty.")

    train_ds = PairRMDataset(train_items, n_region=n_region, n_price_levels=n_price_levels)
    val_ds = PairRMDataset(
        val_items,
        n_region=n_region,
        n_price_levels=n_price_levels,
        state_mean=train_ds.state_mean,
        state_std=train_ds.state_std,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryRewardModel(
        state_dim=state_dim, n_region=n_region, n_price_levels=n_price_levels, hidden_dim=cfg.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = -1.0
    best_path = os.path.join(args.save_dir, "rm_best.pt")
    last_path = os.path.join(args.save_dir, "rm_last.pt")

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        va = run_epoch(model, val_loader, optimizer, device, cfg, train=False)
        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
            f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "n_region": n_region,
            "n_price_levels": n_price_levels,
            "state_mean": train_ds.state_mean,
            "state_std": train_ds.state_std,
            "config": vars(cfg),
            "epoch": ep,
            "val_acc": va["acc"],
            "val_loss": va["loss"],
        }
        torch.save(ckpt, last_path)
        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            torch.save(ckpt, best_path)
            print(f"  -> saved best checkpoint: {best_path}")

    print("Done.")
    print(f"Best val_acc={best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
