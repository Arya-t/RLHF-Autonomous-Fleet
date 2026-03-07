import argparse
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class BCConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 30
    hidden_dim: int = 256
    val_ratio: float = 0.1
    seed: int = 20240908
    rebalance_loss_weight: float = 1.0
    price_loss_weight: float = 1.0
    grad_clip_norm: float = 1.0
    use_region_attn: bool = True


class BCActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_region: int,
        n_price_levels: int,
        hidden_dim: int = 256,
        use_region_attn: bool = True,
    ):
        super().__init__()
        self.n_region = n_region
        self.n_price_levels = n_price_levels
        self.use_region_attn = bool(use_region_attn)
        self.state_dim = int(state_dim)

        mid_dim = max(128, hidden_dim // 2)
        self.region_feat_dim = None
        self.attn_dim = max(32, min(hidden_dim, 128))
        if (self.state_dim - 1) > 0 and (self.state_dim - 1) % max(1, self.n_region) == 0:
            self.region_feat_dim = (self.state_dim - 1) // max(1, self.n_region)

        self.region_proj = None
        self.region_attn = None
        self.region_norm = None
        shared_in_dim = self.state_dim
        if self.use_region_attn and self.region_feat_dim is not None:
            # Token = [slot, region local features], one token per region.
            token_dim = 1 + self.region_feat_dim
            self.region_proj = nn.Linear(token_dim, self.attn_dim)
            self.region_attn = nn.MultiheadAttention(
                embed_dim=self.attn_dim,
                num_heads=1,
                batch_first=True,
            )
            self.region_norm = nn.LayerNorm(self.attn_dim)
            # Concatenate attended region context with original flattened state.
            shared_in_dim = self.state_dim + self.n_region * self.attn_dim

        self.shared_net = nn.Sequential(
            nn.Linear(shared_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
        )
        self.price_head = nn.Sequential(
            nn.Linear(mid_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_region * n_price_levels),
        )
        self.rebalance_head = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_region * n_region),
        )

    def forward(self, x):
        if self.region_proj is not None and self.region_attn is not None and self.region_norm is not None:
            slot = x[:, :1]  # [B,1]
            region_raw = x[:, 1:]  # [B, N*F]
            region_tok = region_raw.view(-1, self.n_region, self.region_feat_dim)  # [B,N,F]
            slot_tok = slot.unsqueeze(1).expand(-1, self.n_region, -1)  # [B,N,1]
            tok = torch.cat([slot_tok, region_tok], dim=-1)  # [B,N,1+F]
            tok = self.region_proj(tok)
            attn_out, _ = self.region_attn(tok, tok, tok, need_weights=False)
            attn_out = self.region_norm(attn_out + tok)
            x = torch.cat([x, attn_out.reshape(x.shape[0], -1)], dim=-1)

        h = self.shared_net(x)
        price_logits = self.price_head(h).view(-1, self.n_region, self.n_price_levels)
        rebalance = self.rebalance_head(h).view(-1, self.n_region, self.n_region)
        rebalance = torch.relu(rebalance)
        return price_logits, rebalance


class SFTDataset(Dataset):
    def __init__(self, items, state_mean=None, state_std=None):
        self.items = items
        states = np.stack([it["state"] for it in items], axis=0).astype(np.float32)

        if state_mean is None or state_std is None:
            state_mean = states.mean(axis=0, keepdims=True)
            state_std = states.std(axis=0, keepdims=True)

        state_std = np.where(state_std < 1e-6, 1.0, state_std)
        states = (states - state_mean) / state_std

        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)

        self.states = states.astype(np.float32)
        self.price = np.stack([it["price_action"] for it in items], axis=0).astype(np.int64)
        self.rebalance = np.stack([it["rebalance_action"] for it in items], axis=0).astype(np.float32)
        self.price_mask = np.array([it.get("price_mask", 1) for it in items], dtype=np.float32)
        self.rebalance_mask = np.array([it.get("rebalance_mask", 1) for it in items], dtype=np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.price[idx]),
            torch.from_numpy(self.rebalance[idx]),
            torch.tensor(self.price_mask[idx], dtype=torch.float32),
            torch.tensor(self.rebalance_mask[idx], dtype=torch.float32),
        )


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(items, val_ratio=0.1, seed=20240908):
    idx = np.arange(len(items))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(len(items) * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items


def run_epoch(model, loader, optimizer, device, cfg: BCConfig, train=True):
    if train:
        model.train()
    else:
        model.eval()

    ce = nn.CrossEntropyLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    total_loss = 0.0
    total_price_loss = 0.0
    total_reb_loss = 0.0
    total_price_acc = 0.0
    total_count = 0

    for states, price_tgt, reb_tgt, price_mask, reb_mask in loader:
        states = states.to(device)
        price_tgt = price_tgt.to(device)
        reb_tgt = reb_tgt.to(device)
        price_mask = price_mask.to(device)
        reb_mask = reb_mask.to(device)

        with torch.set_grad_enabled(train):
            price_logits, reb_pred = model(states)

            # price CE over all regions, masked by price decision steps
            bsz, n_region, n_levels = price_logits.shape
            price_loss_all = ce(
                price_logits.reshape(bsz * n_region, n_levels),
                price_tgt.reshape(bsz * n_region),
            ).reshape(bsz, n_region)
            price_loss = (price_loss_all.mean(dim=1) * price_mask).sum() / (price_mask.sum() + 1e-6)

            # rebalance regression loss, masked by rebalance decision steps
            reb_loss_all = mse(reb_pred, reb_tgt).mean(dim=(1, 2))
            reb_loss = (reb_loss_all * reb_mask).sum() / (reb_mask.sum() + 1e-6)

            loss = cfg.price_loss_weight * price_loss + cfg.rebalance_loss_weight * reb_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        with torch.no_grad():
            pred = price_logits.argmax(dim=-1)
            price_acc = (pred == price_tgt).float().mean(dim=1)
            price_acc = (price_acc * price_mask).sum() / (price_mask.sum() + 1e-6)

        cnt = states.shape[0]
        total_count += cnt
        total_loss += float(loss.item()) * cnt
        total_price_loss += float(price_loss.item()) * cnt
        total_reb_loss += float(reb_loss.item()) * cnt
        total_price_acc += float(price_acc.item()) * cnt

    if total_count == 0:
        return {"loss": 0.0, "price_loss": 0.0, "reb_loss": 0.0, "price_acc": 0.0}

    return {
        "loss": total_loss / total_count,
        "price_loss": total_price_loss / total_count,
        "reb_loss": total_reb_loss / total_count,
        "price_acc": total_price_acc / total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Train BC actor on sft_dataset.pkl.")
    parser.add_argument("--data", type=str, default="sft_dataset.pkl")
    parser.add_argument("--save-dir", type=str, default="checkpoints_bc")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20240908)
    parser.add_argument("--price-loss-weight", type=float, default=1.0)
    parser.add_argument("--rebalance-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-region-attn", type=int, default=1, help="1 enables lightweight single-head region attention.")
    args = parser.parse_args()

    cfg = BCConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        val_ratio=args.val_ratio,
        seed=args.seed,
        price_loss_weight=args.price_loss_weight,
        rebalance_loss_weight=args.rebalance_loss_weight,
        use_region_attn=bool(args.use_region_attn),
    )
    set_seed(cfg.seed)

    with open(args.data, "rb") as f:
        items = pickle.load(f)
    if len(items) == 0:
        raise ValueError("Empty dataset.")

    train_items, val_items = split_dataset(items, val_ratio=cfg.val_ratio, seed=cfg.seed)
    if len(val_items) == 0:
        raise ValueError("Validation split is empty. Increase dataset size or val_ratio.")

    sample = items[0]
    state_dim = int(np.asarray(sample["state"]).shape[0])
    n_region = int(np.asarray(sample["price_action"]).shape[0])
    n_price_levels = int(np.max(np.stack([it["price_action"] for it in items], axis=0))) + 1

    train_ds = SFTDataset(train_items)
    val_ds = SFTDataset(val_items, state_mean=train_ds.state_mean, state_std=train_ds.state_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCActor(
        state_dim,
        n_region,
        n_price_levels,
        hidden_dim=cfg.hidden_dim,
        use_region_attn=cfg.use_region_attn,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "bc_actor_best.pt")
    last_path = os.path.join(args.save_dir, "bc_actor_last.pt")

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        va = run_epoch(model, val_loader, optimizer, device, cfg, train=False)

        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={tr['loss']:.6f} train_price_acc={tr['price_acc']:.4f} "
            f"val_loss={va['loss']:.6f} val_price_acc={va['price_acc']:.4f}"
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
            "val_loss": va["loss"],
        }
        torch.save(ckpt, last_path)

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, best_path)
            print(f"  -> saved best checkpoint: {best_path}")

    print("Done.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
