#!/usr/bin/env bash
set -euo pipefail

DATASET="allarea_set_hybrid7_p12.pkl"
SEED="20240908"
API_KEY="${DASHSCOPE_API_KEY:-}"

if [ -z "$API_KEY" ]; then
  echo "Please set DASHSCOPE_API_KEY before running."
  exit 1
fi

(
  echo "=== [Branch A] Build dummy RM ==="
  python -c '
import torch
from train_rm import TrajectoryRewardModel
bc_ckpt = torch.load("ckpt_bc_full/bc_actor_best.pt", map_location="cpu")
rm = TrajectoryRewardModel(
    state_dim=bc_ckpt["state_dim"],
    n_region=bc_ckpt["n_region"],
    n_price_levels=bc_ckpt["n_price_levels"],
    hidden_dim=bc_ckpt["config"]["hidden_dim"],
)
torch.save(
    {
        "model_state_dict": rm.state_dict(),
        "state_dim": bc_ckpt["state_dim"],
        "n_region": bc_ckpt["n_region"],
        "n_price_levels": bc_ckpt["n_price_levels"],
        "config": {"hidden_dim": bc_ckpt["config"]["hidden_dim"]},
        "state_mean": bc_ckpt["state_mean"],
        "state_std": bc_ckpt["state_std"],
    },
    "dummy_rm_full.pt",
)
'

  echo "=== [Branch A] PPO warm start without RM ==="
  python train_ppov2_rlhf.py \
    --dataset "$DATASET" \
    --pi-init ckpt_bc_full/bc_actor_best.pt \
    --pi-ref ckpt_bc_full/bc_actor_best.pt \
    --rm-ckpt dummy_rm_full.pt \
    --save-dir ckpt_ppo_full \
    --total-updates 2000 \
    --rm-coef 0.0 \
    --kl-coef 0.0 \
    --target-kl 0.0 \
    --actor-lr 3e-5 \
    --seed "$SEED"

  echo "=== [Branch A] LLM preference collection ==="
  python collect_rm_data_llm.py \
    --dataset "$DATASET" \
    --bc-ckpt ckpt_ppo_full/ppo_rlhf_best.pt \
    --save-path dpo_data_full.pkl \
    --pairs 800 \
    --mode qwen \
    --api-key "$API_KEY" \
    --seed "$SEED"

  echo "=== [Branch A] DPO alignment ==="
  python train_dpo.py \
    --data-path dpo_data_full.pkl \
    --pi-init ckpt_ppo_full/ppo_rlhf_best.pt \
    --pi-ref ckpt_ppo_full/ppo_rlhf_best.pt \
    --save-dir ckpt_dpo_full \
    --beta 0.1 \
    --lr 5e-6 \
    --weight-decay 1e-5 \
    --epochs 10 \
    --batch-size 16 \
    --seed "$SEED"

  echo "=== [Branch A] Convert DPO ckpt for compare ==="
  python -c '
import numpy as np
import torch
ckpt = torch.load("ckpt_dpo_full/dpo_actor_best.pt", map_location="cpu")
n_reg = ckpt.get("n_region", 7)
ckpt["actor_state_dict"] = {"base_actor." + k: v for k, v in ckpt["model_state_dict"].items()}
ckpt["actor_state_dict"]["log_std"] = torch.full((n_reg, n_reg), float(np.log(0.10)), dtype=torch.float32)
torch.save(ckpt, "ckpt_dpo_full/dpo_for_eval.pt")
'
) > branch_a_dpo.log 2>&1

echo "Branch A finished. Log: branch_a_dpo.log"
