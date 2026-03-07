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
  echo "=== [Branch B] Collect RM data ==="
  python collect_rm_data_llm.py \
    --dataset "$DATASET" \
    --bc-ckpt ckpt_bc_full/bc_actor_best.pt \
    --save-path rm_data_baseline.pkl \
    --pairs 500 \
    --mode qwen \
    --api-key "$API_KEY" \
    --seed "$SEED"

  echo "=== [Branch B] Train RM ==="
  python train_rm.py \
    --data rm_data_baseline.pkl \
    --save-dir ckpt_rm_baseline \
    --epochs 30 \
    --batch-size 64 \
    --seed "$SEED"

  echo "=== [Branch B] PPO + RM ==="
  python train_ppov2_rlhf.py \
    --dataset "$DATASET" \
    --pi-init ckpt_bc_full/bc_actor_best.pt \
    --pi-ref ckpt_bc_full/bc_actor_best.pt \
    --rm-ckpt ckpt_rm_baseline/rm_best.pt \
    --save-dir ckpt_ppo_with_rm_baseline \
    --total-updates 2000 \
    --rm-coef 0.05 \
    --kl-coef 0.02 \
    --target-kl 10.0 \
    --actor-lr 1e-5 \
    --seed "$SEED"
) > branch_b_ppo_rm.log 2>&1

echo "Branch B finished. Log: branch_b_ppo_rm.log"
