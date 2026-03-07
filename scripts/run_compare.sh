#!/usr/bin/env bash
set -euo pipefail

DATASET="allarea_set_hybrid7_p12.pkl"

echo "=== Compare baseline PPO+RM branch ==="
python compare_policies.py \
  --dataset "$DATASET" \
  --pi-init-ckpt ckpt_bc_full/bc_actor_best.pt \
  --ppo-ckpt ckpt_ppo_with_rm_baseline/ppo_rlhf_best.pt \
  --num-seeds 5 \
  --max-steps 288

echo "=== Compare DPO branch ==="
python compare_policies.py \
  --dataset "$DATASET" \
  --pi-init-ckpt ckpt_bc_full/bc_actor_best.pt \
  --ppo-ckpt ckpt_dpo_full/dpo_for_eval.pt \
  --num-seeds 5 \
  --max-steps 288
