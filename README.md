# RLHF Dispatch Pricing: PPO + RM vs PPO -> DPO

This repo packages the key code used for a two-branch comparison study in a multi-region dispatch + dynamic pricing simulator.

## What this project does

- Builds a strong behavior-cloning (BC) policy as the shared expert base.
- Branch A: PPO profit breakthrough first, then LLM preference labeling, then DPO alignment.
- Branch B: Traditional RLHF pipeline with learned reward model (RM) + PPO.
- Compares BC, heuristic, PPO variants, and no-dispatch baselines with multi-seed metrics.

## Repository structure

```text
.
|- env_hybrid.py
|- collect_sft_data.py
|- train_bc.py
|- train_rm.py
|- train_ppov2_rlhf.py
|- collect_rm_data_llm.py
|- train_dpo.py
|- compare_policies.py
|- prepare_policies.py
|- requirements.txt
`- scripts
   |- run_branch_a.sh
   |- run_branch_b.sh
   `- run_compare.sh
```

## Setup

1. Create and activate environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place dataset in repo root:

```text
allarea_set_hybrid7_p12.pkl
```

## Reproduce experiment

### Step 1: shared BC foundation

```bash
python collect_sft_data.py --dataset allarea_set_hybrid7_p12.pkl --episodes 500 --save-path sft_full.pkl --seed 20240908
python train_bc.py --data sft_full.pkl --save-dir ckpt_bc_full --epochs 100 --batch-size 256 --seed 20240908
```

### Step 2: Branch A (PPO -> DPO)

```bash
bash scripts/run_branch_a.sh
```

Before running, set API key:

```bash
export DASHSCOPE_API_KEY="YOUR_KEY"
```

### Step 3: Branch B (PPO + RM baseline)

```bash
bash scripts/run_branch_b.sh
```

### Step 4: Final comparison

```bash
bash scripts/run_compare.sh
```

## Example summary from current run

Multi-seed mean (5 seeds), relative to `BC_Clone`:

- Branch A (`PPO_RLHF` from DPO eval ckpt): `total_reward +15.74%`
- `timeout_rate -0.16%`
- `empty_cost_sum -6.24%`
- `service_gini -0.68%`
