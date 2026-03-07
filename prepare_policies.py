import argparse
import copy
import os

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Create pi_init and pi_ref from BC checkpoint."
    )
    parser.add_argument("--bc-ckpt", type=str, default="checkpoints_bc/bc_actor_best.pt")
    parser.add_argument("--out-dir", type=str, default="checkpoints_rlhf")
    args = parser.parse_args()

    if not os.path.exists(args.bc_ckpt):
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = torch.load(args.bc_ckpt, map_location="cpu")

    pi_init = copy.deepcopy(ckpt)
    pi_ref = copy.deepcopy(ckpt)
    pi_init["policy_role"] = "pi_init"
    pi_ref["policy_role"] = "pi_ref"
    pi_ref["frozen"] = True

    init_path = os.path.join(args.out_dir, "pi_init.pt")
    ref_path = os.path.join(args.out_dir, "pi_ref.pt")
    torch.save(pi_init, init_path)
    torch.save(pi_ref, ref_path)

    print(f"Saved: {init_path}")
    print(f"Saved: {ref_path}")


if __name__ == "__main__":
    main()

