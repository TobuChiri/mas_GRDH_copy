"""
Comprehensive Comparison Script for GRDH Mapping Improvements.

Tests all mapping methods (ours_mapping, bics_mapping, fale_mapping,
combined_mapping) across all attack scenarios and reports accuracy.

Usage:
  python run_comparison.py \\
      --ckpt /path/to/sd.ckpt \\
      --config ../configs/stable-diffusion/ldm.yaml \\
      --test_prompts ./test_prompts.txt \\
      --outdir ./comparison_results

Output:
  - comparison_results.json: Full results table
  - comparison_summary.txt: Readable summary
  - Per-attack, per-method accuracy tables
"""
import sys
sys.path.append('..')
import argparse
import os
import json
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from scripts.utils import gray_code
from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
import mapping_module


def cal_acc(input, gt, gray_list, bits):
    trans_fn = np.frompyfunc(lambda x: int(gray_list[int(x)], 2), 1, 1)
    count_fn = np.frompyfunc(lambda x: bin(int(x)).count('1'), 1, 1)
    a1 = trans_fn(input).astype(np.int32)
    a2 = trans_fn(gt).astype(np.int32)
    result = a1 ^ a2
    result = count_fn(result).flatten()
    shape = len(result)
    count = sum(result)
    acc = 1 - count/(int(shape)*bits)
    return acc


def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def test_mapping_method(mapping, mapping_name, model, sampler, prompts,
                         opt, gray_list, bits, attacks, device):
    """Test a single mapping method across all attacks and prompts."""
    results = {name: [] for name, _, _ in attacks}

    for idx, prompt in enumerate(tqdm(prompts, desc=f"{mapping_name:20s}")):
        # Get embeddings
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(opt.n_samples * [""])
        else:
            uc = None
        c = model.get_learned_conditioning([prompt])

        latent_shape = (opt.n_samples, 4, 64, 64)

        # Generate secret
        np.random.seed(idx)
        random_input = np.random.randint(0, 2 ** bits, latent_shape)
        seed_shuffle = np.random.randint(0, 2 ** 31 - 1, 1)
        seed_kernel = np.random.randint(0, 2 ** 31 - 1, 1)

        init_latent = mapping.encode_secret(
            secret_message=random_input,
            **({"seed_kernel": seed_kernel, "seed_shuffle": seed_shuffle}
               if mapping_name != 'tmm_mapping' else {})
        ).astype(np.float32)
        init_latent = torch.from_numpy(init_latent).to(device)

        # Generate image
        shape = init_latent.shape[1:]
        z_0, _ = sampler.sample(
            steps=opt.dpm_gen_steps,
            unconditional_conditioning=uc,
            conditioning=c,
            batch_size=opt.n_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt.scale,
            eta=opt.ddim_eta,
            order=opt.dpm_order,
            x_T=init_latent,
            width=512, height=512,
            DPMencode=False, DPMdecode=True,
        )
        x0_samples = model.decode_first_stage(z_0)

        # For each attack, extract secret and compute accuracy
        for attack_name, attack_fn, attack_factor in attacks:
            tmp_name = f"{opt.outdir}/tmp_{mapping_name}_{idx:03d}"
            attacked = attack_fn(x0_samples.clone(), factor=attack_factor, tmp_image_name=tmp_name)

            init_latent_hat = model.get_first_stage_encoding(
                model.encode_first_stage(attacked.to(device))
            )
            z_enc, _ = sampler.sample(
                steps=opt.dpm_inv_steps,
                unconditional_conditioning=uc,
                conditioning=c,
                batch_size=opt.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                eta=opt.ddim_eta,
                order=opt.dpm_order,
                x_T=init_latent_hat,
                width=512, height=512,
                DPMencode=True,
            )

            pred_noise = z_enc.clone().cpu().numpy()
            recon_args = {}
            if mapping_name not in ['simple_mapping', 'tmm_mapping']:
                recon_args = {"seed_kernel": seed_kernel, "seed_shuffle": seed_shuffle}
            recon_latent = mapping.decode_secret(pred_noise=pred_noise, **recon_args)
            acc = cal_acc(recon_latent, random_input, gray_list=gray_list, bits=bits)
            results[attack_name].append(float(acc))

    # Average results
    avg_results = {}
    for attack_name, _, _ in attacks:
        if results[attack_name]:
            avg_results[attack_name] = {
                "mean": float(np.mean(results[attack_name])),
                "std": float(np.std(results[attack_name])),
                "all": results[attack_name],
            }
    return avg_results, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="../configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--test_prompts", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./comparison_results")
    parser.add_argument("--dpm_gen_steps", type=int, default=20)
    parser.add_argument("--dpm_inv_steps", type=int, default=20)
    parser.add_argument("--dpm_order", type=int, default=2)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--bit_num", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='cuda:0')
    parser.add_argument("--precision", type=str, default="autocast",
                        choices=["full", "autocast"])
    parser.add_argument("--n_prompts", type=int, default=20,
                        help="number of prompts to test (for quick comparison)")
    parser.add_argument("--mappings", type=str, nargs="+",
                        default=["ours_mapping", "bics_mapping", "fale_mapping", "combined_mapping"],
                        help="mapping methods to compare")
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--tau_a", type=float, default=0.3)
    parser.add_argument("--tau_b", type=float, default=0.7)
    opt = parser.parse_args()

    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Using device: {device}")

    # Load model
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    model = model.to(device)
    sampler = DPMSolverSampler(model)

    # Setup
    bits = opt.bit_num
    gray_list = gray_code(bits)

    # Load prompts
    with open(opt.test_prompts, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:opt.n_prompts]
    print(f"Testing {len(prompts)} prompts")

    # Attack configurations (matching GRDH paper)
    attacks = [
        ("identity",   identity,  None),
        ("storage",    storage,   None),
        ("jpeg_90",    jpeg,      90),
        ("jpeg_70",    jpeg,      70),
        ("jpeg_50",    jpeg,      50),
        ("resize_125", resize,    1.25),
        ("resize_075", resize,    0.75),
        ("mblur_3",    mblur,     3),
        ("mblur_5",    mblur,     5),
        ("gblur_3",    gblur,     3),
        ("gblur_5",    gblur,     5),
        ("gblur_7",    gblur,     7),
        ("awgn_001",   awgn,      0.01),
        ("awgn_005",   awgn,      0.05),
        ("awgn_01",    awgn,      0.1),
    ]

    # Instantiate mapping methods
    mapping_constructors = {
        "ours_mapping": lambda: mapping_module.ours_mapping(bits=bits),
        "bics_mapping": lambda: mapping_module.bics_mapping(bits=bits, block_size=opt.block_size),
        "fale_mapping": lambda: mapping_module.fale_mapping(bits=bits, tau_a=opt.tau_a, tau_b=opt.tau_b),
        "combined_mapping": lambda: mapping_module.combined_mapping(
            bits=bits, block_size=opt.block_size, tau_a=opt.tau_a, tau_b=opt.tau_b),
        "simple_mapping": lambda: mapping_module.simple_mapping(bits=1),
        "tmm_mapping": lambda: mapping_module.tmm_mapping(bits=bits),
    }

    # Run comparisons
    all_results = {}
    for mapping_name in opt.mappings:
        if mapping_name not in mapping_constructors:
            print(f"  Unknown mapping: {mapping_name}, skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Testing: {mapping_name}")
        mapping = mapping_constructors[mapping_name]()
        avg_results, raw_results = test_mapping_method(
            mapping, mapping_name, model, sampler, prompts,
            opt, gray_list, bits, attacks, device
        )
        all_results[mapping_name] = avg_results

    # Print results table
    print("\n" + "="*70)
    print("COMPARISON RESULTS: Mean Extraction Accuracy")
    print("="*70)

    header = f"{'Attack':15s}"
    for m in all_results:
        header += f" | {m:18s}"
    print(header)
    print("-" * (15 + 20 * len(all_results)))

    for attack_name, _, _ in attacks:
        row = f"{attack_name:15s}"
        for m in all_results:
            if attack_name in all_results[m]:
                val = all_results[m][attack_name]["mean"]
                row += f" | {val*100:8.4f}%     "
            else:
                row += f" | {'N/A':>8s}        "
        print(row)

    # Save results
    output = {
        "config": vars(opt),
        "attacks": [a[0] for a in attacks],
        "results": all_results,
    }
    with open(f"{opt.outdir}/comparison_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    # Write summary text
    with open(f"{opt.outdir}/comparison_summary.txt", 'w') as f:
        f.write("GRDH Mapping Comparison Results\n")
        f.write("="*60 + "\n")
        f.write(f"Prompts: {len(prompts)}, Bits: {bits}\n\n")
        header = f"{'Attack':15s}"
        for m in all_results:
            header += f" | {m:18s}"
        f.write(header + "\n")
        f.write("-" * (15 + 20 * len(all_results)) + "\n")
        for attack_name, _, _ in attacks:
            row = f"{attack_name:15s}"
            for m in all_results:
                if attack_name in all_results[m]:
                    val = all_results[m][attack_name]["mean"]
                    row += f" | {val*100:8.4f}%     "
                else:
                    row += f" | {'N/A':>8s}        "
            f.write(row + "\n")
        f.write("\n" + "="*60 + "\n")
        # Best method per attack
        if len(all_results) >= 2:
            f.write("\nBest method per attack:\n")
            for attack_name, _, _ in attacks:
                best_method = max(all_results.keys(),
                                 key=lambda m: all_results[m].get(attack_name, {}).get("mean", -1))
                best_val = all_results[best_method].get(attack_name, {}).get("mean", 0)
                f.write(f"  {attack_name:15s} -> {best_method:20s} ({best_val*100:.2f}%)\n")

    print(f"\nResults saved to {opt.outdir}/")
    print(f"  - comparison_results.json")
    print(f"  - comparison_summary.txt")

    # Print improvement summary
    if "ours_mapping" in all_results and len(all_results) > 1:
        print("\n--- Improvement Summary vs ours_mapping ---")
        baseline = all_results["ours_mapping"]
        for m in all_results:
            if m == "ours_mapping":
                continue
            print(f"\n{m}:")
            for attack_name, _, _ in attacks:
                if attack_name in baseline and attack_name in all_results[m]:
                    base_acc = baseline[attack_name]["mean"]
                    new_acc = all_results[m][attack_name]["mean"]
                    diff = new_acc - base_acc
                    sign = "+" if diff > 0 else ""
                    print(f"  {attack_name:15s}: {base_acc*100:.2f}% -> {new_acc*100:.2f}% ({sign}{diff*100:.2f}%)")


if __name__ == '__main__':
    main()
