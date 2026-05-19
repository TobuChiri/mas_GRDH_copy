"""
ECC Comparison Test: Repetition Coding + BICS for SD Latent Steganography.

Tests whether adding repetition error correction coding (majority voting)
on top of BICS interleaving fixes the JPEG regression while preserving
blur/resize improvements.

Usage:
  python test_ecc_comparison.py --ckpt ../models/v1-5-pruned-emaonly.ckpt \
      --config ../configs/stable-diffusion/ldm.yaml \
      --test_prompts ./test_prompts.txt --n_prompts 5
"""
import sys
sys.path.append('..')
import argparse, os, json
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from scripts.utils import gray_code
from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
import mapping_module


def load_model_from_config(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="../configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--test_prompts", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./ecc_comparison")
    parser.add_argument("--dpm_gen_steps", type=int, default=20)
    parser.add_argument("--dpm_inv_steps", type=int, default=20)
    parser.add_argument("--dpm_order", type=int, default=2)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='cuda:0')
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3, help="Repetition code factor (R=3,5,7,...)")
    parser.add_argument("--block_size", type=int, default=8)
    opt = parser.parse_args()

    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Using device: {device}")

    # Load model
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    model = model.to(device)
    sampler = DPMSolverSampler(model)

    bits = 1
    gray_list = gray_code(bits)
    R = opt.repeats

    # Load prompts
    with open(opt.test_prompts, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:opt.n_prompts]
    print(f"Testing {len(prompts)} prompts with repetition code R={R}")
    print(f"Data bits per prompt: {4*64*64 // R} (out of {4*64*64} total)")

    latent_shape = (opt.n_samples, opt.C, 64, 64)
    n_total = np.prod(latent_shape)  # 16384
    n_data = n_total // R

    attacks = [
        ("identity",   identity,  None),
        ("jpeg_70",    jpeg,      70),
        ("jpeg_50",    jpeg,      50),
        ("resize_075", resize,    0.75),
        ("mblur_5",    mblur,     5),
        ("gblur_5",    gblur,     5),
        ("awgn_005",   awgn,      0.05),
        ("awgn_01",    awgn,      0.1),
    ]

    # Mapping configurations
    mappings = [
        ("ours_mapping",      mapping_module.ours_mapping(bits=bits), False),
        ("bics_mapping",      mapping_module.bics_mapping(bits=bits, block_size=opt.block_size), False),
        ("hadamard_mapping",  mapping_module.hadamard_mapping(bits=bits), False),
        ("ours+ECC(R)",       mapping_module.ours_mapping(bits=bits), True),
        ("bics+ECC(R)",       mapping_module.bics_mapping(bits=bits, block_size=opt.block_size), True),
        ("hadamard+ECC(R)",   mapping_module.hadamard_mapping(bits=bits), True),
    ]

    all_results = {}
    for mapping_name, mapping, use_ecc in mappings:
        suffix = f"_R{R}" if use_ecc else ""
        display_name = f"{mapping_name}{suffix}"
        print(f"\n{'='*60}")
        print(f"Testing: {display_name}")

        attack_results = {name: [] for name, _, _ in attacks}

        for idx, prompt in enumerate(tqdm(prompts, desc=f"{display_name:20s}")):
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            else:
                uc = None
            c = model.get_learned_conditioning([prompt])

            # Generate data bits with repetition coding
            np.random.seed(idx)
            if use_ecc:
                data_bits = np.random.randint(0, 2, n_data)  # Fewer data bits
                secret_flat = np.repeat(data_bits, R)  # Repeat each bit R times
                # Pad to exact size
                if len(secret_flat) < n_total:
                    secret_flat = np.pad(secret_flat, (0, n_total - len(secret_flat)))
                random_input = secret_flat[:n_total].reshape(latent_shape).astype(np.float64)
            else:
                random_input = np.random.randint(0, 2 ** bits, latent_shape).astype(np.float64)

            seed_shuffle = np.random.randint(0, 2 ** 31 - 1, 1)
            seed_kernel = np.random.randint(0, 2 ** 31 - 1, 1)

            init_latent = mapping.encode_secret(
                secret_message=random_input,
                seed_kernel=seed_kernel, seed_shuffle=seed_shuffle
            ).astype(np.float32)
            init_latent = torch.from_numpy(init_latent).to(device)

            # Generate image
            shape = init_latent.shape[1:]
            z_0, _ = sampler.sample(
                steps=opt.dpm_gen_steps,
                unconditional_conditioning=uc, conditioning=c,
                batch_size=opt.n_samples, shape=shape,
                verbose=False, unconditional_guidance_scale=opt.scale,
                eta=opt.ddim_eta, order=opt.dpm_order,
                x_T=init_latent, width=512, height=512,
                DPMencode=False, DPMdecode=True,
            )
            x0_samples = model.decode_first_stage(z_0)

            for attack_name, attack_fn, attack_factor in attacks:
                tmp_name = f"{opt.outdir}/tmp_{display_name}_{idx:03d}"
                attacked = attack_fn(x0_samples.clone(), factor=attack_factor, tmp_image_name=tmp_name)

                init_latent_hat = model.get_first_stage_encoding(
                    model.encode_first_stage(attacked.to(device))
                )
                z_enc, _ = sampler.sample(
                    steps=opt.dpm_inv_steps,
                    unconditional_conditioning=uc, conditioning=c,
                    batch_size=opt.n_samples, shape=shape,
                    verbose=False, unconditional_guidance_scale=opt.scale,
                    eta=opt.ddim_eta, order=opt.dpm_order,
                    x_T=init_latent_hat, width=512, height=512,
                    DPMencode=True,
                )

                pred_noise = z_enc.clone().cpu().numpy()
                recon_latent = mapping.decode_secret(
                    pred_noise=pred_noise, seed_kernel=seed_kernel, seed_shuffle=seed_shuffle
                )

                if use_ecc:
                    # Majority voting decoding
                    flat = recon_latent.flatten()
                    groups = flat[:n_data * R].reshape(-1, R)
                    decoded = (np.sum(groups, axis=1) > R / 2).astype(np.float64)
                    acc = np.mean(decoded == data_bits)
                else:
                    # Standard accuracy calculation
                    trans_fn = np.frompyfunc(lambda x: int(gray_list[int(x)], 2), 1, 1)
                    count_fn = np.frompyfunc(lambda x: bin(int(x)).count('1'), 1, 1)
                    a1 = trans_fn(recon_latent).astype(np.int32)
                    a2 = trans_fn(random_input).astype(np.int32)
                    result = a1 ^ a2
                    result = count_fn(result).flatten()
                    count = sum(result)
                    acc = 1 - count / (len(result) * bits)

                attack_results[attack_name].append(float(acc))

        # Print results for this mapping
        print(f"\n  Results for {display_name}:")
        avg_results = {}
        for attack_name, _, _ in attacks:
            vals = attack_results[attack_name]
            mean_val = np.mean(vals)
            print(f"    {attack_name:15s}: {mean_val*100:.4f}%")
            avg_results[attack_name] = {"mean": float(mean_val)}
        all_results[display_name] = avg_results

    # Print comparison table
    print("\n" + "="*70)
    print("ECC COMPARISON RESULTS")
    print("="*70)
    header = f"{'Attack':15s}"
    for m_name in [m[0] + (f"_R{R}" if m[2] else "") for m in mappings]:
        header += f" | {m_name:18s}"
    print(header)
    print("-" * (15 + 22 * len(mappings)))

    for attack_name, _, _ in attacks:
        row = f"{attack_name:15s}"
        for m_name in [m[0] + (f"_R{R}" if m[2] else "") for m in mappings]:
            if m_name in all_results and attack_name in all_results[m_name]:
                val = all_results[m_name][attack_name]["mean"]
                row += f" | {val*100:8.4f}%     "
            else:
                row += f" | {'N/A':>8s}        "
        print(row)

    # Delta table
    if len(mappings) >= 4:
        print("\n--- Deltas vs ours_mapping (baseline) ---")
        baseline = all_results.get(mappings[0][0] + ("_R" + str(R) if mappings[0][2] else ""), {})
        if not baseline:
            baseline = all_results.get(mappings[0][0], {})

        for m_idx in range(1, len(mappings)):
            m_name = mappings[m_idx][0] + (f"_R{R}" if mappings[m_idx][2] else "")
            m_res = all_results.get(m_name, {})
            if not m_res:
                continue
            print(f"\n  {m_name}:")
            for attack_name, _, _ in attacks:
                if attack_name in baseline and attack_name in m_res:
                    base = baseline[attack_name]["mean"]
                    cur = m_res[attack_name]["mean"]
                    diff = cur - base
                    sign = "+" if diff > 0 else ""
                    print(f"    {attack_name:15s}: {base*100:.2f}% -> {cur*100:.2f}% ({sign}{diff*100:.2f}%)")

    # Save results
    output = {
        "config": vars(opt),
        "attacks": [a[0] for a in attacks],
        "results": all_results,
    }
    with open(f"{opt.outdir}/ecc_comparison_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {opt.outdir}/")


if __name__ == '__main__':
    main()
