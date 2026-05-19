"""
RS Mapping Pilot: Test rejection sampling-based embedding methods.

Tests rs_mapping (binary RS), mlq_rs_mapping (multi-bit RS),
and rs_ecc_mapping (RS + repetition coding) against QR baseline.

Usage:
  E:/libraries/anaconda/envs/CGIS/python.exe test_rs_comparison.py \
      --ckpt ../models/v1-5-pruned-emaonly.ckpt \
      --config ../configs/stable-diffusion/ldm.yaml \
      --test_prompts test_prompts.txt --n_prompts 3
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
    parser.add_argument("--outdir", type=str, default="./rs_comparison")
    parser.add_argument("--dpm_gen_steps", type=int, default=20)
    parser.add_argument("--dpm_inv_steps", type=int, default=20)
    parser.add_argument("--dpm_order", type=int, default=2)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='cuda:0')
    parser.add_argument("--n_prompts", type=int, default=3)
    opt = parser.parse_args()

    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Using device: {device}")

    # Load model
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    model = model.to(device)
    sampler = DPMSolverSampler(model)

    # Load prompts
    with open(opt.test_prompts, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:opt.n_prompts]
    print(f"Testing {len(prompts)} prompts")

    latent_shape = (opt.n_samples, opt.C, 64, 64)
    n_total = np.prod(latent_shape)  # 16384

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

    # Mapping configurations to test
    # (name, mapping_obj, bits_for_data, use_ecc)
    mappings = [
        ("QR baseline",       mapping_module.ours_mapping(bits=1), 1, False),
        ("RS(α=0.8)",         mapping_module.rs_mapping(alpha=0.8, max_iter=100), 1, False),
        ("RS(α=1.0)",         mapping_module.rs_mapping(alpha=1.0, max_iter=100), 1, False),
        ("RS(α=1.5)",         mapping_module.rs_mapping(alpha=1.5, max_iter=100), 1, False),
        ("ND(1bit)",          mapping_module.nd_mapping(bits=1), 1, False),
        ("ND(2bit)",          mapping_module.nd_mapping(bits=2), 2, False),
        ("ND(3bit)",          mapping_module.nd_mapping(bits=3), 3, False),
        ("ND(4bit)",          mapping_module.nd_mapping(bits=4), 4, False),
    ]

    all_results = {}
    for mapping_name, mapping, data_bits, use_ecc in mappings:
        print(f"\n{'='*60}")
        print(f"Testing: {mapping_name}")

        attack_results = {name: [] for name, _, _ in attacks}

        for idx, prompt in enumerate(tqdm(prompts, desc=f"{mapping_name:20s}")):
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            else:
                uc = None
            c = model.get_learned_conditioning([prompt])

            np.random.seed(idx)

            if use_ecc:
                # rs_ecc_mapping: data bits are the secret message
                n_data = n_total // mapping.repeats
                data_bits_arr = np.random.randint(0, 2, n_data).astype(np.float64)
                random_input = np.pad(data_bits_arr, (0, n_total - n_data),
                                      'constant')[:n_total].reshape(latent_shape)
            else:
                # Standard: generate random secret
                random_input = np.random.randint(0, 2 ** data_bits, latent_shape).astype(np.float64)

            # For RS mapping, no seed_kernel/shuffle needed
            if hasattr(mapping, 'alpha') or 'rs_' in mapping_name.lower().replace('-', '_'):
                init_latent = mapping.encode_secret(secret_message=random_input).astype(np.float32)
            else:
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
                tmp_name = f"{opt.outdir}/tmp_{idx:03d}"
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

                if hasattr(mapping, 'alpha') or 'rs_' in mapping_name.lower().replace('-', '_'):
                    recon_latent = mapping.decode_secret(pred_noise=pred_noise)
                else:
                    recon_latent = mapping.decode_secret(
                        pred_noise=pred_noise, seed_kernel=seed_kernel, seed_shuffle=seed_shuffle
                    )

                if use_ecc:
                    # Majority voting decoding for RS-ECC
                    flat = recon_latent.flatten()
                    n_data = n_total // mapping.repeats
                    groups = flat[:n_data * mapping.repeats].reshape(-1, mapping.repeats)
                    decoded = (np.sum(groups, axis=1) > mapping.repeats / 2).astype(np.float64)
                    acc = np.mean(decoded == data_bits_arr)
                else:
                    # Standard accuracy
                    gray_list = gray_code(data_bits)
                    trans_fn = np.frompyfunc(lambda x: int(gray_list[int(x)], 2), 1, 1)
                    count_fn = np.frompyfunc(lambda x: bin(int(x)).count('1'), 1, 1)
                    a1 = trans_fn(recon_latent).astype(np.int32)
                    a2 = trans_fn(random_input).astype(np.int32)
                    result = a1 ^ a2
                    result = count_fn(result).flatten()
                    count = sum(result)
                    acc = 1 - count / (len(result) * data_bits)

                attack_results[attack_name].append(float(acc))

        # Print results for this mapping
        print(f"\n  Results for {mapping_name}:")
        for attack_name, _, _ in attacks:
            vals = attack_results[attack_name]
            mean_val = np.mean(vals)
            print(f"    {attack_name:15s}: {mean_val*100:.4f}%")

        all_results[mapping_name] = {
            a: {"mean": float(np.mean(attack_results[a]))}
            for a, _, _ in attacks
        }

    # Print comparison table
    print("\n" + "="*70)
    print("RS MAPPING COMPARISON RESULTS")
    print("="*70)
    header = f"{'Attack':15s}"
    for m_name, _, _, _ in mappings:
        header += f" | {m_name:18s}"
    print(header)
    print("-" * (15 + 22 * len(mappings)))

    for attack_name, _, _ in attacks:
        row = f"{attack_name:15s}"
        for m_name, _, _, _ in mappings:
            if m_name in all_results and attack_name in all_results[m_name]:
                val = all_results[m_name][attack_name]["mean"]
                row += f" | {val*100:8.4f}%     "
            else:
                row += f" | {'N/A':>8s}        "
        print(row)

    # Save
    output = {
        "config": vars(opt),
        "attacks": [a[0] for a in attacks],
        "results": all_results,
    }
    with open(f"{opt.outdir}/rs_comparison_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {opt.outdir}/")


if __name__ == '__main__':
    main()
