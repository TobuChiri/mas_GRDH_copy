"""
Frequency Robustness Analyzer for SD Latent Space.

Analyzes which spatial-frequency regions of SD's VAE latent space
are most robust to common image attacks (JPEG, blur, noise, resize).

This script runs the GRDH pipeline for a set of prompts and attacks,
records per-frequency BER, and builds an empirical robustness profile
that can be used by fale_mapping / combined_mapping.

Usage:
  python analyze_frequency_robustness.py \\
      --ckpt /path/to/sd.ckpt \\
      --config ../configs/stable-diffusion/ldm.yaml \\
      --test_prompts ./test_prompts.txt \\
      --outdir ./freq_analysis
"""
import sys
sys.path.append('..')
import argparse
import os
import json
import numpy as np
from scipy.fft import dct, idct
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from scripts.utils import gray_code, load_512, image_grid
from robust_eval import jpeg, gblur, resize, awgn, mblur, storage
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


def analyze_latent_frequency(init_latent, z_enc):
    """
    Compare original and reconstructed latent in frequency domain.
    Returns per-frequency error map.
    """
    _, C, H, W = init_latent.shape
    freq_error = np.zeros((H, W), dtype=np.float64)
    for c in range(C):
        orig_dct = dct(dct(init_latent[0, c].cpu().numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')
        recon_dct = dct(dct(z_enc[0, c].cpu().numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')
        freq_error += np.abs(orig_dct - recon_dct)
    return freq_error / C  # average over channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="../configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--test_prompts", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./freq_analysis")
    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--dpm_order", type=int, default=2)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--bit_num", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='cuda:0')
    parser.add_argument("--n_prompts", type=int, default=50, help="number of prompts to test")
    opt = parser.parse_args()

    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(opt.outdir, exist_ok=True)

    # Load model
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    model = model.to(device)
    sampler = DPMSolverSampler(model)

    # Setup mapping
    bits = opt.bit_num
    gray_list = gray_code(bits)
    mapping = mapping_module.ours_mapping(bits=bits)

    # Load prompts
    with open(opt.test_prompts, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:opt.n_prompts]

    # Attack configurations to test
    attacks = [
        ("jpeg_90", lambda x: jpeg(x, 90, f"{opt.outdir}/tmp")),
        ("jpeg_70", lambda x: jpeg(x, 70, f"{opt.outdir}/tmp")),
        ("jpeg_50", lambda x: jpeg(x, 50, f"{opt.outdir}/tmp")),
        ("gblur_5", lambda x: gblur(x, 5, f"{opt.outdir}/tmp")),
        ("awgn_005", lambda x: awgn(x, 0.05, f"{opt.outdir}/tmp")),
        ("resize_075", lambda x: resize(x, 0.75, f"{opt.outdir}/tmp")),
        ("identity", lambda x: x),
    ]

    # Results accumulator
    # freq_error_maps[attack_name] = list of (H,W) numpy arrays
    freq_error_maps = {name: [] for name, _ in attacks}

    print(f"Analyzing {len(prompts)} prompts over {len(attacks)} attack conditions...")
    print(f"Results will be saved to {opt.outdir}/")

    with torch.no_grad():
        for idx, prompt in enumerate(tqdm(prompts)):
            # Get embeddings
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            else:
                uc = None
            c = model.get_learned_conditioning([prompt])

            latent_shape = (opt.n_samples, 4, 64, 64)

            # Generate secret and encode to latent
            np.random.seed(idx)  # reproducible
            random_input = np.random.randint(0, 2 ** bits, latent_shape)
            seed_shuffle = np.random.randint(0, 2 ** 31 - 1, 1)
            seed_kernel = np.random.randint(0, 2 ** 31 - 1, 1)
            init_latent = mapping.encode_secret(
                secret_message=random_input,
                seed_kernel=seed_kernel,
                seed_shuffle=seed_shuffle
            ).astype(np.float32)
            init_latent = torch.from_numpy(init_latent).to(device)

            # Generate image
            shape = init_latent.shape[1:]
            z_0, _ = sampler.sample(
                steps=opt.dpm_steps,
                unconditional_conditioning=uc,
                conditioning=c,
                batch_size=opt.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                eta=0.0,
                order=opt.dpm_order,
                x_T=init_latent,
                width=512,
                height=512,
                DPMencode=False,
                DPMdecode=True,
            )
            x0_samples = model.decode_first_stage(z_0)

            # For each attack, re-encode and analyze frequency errors
            tmp_base = f"{opt.outdir}/tmp_{idx:03d}"
            for attack_name, attack_fn in attacks:
                # Apply attack
                attacked = attack_fn(x0_samples.clone())

                # Re-encode
                init_latent_hat = model.get_first_stage_encoding(
                    model.encode_first_stage(attacked.to(device))
                )
                z_enc, _ = sampler.sample(
                    steps=opt.dpm_steps,
                    unconditional_conditioning=uc,
                    conditioning=c,
                    batch_size=opt.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    eta=0.0,
                    order=opt.dpm_order,
                    x_T=init_latent_hat,
                    width=512,
                    height=512,
                    DPMencode=True,
                )

                # Analyze frequency error
                freq_err = analyze_latent_frequency(init_latent, z_enc)
                freq_error_maps[attack_name].append(freq_err)

    # Compute aggregate statistics
    print("\n=== Frequency Robustness Analysis Results ===")
    results = {}
    for attack_name, err_maps in freq_error_maps.items():
        if not err_maps:
            continue
        # Average over all prompts
        avg_err = np.mean(err_maps, axis=0)  # (H, W)
        # Summary statistics
        total_err = np.mean(avg_err)
        # Divide into 4 frequency quadrants
        H, W = avg_err.shape
        h_half, w_half = H // 4, W // 4
        low_low = np.mean(avg_err[:h_half, :w_half])       # LL
        low_high = np.mean(avg_err[:h_half, w_half:])      # LH
        high_low = np.mean(avg_err[h_half:, :w_half])      # HL
        high_high = np.mean(avg_err[h_half:, w_half:])     # HH

        results[attack_name] = {
            "total_err": float(total_err),
            "ll_err": float(low_low),
            "lh_err": float(low_high),
            "hl_err": float(high_low),
            "hh_err": float(high_high),
        }
        print(f"  {attack_name:15s} | total: {total_err:.6f} | LL: {low_low:.6f} LH: {low_high:.6f} HL: {high_low:.6f} HH: {high_high:.6f}")

    # Save results
    with open(f"{opt.outdir}/frequency_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Build and save empirical robustness profile
    # Low error = high robustness = higher weight
    if "jpeg_70" in freq_error_maps and freq_error_maps["jpeg_70"]:
        profile = 1.0 / (np.mean(freq_error_maps["jpeg_70"], axis=0) + 1e-8)
        profile = profile / np.max(profile)  # normalize to [0, 1]
        np.save(f"{opt.outdir}/frequency_robustness_profile.npy", profile)
        print(f"\nRobustness profile saved to {opt.outdir}/frequency_robustness_profile.npy")
        print(f"  Shape: {profile.shape}, Range: [{profile.min():.4f}, {profile.max():.4f}]")

    print(f"\nFull results saved to {opt.outdir}/frequency_analysis_results.json")
    print(f"Run with --outdir <path> to specify output location.")


if __name__ == '__main__':
    main()
