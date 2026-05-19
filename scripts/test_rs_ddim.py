"""
RS/ND Mapping test with DDIM (matching CGIS paper exactly).

Tests rs_mapping and nd_mapping using DDIM sampler for both
generation (decode) and inversion (encode).

Usage:
  conda run -n CGIS python test_rs_ddim.py \
      --ckpt ../models/v1-5-pruned-emaonly.ckpt \
      --config ../configs/stable-diffusion/ldm.yaml \
      --test_prompts test_prompts.txt --n_prompts 1
"""
import sys
sys.path.append('..')
import argparse, os, json
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
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
    parser.add_argument("--outdir", type=str, default="./rs_ddim")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps for both gen and inv")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='cuda:0')
    parser.add_argument("--n_prompts", type=int, default=1)
    opt = parser.parse_args()

    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Using device: {device}")

    # Load model
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta)
    print(f"DDIM sampler ready with {opt.ddim_steps} steps")

    # Load prompts
    with open(opt.test_prompts, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:opt.n_prompts]
    print(f"Testing {len(prompts)} prompts")

    latent_shape = (opt.n_samples, opt.C, 64, 64)
    n_total = np.prod(latent_shape)

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

    # Mapping configs
    mappings = [
        ("RS(α=0.8)",   mapping_module.rs_mapping(alpha=0.8, max_iter=100), 1),
        ("RS(α=1.0)",   mapping_module.rs_mapping(alpha=1.0, max_iter=100), 1),
        ("RS(α=1.5)",   mapping_module.rs_mapping(alpha=1.5, max_iter=100), 1),
        ("ND(1bit)",    mapping_module.nd_mapping(bits=1), 1),
        ("ND(2bit)",    mapping_module.nd_mapping(bits=2), 2),
        ("ND(3bit)",    mapping_module.nd_mapping(bits=3), 3),
    ]

    all_results = {}
    for mapping_name, mapping, data_bits in mappings:
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

            # Generate secret
            random_input = np.random.randint(0, 2 ** data_bits, latent_shape).astype(np.float64)

            # Encode secret into init latent (z_T)
            is_rs = hasattr(mapping, 'alpha')
            init_latent = mapping.encode_secret(secret_message=random_input).astype(np.float32)
            init_latent = torch.from_numpy(init_latent).to(device)

            # DDIM reverse (generation): z_T → z_0
            shape = init_latent.shape[1:]
            z_0, _ = sampler.sample(S=opt.ddim_steps,
                                    batch_size=opt.n_samples,
                                    shape=shape,
                                    conditioning=c,
                                    unconditional_conditioning=uc,
                                    unconditional_guidance_scale=opt.scale,
                                    eta=opt.ddim_eta,
                                    x_T=init_latent,
                                    verbose=False)
            x0_samples = model.decode_first_stage(z_0)

            for attack_name, attack_fn, attack_factor in attacks:
                tmp_name = f"{opt.outdir}/tmp_{idx:03d}"
                attacked = attack_fn(x0_samples.clone(), factor=attack_factor, tmp_image_name=tmp_name)

                # VAE encode the attacked image → z_0_hat
                init_latent_hat = model.get_first_stage_encoding(
                    model.encode_first_stage(attacked.to(device))
                )

                # DDIM forward (inversion): z_0_hat → z_T_hat
                z_enc, _ = sampler.sample(S=opt.ddim_steps,
                                          batch_size=opt.n_samples,
                                          shape=shape,
                                          conditioning=c,
                                          unconditional_conditioning=uc,
                                          unconditional_guidance_scale=opt.scale,
                                          eta=opt.ddim_eta,
                                          x_T=init_latent_hat,
                                          verbose=False,
                                          encode=True)

                pred_noise = z_enc.clone().cpu().numpy()
                recon_latent = mapping.decode_secret(pred_noise=pred_noise)

                # Accuracy
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

        # Print results
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
    print("RS vs ND MAPPING WITH DDIM")
    print("="*70)
    header = f"{'Attack':15s}"
    for m_name, _, _ in mappings:
        header += f" | {m_name:18s}"
    print(header)
    print("-" * (15 + 22 * len(mappings)))

    for attack_name, _, _ in attacks:
        row = f"{attack_name:15s}"
        for m_name, _, _ in mappings:
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
    with open(f"{opt.outdir}/results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {opt.outdir}/")


if __name__ == '__main__':
    main()
