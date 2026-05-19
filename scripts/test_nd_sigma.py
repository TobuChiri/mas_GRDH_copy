"""Sweep ND mapping sigma values under DDIM."""
import sys; sys.path.append('..')
import argparse, os, numpy as np, json
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.utils import gray_code
from robust_eval import identity, jpeg, resize, mblur, gblur, awgn
import mapping_module

def load_model(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location=device, weights_only=False)
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd['state_dict'], strict=False)
    model.eval()
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--config", type=str, default="../configs/stable-diffusion/ldm.yaml")
parser.add_argument("--test_prompts", type=str, required=True)
parser.add_argument("--outdir", type=str, default="./nd_sigma_sweep")
parser.add_argument("--ddim_steps", type=int, default=50)
parser.add_argument("--ddim_eta", type=float, default=0.0)
parser.add_argument("--scale", type=float, default=5.0)
parser.add_argument("--n_prompts", type=int, default=1)
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(opt.outdir, exist_ok=True)
config = OmegaConf.load(opt.config)
model = load_model(config, opt.ckpt, device).to(device)
sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta)

with open(opt.test_prompts, 'r') as f:
    prompts = [l.strip() for l in f if l.strip()][:opt.n_prompts]

latent_shape = (1, 4, 64, 64)

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

# Sweep ND(1bit) with different sigmas
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

all_results = {}
for sigma in sigmas:
    name = f"ND(1bit,σ={sigma})"
    print(f"\n{'='*60}\n{name}")
    mapping = mapping_module.nd_mapping(bits=1, sigma=sigma)

    attack_results = {a[0]: [] for a in attacks}
    for idx, prompt in enumerate(prompts):
        uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
        c = model.get_learned_conditioning([prompt])
        np.random.seed(idx)

        random_input = np.random.randint(0, 2, latent_shape).astype(np.float64)
        init_latent = mapping.encode_secret(secret_message=random_input).astype(np.float32)
        init_latent = torch.from_numpy(init_latent).to(device)

        shape = init_latent.shape[1:]
        z_0, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=opt.scale, eta=opt.ddim_eta,
            x_T=init_latent, verbose=False)
        x0 = model.decode_first_stage(z_0)

        for aname, afn, afac in attacks:
            tmp = f"{opt.outdir}/tmp_{idx:03d}"
            attacked = afn(x0.clone(), factor=afac, tmp_image_name=tmp)

            z_hat = model.get_first_stage_encoding(model.encode_first_stage(attacked.to(device)))
            z_enc, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
                conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=opt.scale, eta=opt.ddim_eta,
                x_T=z_hat, verbose=False, encode=True)

            recon = mapping.decode_secret(pred_noise=z_enc.clone().cpu().numpy())
            acc = np.mean(recon == random_input)
            attack_results[aname].append(float(acc))

    print(f"  Results for {name}:")
    for aname, _, _ in attacks:
        mean_val = np.mean(attack_results[aname])
        print(f"    {aname:15s}: {mean_val*100:.2f}%")

    all_results[name] = {a: {"mean": float(np.mean(attack_results[a]))} for a, _, _ in attacks}

# Print summary table
print("\n" + "="*70)
print("ND MAPPING SIGMA SWEEP RESULTS")
print("="*70)
header = f"{'Attack':15s}"
for s in sigmas:
    header += f" | {'σ='+str(s):>10s}"
print(header)
print("-" * (15 + 14 * len(sigmas)))
for aname, _, _ in attacks:
    row = f"{aname:15s}"
    for s in sigmas:
        key = f"ND(1bit,σ={s})"
        val = all_results[key][aname]["mean"]
        row += f" | {val*100:8.2f}%"
    print(row)

output = {"config": vars(opt), "attacks": [a[0] for a in attacks], "results": all_results}
with open(f"{opt.outdir}/nd_sigma_results.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {opt.outdir}/")
