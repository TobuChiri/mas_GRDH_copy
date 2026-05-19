"""Test if image denoising (GMM-ID proxy) helps ND mapping accuracy."""
import sys; sys.path.append('..')
import argparse, os, numpy as np, json
from omegaconf import OmegaConf
import torch
from skimage.restoration import denoise_nl_means
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import mapping_module

def load_model(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location=device, weights_only=False)
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd['state_dict'], strict=False)
    model.eval()
    return model

def apply_denoise(img_tensor, sigma=0.1):
    """Apply Non-local Means denoising to an image tensor [1,C,H,W] in [-1,1]."""
    img_np = img_tensor.detach().cpu().numpy()
    denoised = np.zeros_like(img_np)
    for i in range(img_np.shape[1]):
        denoised[0, i] = denoise_nl_means(img_np[0, i], h=sigma * 2, patch_size=5, patch_distance=3)
    return torch.from_numpy(denoised).to(img_tensor.device)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--config", type=str, default="../configs/stable-diffusion/ldm.yaml")
parser.add_argument("--test_prompts", type=str, required=True)
parser.add_argument("--outdir", type=str, default="./denoise_test")
parser.add_argument("--ddim_steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=5.0)
parser.add_argument("--n_prompts", type=int, default=1)
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(opt.outdir, exist_ok=True)
config = OmegaConf.load(opt.config)
model = load_model(config, opt.ckpt, device).to(device)
sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=0.0)

with open(opt.test_prompts, 'r') as f:
    prompts = [l.strip() for l in f if l.strip()][:opt.n_prompts]

from robust_eval import identity, jpeg, resize, mblur, gblur, awgn
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

configs = [
    ("RS(α=0.8) no denoise", mapping_module.rs_mapping(alpha=0.8), False),
    ("ND(σ=1.5) no denoise", mapping_module.nd_mapping(bits=1, sigma=1.5), False),
    ("ND(σ=1.5) + denoise",  mapping_module.nd_mapping(bits=1, sigma=1.5), True),
    ("ND(σ=2.0) + denoise",  mapping_module.nd_mapping(bits=1, sigma=2.0), True),
]

all_results = {}
for cfg_name, mapping, use_denoise in configs:
    print(f"\n{'='*60}\n{cfg_name}")
    attack_results = {a[0]: [] for a in attacks}

    for idx, prompt in enumerate(prompts):
        uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
        c = model.get_learned_conditioning([prompt])
        np.random.seed(idx)

        secret = np.random.randint(0, 2, (1, 4, 64, 64)).astype(np.float64)
        init_latent = torch.from_numpy(mapping.encode_secret(secret_message=secret).astype(np.float32)).to(device)

        shape = init_latent.shape[1:]
        z0, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=opt.scale, eta=0.0, x_T=init_latent, verbose=False)
        x0 = model.decode_first_stage(z0)

        for aname, afn, afac in attacks:
            tmp = f"{opt.outdir}/tmp_{idx:03d}"
            attacked = afn(x0.clone(), factor=afac, tmp_image_name=tmp)

            # ★ GMM-ID proxy: denoise attacked image before VAE encode ★
            if use_denoise:
                attacked = apply_denoise(attacked)

            z_hat = model.get_first_stage_encoding(model.encode_first_stage(attacked.to(device)))
            z_enc, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
                conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=opt.scale, eta=0.0, x_T=z_hat, verbose=False, encode=True)

            recon = mapping.decode_secret(pred_noise=z_enc.clone().cpu().numpy())
            acc = float(np.mean(recon == secret))
            attack_results[aname].append(acc)

    print(f"  Results for {cfg_name}:")
    for aname, _, _ in attacks:
        mean_val = np.mean(attack_results[aname])
        print(f"    {aname:15s}: {mean_val*100:.2f}%")
    all_results[cfg_name] = {a: {"mean": float(np.mean(attack_results[a]))} for a, _, _ in attacks}

# Summary
print("\n" + "="*70)
print("DENOISE EFFECT ON ND MAPPING")
print("="*70)
for cfg_name, _, _ in configs:
    print(f"\n{cfg_name}:")
    for aname, _, _ in attacks:
        print(f"  {aname:15s}: {all_results[cfg_name][aname]['mean']*100:.2f}%")

output = {"results": all_results}
with open(f"{opt.outdir}/denoise_results.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {opt.outdir}/")
