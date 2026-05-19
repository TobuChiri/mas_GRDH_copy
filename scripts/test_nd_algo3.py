"""Test ND mapping from CGIS Algorithm 3 with full DDIM pipeline."""
import sys; sys.path.append('..')
import argparse, os, numpy as np, json
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
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
parser.add_argument("--outdir", type=str, default="./nd_algo3")
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

# ND configs: explore [a,b] range and sigma
nd_configs = [
    ("ND(1bit,σ=0.3,[-1,1])",  dict(bits=1, sigma=0.3, a=-1.0, b=1.0)),
    ("ND(1bit,σ=0.5,[-1,1])",  dict(bits=1, sigma=0.5, a=-1.0, b=1.0)),
    ("ND(1bit,σ=0.3,[-2,2])",  dict(bits=1, sigma=0.3, a=-2.0, b=2.0)),
    ("ND(1bit,σ=0.5,[-2,2])",  dict(bits=1, sigma=0.5, a=-2.0, b=2.0)),
    ("ND(1bit,σ=0.3,[-3,3])",  dict(bits=1, sigma=0.3, a=-3.0, b=3.0)),
    ("ND(1bit,σ=0.5,[-3,3])",  dict(bits=1, sigma=0.5, a=-3.0, b=3.0)),
]
# RS(alpha=0.01) as baseline
RS_ALPHA = 0.01

all_results = {}

# RS baseline
rs = mapping_module.rs_mapping(alpha=RS_ALPHA)
cfg_name = f"RS(α={RS_ALPHA})"
print(f"\n{'='*60}\n{cfg_name}")
attack_results = {a[0]: [] for a in attacks}

for idx, prompt in enumerate(prompts):
    uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
    c = model.get_learned_conditioning([prompt])
    np.random.seed(idx)

    random_input = np.random.randint(0, 2, latent_shape).astype(np.float64)
    init_latent = rs.encode_secret(secret_message=random_input).astype(np.float32)
    init_latent = torch.from_numpy(init_latent).to(device)

    shape = init_latent.shape[1:]
    z0, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
        conditioning=c, unconditional_conditioning=uc,
        unconditional_guidance_scale=opt.scale, eta=0.0, x_T=init_latent, verbose=False)
    x0 = model.decode_first_stage(z0)

    for aname, afn, afac in attacks:
        tmp = f"{opt.outdir}/tmp_{idx:03d}"
        attacked = afn(x0.clone(), factor=afac, tmp_image_name=tmp)
        z_hat = model.get_first_stage_encoding(model.encode_first_stage(attacked.to(device)))
        z_enc, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=opt.scale, eta=0.0, x_T=z_hat, verbose=False, encode=True)
        recon = rs.decode_secret(pred_noise=z_enc.clone().cpu().numpy())
        acc = float(np.mean(recon == random_input))
        attack_results[aname].append(acc)

for aname, _, _ in attacks:
    print(f"  {aname:15s}: {np.mean(attack_results[aname])*100:.2f}%")
all_results[cfg_name] = {a: {"mean": float(np.mean(attack_results[a]))} for a, _, _ in attacks}

# ND configs
for cfg_name, nd_kwargs in nd_configs:
    print(f"\n{'='*60}\n{cfg_name}")
    nd = mapping_module.nd_mapping(**nd_kwargs)
    attack_results = {a[0]: [] for a in attacks}

    for idx, prompt in enumerate(prompts):
        uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
        c = model.get_learned_conditioning([prompt])
        np.random.seed(idx)

        random_input = np.random.randint(0, 2, latent_shape).astype(np.float64)
        init_latent = nd.encode_secret(secret_message=random_input).astype(np.float32)
        init_latent = torch.from_numpy(init_latent).to(device)

        shape = init_latent.shape[1:]
        z0, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=opt.scale, eta=0.0, x_T=init_latent, verbose=False)
        x0 = model.decode_first_stage(z0)

        for aname, afn, afac in attacks:
            tmp = f"{opt.outdir}/tmp_{idx:03d}"
            attacked = afn(x0.clone(), factor=afac, tmp_image_name=tmp)
            z_hat = model.get_first_stage_encoding(model.encode_first_stage(attacked.to(device)))
            z_enc, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
                conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=opt.scale, eta=0.0, x_T=z_hat, verbose=False, encode=True)
            recon = nd.decode_secret(pred_noise=z_enc.clone().cpu().numpy())
            acc = float(np.mean(recon == random_input))
            attack_results[aname].append(acc)

    for aname, _, _ in attacks:
        print(f"  {aname:15s}: {np.mean(attack_results[aname])*100:.2f}%")
    all_results[cfg_name] = {a: {"mean": float(np.mean(attack_results[a]))} for a, _, _ in attacks}

# Summary
print("\n" + "="*70)
print("ND ALGORITHM 3 RESULTS")
print("="*70)
for cfg_name in list(all_results.keys()):
    print(f"\n{cfg_name}:")
    for aname, _, _ in attacks:
        print(f"  {aname:15s}: {all_results[cfg_name][aname]['mean']*100:.2f}%")

output = {"results": all_results}
with open(f"{opt.outdir}/nd_algo3_results.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {opt.outdir}/")
