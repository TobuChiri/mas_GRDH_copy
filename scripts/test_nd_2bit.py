"""Test ND 2-bit with optimal parameter tuning."""
import sys; sys.path.append('..')
import numpy as np, json
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import mapping_module

CKPT = '../models/v1-5-pruned-emaonly.ckpt'
CONFIG = '../configs/stable-diffusion/ldm.yaml'
device = torch.device("cuda:0")

config = OmegaConf.load(CONFIG)
pl_sd = torch.load(CKPT, map_location=device, weights_only=False)
model = instantiate_from_config(config.model)
model.load_state_dict(pl_sd['state_dict'], strict=False)
model.eval()
model = model.to(device)
model.cond_stage_model = model.cond_stage_model.to(device)

sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0)

# Also test a 2-bit config with smaller range
configs = [
    ("ND(2bit,σ=0.1,[-3,3])",  dict(bits=2, sigma=0.1, a=-3.0, b=3.0)),
    ("ND(2bit,σ=0.15,[-3,3])", dict(bits=2, sigma=0.15, a=-3.0, b=3.0)),
    ("ND(2bit,σ=0.2,[-3,3])",  dict(bits=2, sigma=0.2, a=-3.0, b=3.0)),
    ("ND(2bit,σ=0.3,[-3,3])",  dict(bits=2, sigma=0.3, a=-3.0, b=3.0)),
    ("ND(2bit,σ=0.1,[-4,4])",  dict(bits=2, sigma=0.1, a=-4.0, b=4.0)),
    ("ND(2bit,σ=0.15,[-4,4])", dict(bits=2, sigma=0.15, a=-4.0, b=4.0)),
    ("ND(2bit,σ=0.1,[-2,2])",  dict(bits=2, sigma=0.1, a=-2.0, b=2.0)),
    ("1bit baseline ([−3,3],σ=0.3)", dict(bits=1, sigma=0.3, a=-3.0, b=3.0)),
]

tests = [
    "a photo of a cat",
    "a beautiful landscape",
    "a portrait of a dog",
]

uc = model.get_learned_conditioning([""])
latent_shape = (1, 4, 64, 64)

results = {}
for cfg_name, nd_kw in configs:
    nd = mapping_module.nd_mapping(**nd_kw)
    n_vals = nd.n_intervals
    acc_list = []

    for prompt in tests:
        c = model.get_learned_conditioning([prompt])
        np.random.seed(42)

        secret = np.random.randint(0, n_vals, latent_shape).astype(np.float64)
        init_latent = nd.encode_secret(secret).astype(np.float32)
        init_latent_t = torch.from_numpy(init_latent).to(device)

        with torch.no_grad():
            z0, _ = sampler.sample(S=50, batch_size=1, shape=(4, 64, 64),
                conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=5.0, eta=0.0,
                x_T=init_latent_t, verbose=False)
            x0 = model.decode_first_stage(z0)
            z_hat = model.get_first_stage_encoding(model.encode_first_stage(x0))
            z_enc, _ = sampler.sample(S=50, batch_size=1, shape=(4, 64, 64),
                conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=5.0, eta=0.0,
                x_T=z_hat, verbose=False, encode=True)

        recon = z_enc.cpu().numpy()
        diff = recon - init_latent
        acc = float(np.mean(nd.decode_secret(recon) == secret))
        acc_list.append(acc)
        print(f"{cfg_name:40s} | {prompt:30s} | acc={acc*100:.2f}% | diff_std={np.std(diff):.4f}")

    mean_acc = np.mean(acc_list)
    print(f"{'─'*90}\n{'':40s} | {'MEAN':30s} | {mean_acc*100:.2f}%\n")
    results[cfg_name] = dict(mean_acc=mean_acc, per_prompt=acc_list,
                             diff_std=float(np.std(diff)))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for cfg_name, res in sorted(results.items(), key=lambda x: -x[1]['mean_acc']):
    print(f"  {cfg_name:40s}: {res['mean_acc']*100:.2f}%")

with open("./nd_multibit/nd_2bit_results.json", 'w') as f:
    json.dump(results, f, indent=2)
