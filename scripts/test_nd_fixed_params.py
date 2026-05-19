"""Compare ND bits=1,2,3 with FIXED [a,b] and sigma."""
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

# Fixed parameters across all bit depths
A, B = -3.0, 3.0
SIGMA = 0.1
configs = [
    (f"ND(1bit,σ={SIGMA},[{A:.0f},{B:.0f}])", dict(bits=1, sigma=SIGMA, a=A, b=B)),
    (f"ND(2bit,σ={SIGMA},[{A:.0f},{B:.0f}])", dict(bits=2, sigma=SIGMA, a=A, b=B)),
    (f"ND(3bit,σ={SIGMA},[{A:.0f},{B:.0f}])", dict(bits=3, sigma=SIGMA, a=A, b=B)),
]

# Extra comparison: also test [-2,2] which keeps values in N(0,1) range
A2, B2 = -2.0, 2.0
for bits in [1, 2]:
    configs.append((f"ND({bits}bit,σ={SIGMA},[{A2:.0f},{B2:.0f}])", dict(bits=bits, sigma=SIGMA, a=A2, b=B2)))

prompts = [
    "a photo of a cat",
    "a beautiful landscape",
    "a portrait of a dog",
    "a bowl of fruit",
    "a mountain view at sunset",
]

uc = model.get_learned_conditioning([""])
latent_shape = (1, 4, 64, 64)

results = {}
for cfg_name, nd_kw in configs:
    nd = mapping_module.nd_mapping(**nd_kw)
    n_vals = nd.n_intervals
    acc_list = []
    diff_stds = []

    for prompt in prompts:
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
        diff_stds.append(float(np.std(diff)))

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    mean_diff_std = np.mean(diff_stds)
    print(f"{cfg_name:40s}  acc={mean_acc*100:.2f}% +/-{std_acc*100:.2f}  noise_std={mean_diff_std:.4f}  (range [{nd.a:.0f},{nd.b:.0f}], delta={nd.delta:.4f})")
    results[cfg_name] = dict(mean_acc=mean_acc, std_acc=std_acc, mean_diff_std=mean_diff_std,
                             a=nd.a, b=nd.b, delta=nd.delta, bits=nd.bits)

print("\n" + "=" * 70)
print("COMPARISON: FIXED [a,b] AND sigma ACROSS BIT DEPTHS")
print("=" * 70)
print(f"{'Config':40s} {'Bits':>5} {'Range':>10} {'Delta':>8} {'Noise_std':>10} {'Accuracy':>10}")
print("-" * 83)
for cfg_name, r in sorted(results.items(), key=lambda x: (x[1]['a'], x[1]['bits'])):
    print(f"{cfg_name:40s} {r['bits']:>5d} [{r['a']:.0f},{r['b']:.0f}] {r['delta']:>8.4f} {r['mean_diff_std']:>10.4f} {r['mean_acc']*100:>9.2f}%")

with open("./nd_multibit/fixed_params_comparison.json", 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to nd_multibit/fixed_params_comparison.json")
