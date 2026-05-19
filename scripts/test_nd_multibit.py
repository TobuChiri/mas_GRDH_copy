"""Test ND multi-bit accuracy: offline screening + DDIM verification."""
import sys; sys.path.append('..')
import argparse, os, numpy as np, json
import mapping_module

# Lazy imports for DDIM phase — only loaded when needed (avoids heavy deps for offline sweep)
def _load_ddim():
    from omegaconf import OmegaConf
    import torch
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    return OmegaConf, torch, instantiate_from_config, DDIMSampler


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
parser.add_argument("--outdir", type=str, default="./nd_multibit")
parser.add_argument("--ddim_steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=5.0)
parser.add_argument("--n_prompts", type=int, default=5)
parser.add_argument("--sweep_only", action="store_true", help="Only run offline sweep, skip DDIM")
parser.add_argument("--ddim_only", action="store_true", help="Skip offline sweep, run DDIM only")
opt = parser.parse_args()

os.makedirs(opt.outdir, exist_ok=True)

latent_shape = (1, 4, 64, 64)

# ===== Phase 1: Offline Sweep (simulate DDIM noise) =====
DDIM_NOISE_STD = 0.6  # estimated from 1-bit ND identity results

offline_configs = []

# bits=3 (8 intervals)
for a, b in [(-3, 3), (-4, 4), (-5, 5), (-6, 6), (-9, 9)]:
    for sigma in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        offline_configs.append(dict(bits=3, sigma=sigma, a=float(a), b=float(b)))

# bits=6 (64 intervals)
for a, b in [(-4, 4), (-6, 6), (-9, 9), (-12, 12), (-18, 18)]:
    for sigma in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
        offline_configs.append(dict(bits=6, sigma=sigma, a=float(a), b=float(b)))

# bits=9 (512 intervals)
for a, b in [(-6, 6), (-9, 9), (-12, 12), (-18, 18), (-24, 24)]:
    for sigma in [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]:
        offline_configs.append(dict(bits=9, sigma=sigma, a=float(a), b=float(b)))

all_sweep_results = {}

if not opt.ddim_only:
    print("=" * 70)
    print("PHASE 1: OFFLINE SWEEP (DDIM noise simulated as N(0, {:.2f}))".format(DDIM_NOISE_STD))
    print("=" * 70)
    print(f"{'bits':>4} {'sigma':>5} {'a':>6} {'b':>6} {'acc_clean':>10} {'acc_noisy':>10}")
    print("-" * 55)

    for cfg in offline_configs:
        nd = mapping_module.nd_mapping(**cfg)
        n_intervals = 2 ** cfg['bits']

        np.random.seed(42)
        secret = np.random.randint(0, n_intervals, latent_shape).astype(np.float64)

        # Clean: encode → decode (no pipeline noise)
        encoded = nd.encode_secret(secret)
        decoded_clean = nd.decode_secret(encoded)
        acc_clean = float(np.mean(decoded_clean == secret))

        # Noisy: encode → add simulated DDIM noise → decode
        noisy = encoded + DDIM_NOISE_STD * np.random.randn(*encoded.shape).astype(np.float64)
        decoded_noisy = nd.decode_secret(noisy)
        acc_noisy = float(np.mean(decoded_noisy == secret))

        print(f"  {cfg['bits']:>3d}  {cfg['sigma']:>4.2f}  {cfg['a']:>5.0f}  {cfg['b']:>5.0f}  {acc_clean:>8.4f}  {acc_noisy:>8.4f}")

        key = f"ND({cfg['bits']}bit,σ={cfg['sigma']},[{cfg['a']:.0f},{cfg['b']:.0f}])"
        all_sweep_results[key] = dict(acc_clean=acc_clean, acc_noisy=acc_noisy)

    # Select promising configs for DDIM test
    # Filter: acc_noisy >= 0.80 and pick top 2-3 per bit depth
    promising = sorted(
        [(cfg['bits'], cfg['sigma'], cfg['a'], cfg['b'],
          all_sweep_results[f"ND({cfg['bits']}bit,σ={cfg['sigma']},[{cfg['a']:.0f},{cfg['b']:.0f}])"]['acc_clean'],
          all_sweep_results[f"ND({cfg['bits']}bit,σ={cfg['sigma']},[{cfg['a']:.0f},{cfg['b']:.0f}])"]['acc_noisy'])
         for cfg in offline_configs],
        key=lambda x: -x[5]
    )
    promising_above = [p for p in promising if p[5] >= 0.80]

    print(f"\nPromising configs (acc_noisy >= 80%): {len(promising_above)}")
    for bits, sigma, a, b, clean, noisy in promising_above:
        print(f"  ND({bits}bit, σ={sigma:.2f}, [{a:.0f},{b:.0f}]): clean={clean:.4f}, noisy={noisy:.4f}")

    # Save sweep results
    with open(f"{opt.outdir}/offline_sweep.json", 'w') as f:
        json.dump(all_sweep_results, f, indent=2)

    if opt.sweep_only:
        print("\nOffline sweep complete. Use --ddim_only to run DDIM verification.")
        exit(0)
else:
    # Load promising configs from previous sweep
    try:
        with open(f"{opt.outdir}/offline_sweep.json", 'r') as f:
            all_sweep_results = json.load(f)
    except FileNotFoundError:
        print("No offline_sweep.json found. Run without --ddim_only first.")
        exit(1)

# ===== Phase 2: DDIM Verification =====
# 3-bit configs (promising from offline sweep)
# 6-bit/9-bit: include extreme configs to empirically confirm the limit
ddim_configs = [
    dict(bits=3, sigma=0.05, a=-6.0, b=6.0, key="ND(3bit,σ=0.05,[-6,6])"),
    dict(bits=3, sigma=0.05, a=-9.0, b=9.0, key="ND(3bit,σ=0.05,[-9,9])"),
    dict(bits=3, sigma=0.10, a=-9.0, b=9.0, key="ND(3bit,σ=0.10,[-9,9])"),
    # 6-bit with extreme range to test boundary
    dict(bits=6, sigma=0.02, a=-18.0, b=18.0, key="ND(6bit,σ=0.02,[-18,18])"),
    # 9-bit with extreme range
    dict(bits=9, sigma=0.01, a=-24.0, b=24.0, key="ND(9bit,σ=0.01,[-24,24])"),
]

print("\n" + "=" * 70)
print("PHASE 2: DDIM VERIFICATION")
print("=" * 70)

# Load model (lazy imports)
OmegaConf, torch, instantiate_from_config, DDIMSampler = _load_ddim()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = OmegaConf.load(opt.config)

def load_model(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location=device, weights_only=False)
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd['state_dict'], strict=False)
    model.eval()
    return model

model = load_model(config, opt.ckpt, device).to(device)
sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=0.0)

with open(opt.test_prompts, 'r') as f:
    prompts = [l.strip() for l in f if l.strip()][:opt.n_prompts]

attacks = [
    ("identity",   None,  None),
]

all_ddim_results = {}

for dc in ddim_configs:
    cfg_name = dc.get('key', f"ND({dc['bits']}bit,σ={dc['sigma']},[{dc['a']:.0f},{dc['b']:.0f}])")
    print(f"\n{cfg_name}")

    nd = mapping_module.nd_mapping(bits=dc['bits'], sigma=dc['sigma'], a=dc['a'], b=dc['b'])
    n_intervals = 2 ** dc['bits']
    acc_list = []

    for idx, prompt in enumerate(prompts):
        uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
        c = model.get_learned_conditioning([prompt])
        np.random.seed(idx)

        random_input = np.random.randint(0, n_intervals, latent_shape).astype(np.float64)
        init_latent = nd.encode_secret(secret_message=random_input).astype(np.float32)
        init_latent = torch.from_numpy(init_latent).to(device)

        shape = init_latent.shape[1:]
        z0, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=opt.scale, eta=0.0, x_T=init_latent, verbose=False)
        x0 = model.decode_first_stage(z0)

        # Encode back via VAE
        z_hat = model.get_first_stage_encoding(model.encode_first_stage(x0.to(device)))
        z_enc, _ = sampler.sample(S=opt.ddim_steps, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=opt.scale, eta=0.0, x_T=z_hat, verbose=False, encode=True)

        recon = nd.decode_secret(pred_noise=z_enc.clone().cpu().numpy())
        acc = float(np.mean(recon == random_input))
        acc_list.append(acc)
        print(f"  prompt {idx:03d}: {acc*100:.2f}%")

    mean_acc = np.mean(acc_list)
    print(f"  >>> MEAN: {mean_acc*100:.2f}%")
    all_ddim_results[cfg_name] = dict(
        bits=dc['bits'], sigma=dc['sigma'], a=dc['a'], b=dc['b'],
        mean_acc=mean_acc, per_prompt=acc_list
    )

# Summary
print("\n" + "=" * 70)
print("MULTI-BIT ND DDIM RESULTS")
print("=" * 70)
for cfg_name, res in sorted(all_ddim_results.items()):
    print(f"  {cfg_name:45s}: {res['mean_acc']*100:.2f}%")

output = dict(offline_sweep=all_sweep_results, ddim_results=all_ddim_results)
with open(f"{opt.outdir}/multibit_results.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {opt.outdir}/")
