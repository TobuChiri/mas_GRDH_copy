"""Debug ND multi-bit: measure actual DDIM pipeline noise."""
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
_, unused = model.load_state_dict(pl_sd['state_dict'], strict=False)
model.eval()
model = model.to(device)

# Move conditioning model to device
model.cond_stage_model = model.cond_stage_model.to(device)

sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0)

# Conditioning
uc = model.get_learned_conditioning([""])
c = model.get_learned_conditioning(["a photo of a cat"])

for label, nd_kwargs in [
    ("1bit_[-3,3]_s0.3", dict(bits=1, sigma=0.3, a=-3.0, b=3.0)),
    ("3bit_[-6,6]_s0.05", dict(bits=3, sigma=0.05, a=-6.0, b=6.0)),
    ("3bit_[-9,9]_s0.05", dict(bits=3, sigma=0.05, a=-9.0, b=9.0)),
    ("N(0,1) baseline", None),
]:
    nd = mapping_module.nd_mapping(**nd_kwargs) if nd_kwargs else None
    latent_shape = (1, 4, 64, 64)
    np.random.seed(42)

    if nd:
        n_vals = nd.n_intervals
        secret = np.random.randint(0, n_vals, latent_shape).astype(np.float64)
        init_latent = nd.encode_secret(secret).astype(np.float32)
    else:
        init_latent = np.random.randn(*latent_shape).astype(np.float32)

    init_latent_t = torch.from_numpy(init_latent).to(device)
    shape = (4, 64, 64)

    with torch.no_grad():
        z0, _ = sampler.sample(S=50, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=5.0, eta=0.0,
            x_T=init_latent_t, verbose=False)
        x0 = model.decode_first_stage(z0)

        z_hat = model.get_first_stage_encoding(model.encode_first_stage(x0))
        z_enc, _ = sampler.sample(S=50, batch_size=1, shape=shape,
            conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=5.0, eta=0.0,
            x_T=z_hat, verbose=False, encode=True)

    recon = z_enc.cpu().numpy()
    diff = recon - init_latent
    print(f"\n{label}:")
    print(f"  init: mean={np.mean(init_latent):.4f}, std={np.std(init_latent):.4f}, range=[{np.min(init_latent):.4f}, {np.max(init_latent):.4f}]")
    print(f"  diff: mean={np.mean(diff):.4f}, std={np.std(diff):.4f}, MAE={np.mean(np.abs(diff)):.4f}")

    if nd:
        secret_recon = nd.decode_secret(pred_noise=recon)
        acc = np.mean(secret_recon == secret)
        print(f"  Decoding acc: {acc*100:.2f}%")

        # Simulate with measured noise to verify model
        noise_std = np.std(diff)
        noisy_sim = init_latent + noise_std * np.random.randn(*init_latent.shape).astype(np.float64)
        acc_sim = np.mean(nd.decode_secret(noisy_sim) == secret)
        print(f"  Expected at σ={noise_std:.4f}: {acc_sim*100:.2f}%")
