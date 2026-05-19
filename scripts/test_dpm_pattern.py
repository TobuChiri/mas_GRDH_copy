"""Diagnose DPM inversion: block pattern vs random pattern."""
import sys; sys.path.append('..')
import numpy as np, torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

device = 'cuda:0'
config = OmegaConf.load('../configs/stable-diffusion/ldm.yaml')
pl_sd = torch.load('../models/v1-5-pruned-emaonly.ckpt', map_location=device, weights_only=False)
model = instantiate_from_config(config.model)
model.load_state_dict(pl_sd['state_dict'], strict=False)
model.eval().to(device)
sampler = DPMSolverSampler(model)
print('Model loaded')
c = model.get_learned_conditioning(['a test image'])

mag = 0.674  # ND(1bit) encoding value
s = 12345
for desc, make_pattern in [
    ('block (top=+mag, bottom=-mag)', lambda: np.concatenate([
        np.full((1, 4, 32, 64), mag),
        np.full((1, 4, 32, 64), -mag)], axis=2)),
    ('random sign (per element)', lambda: (np.random.RandomState(s).rand(1, 4, 64, 64) > 0.5).astype(float) * 2 * mag - mag),
    ('RS-like (truncated normal, random sign)', lambda: (lambda r=np.random.RandomState(s): np.where(r.rand(1, 4, 64, 64) > 0.5,
        np.abs(r.randn(1, 4, 64, 64)) * 1.363, -np.abs(r.randn(1, 4, 64, 64)) * 1.363) )()),
]:
    # Ensure same magnitude distribution for fair comparison
    if 'RS-like' in desc:
        pass
    else:
        np.random.seed(s)
    latent_np = make_pattern()
    latent_t = torch.from_numpy(latent_np.astype(np.float32)).to(device)

    z0, _ = sampler.sample(steps=10, conditioning=c, batch_size=1,
        shape=(4, 64, 64), verbose=False, x_T=latent_t,
        width=512, height=512, DPMencode=False, DPMdecode=True)
    x0 = model.decode_first_stage(z0)
    z_enc, _ = sampler.sample(steps=10, conditioning=c, batch_size=1,
        shape=(4, 64, 64), verbose=False,
        x_T=model.get_first_stage_encoding(model.encode_first_stage(x0)),
        width=512, height=512, DPMencode=True)

    recon = z_enc.cpu().numpy()
    sign_acc = np.mean((latent_np > 0) == (recon > 0))
    mae = np.abs(latent_np - recon).mean()
    print(f'{desc}: sign_acc={sign_acc*100:.2f}%, mae={mae:.4f}')
