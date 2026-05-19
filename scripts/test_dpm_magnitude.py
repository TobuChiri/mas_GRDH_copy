"""Diagnose DPM inversion error vs encoding magnitude."""
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

for steps in [5, 10]:
    print(f'\n--- steps={steps} ---')
    for mag in [0.5, 0.674, 0.8, 1.0, 1.5, 2.0]:
        latent = torch.zeros(1, 4, 64, 64, device=device)
        latent[:, :, :32, :] = mag
        latent[:, :, 32:, :] = -mag
        try:
            z0, _ = sampler.sample(steps=steps, conditioning=c, batch_size=1,
                shape=(4, 64, 64), verbose=False, x_T=latent,
                width=512, height=512, DPMencode=False, DPMdecode=True)
            x0 = model.decode_first_stage(z0)
            x0_enc = model.get_first_stage_encoding(model.encode_first_stage(x0))
            z_enc, _ = sampler.sample(steps=steps, conditioning=c, batch_size=1,
                shape=(4, 64, 64), verbose=False, x_T=x0_enc,
                width=512, height=512, DPMencode=True)
            recon = z_enc.cpu().numpy()
            sign_acc = np.mean((latent.cpu().numpy() > 0) == (recon > 0))
            mae = np.abs(latent.cpu().numpy() - recon).mean()
            print(f'  mag={mag:.3f}: sign_acc={sign_acc*100:.2f}%, mae={mae:.4f}')
        except Exception as e:
            print(f'  mag={mag:.3f}: ERROR {e}')
