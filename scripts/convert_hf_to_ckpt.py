"""
Convert HuggingFace diffusers SD 1.5 checkpoint to the single-file .ckpt format
used by the GRDH (Stable Diffusion) codebase.

Usage:
  python convert_hf_to_ckpt.py \\
      --model_id runwayml/stable-diffusion-v1-5 \\
      --output ./models/sd-v1-5.ckpt
"""
import argparse
import torch
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output", type=str, default="./models/sd-v1-5.ckpt")
    parser.add_argument("--half", action="store_true", help="Save in fp16")
    opt = parser.parse_args()

    print(f"Loading pipeline from {opt.model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        opt.model_id,
        torch_dtype=torch.float16 if opt.half else torch.float32,
    )

    # Build the state dict expected by LDM
    state_dict = {}

    # VAE
    for key, param in pipe.vae.state_dict().items():
        state_dict[f"first_stage_model.{key}"] = param

    # UNet
    for key, param in pipe.unet.state_dict().items():
        state_dict[f"model.diffusion_model.{key}"] = param

    # Text encoder (CLIP)
    for key, param in pipe.text_encoder.state_dict().items():
        state_dict[f"cond_stage_model.transformer.{key}"] = param

    # Add global_step
    state_dict["global_step"] = torch.tensor(0)

    # Save
    print(f"Saving checkpoint to {opt.output}...")
    torch.save({"state_dict": state_dict}, opt.output)
    print(f"Done! Saved {len(state_dict)} keys.")


if __name__ == '__main__':
    main()
