import sys
sys.path.append('..')
import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from tqdm import tqdm
from scripts.utils import gray_code
import robust_eval
import mapping_module


# 计算解码出的秘密信息与原始秘密之间的逐比特准确率
# 使用格雷码将数值转换为二进制，然后按位异或统计错误比特数
def cal_acc(input, gt, gray_list, bits):
    trans_fn = np.frompyfunc(lambda x: int(gray_list[int(x)], 2), 1, 1)
    count_fn = np.frompyfunc(lambda x: bin(int(x)).count('1'), 1, 1)
    a1 = trans_fn(input).astype(np.int32)
    a2 = trans_fn(gt).astype(np.int32)
    result = a1 ^ a2
    result = count_fn(result).flatten()
    shape = len(result)
    count = sum(result)
    acc = 1 - count/(int(shape)*bits)
    return acc


# 从配置文件构建模型并加载 checkpoint 权重
def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


# 将文本提示词编码为 CLIP 条件嵌入向量
# 返回 (条件嵌入, 无条件嵌入)，scale!=1.0 时使用无分类器引导
def load_model_and_get_prompt_embedding(model, opt, prompts):
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)

    return c, uc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of a doggy, ultra realistic",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--dpm_steps",
        type=int,
        default=20,
        help="number of sampling steps",
    )

    parser.add_argument(
        "--dpm_gen_steps",
        type=int,
        default=20,
        help="number of generation steps",
    )

    parser.add_argument(
        "--dpm_inv_steps",
        type=int,
        default=20,
        help="number of inversion steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=58756,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--dpm_order",
        type=int,
        help="",
        choices=[1, 2, 3],
        default=2
    )

    parser.add_argument(
        "--tau_a",
        type=float,
        help="low-frequency threshold for FALE mapping",
        default=0.4
    )

    parser.add_argument(
        "--tau_b",
        type=float,
        help="high-frequency threshold for FALE mapping",
        default=0.8
    )

    parser.add_argument(
        "--gpu",
        type=str,
        help="",
        default='cuda:0'
    )

    parser.add_argument(
        "--test_prompts",
        type=str,
        help="./test_prompts.txt",
        default='chatgpt'
    )
    parser.add_argument(
        "--attack_layer",
        type=str,
        help="which attack",
        default='gblur'
    )
    parser.add_argument(
        "--attack_factor",
        type=float,
        help="different opt",
        default=3
    )
    parser.add_argument(
        "--bit_num",
        type=int,
        help="bit nums per pixel",
        default=1
    )
    parser.add_argument(
        "--mapping_func",
        type=str,
        help="mapping_module",
        default='ours_mapping'
    )

    parser.add_argument(
        "--block_size",
        type=int,
        help="block size for BICS interleaving (default 8)",
        default=8
    )

    parser.add_argument(
        "--mapping_bits",
        type=int,
        help="bits per latent element for mapping (default uses --bit_num)",
        default=None
    )

    opt = parser.parse_args()
    # seed_everything(opt.seed) # this line is for reproducible sampling or comparison
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    # 根据攻击类型和因子选择对应的鲁棒性测试函数
    print(opt.dpm_steps)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    attack_type = opt.attack_layer
    attack_factor = opt.attack_factor
    mapping_type = opt.mapping_func
    print(f'we are testing {attack_type} under setting {attack_factor}')
    attack_func = getattr(robust_eval, attack_type)
    base_count = len(os.listdir(outpath))
    attack_suffix = str(attack_factor).replace('.', '')
    # tmp image for each generation
    tmp_image_name = outpath + '/tmp_%03d' % (base_count) + f'_{attack_type}_{attack_suffix}'
    bits = opt.bit_num
    gray_list = gray_code(bits)
    mapping_func = getattr(mapping_module, mapping_type)(bits=bits)
    # Configure mapping-specific parameters
    if mapping_type == 'bics_mapping':
        mapping_func = mapping_module.bics_mapping(bits=bits, block_size=opt.block_size)
    elif mapping_type == 'fale_mapping':
        mapping_func = mapping_module.fale_mapping(bits=bits, tau_a=opt.tau_a, tau_b=opt.tau_b)
    elif mapping_type == 'combined_mapping':
        mapping_func = mapping_module.combined_mapping(bits=bits, block_size=opt.block_size,
                                                       tau_a=opt.tau_a, tau_b=opt.tau_b)
    elif mapping_type == 'icdf_mapping':
        mapping_func = mapping_module.icdf_mapping(bits=bits, gap_start=0.4, gap_end=0.6)
    elif mapping_type == 'ds_mapping':
        mapping_func = mapping_module.ds_mapping(bits=bits, threshold=0.7)
    elif mapping_type == 'hadamard_mapping':
        mapping_func = mapping_module.hadamard_mapping(bits=bits)
    elif mapping_type == 'multi_icdf_mapping':
        mapping_func = mapping_module.multi_icdf_mapping(bits=min(bits, 2), gap=0.2)
    elif mapping_type == 'rs_mapping':
        mapping_func = mapping_module.rs_mapping(bits=bits, alpha=1.0)
    elif mapping_type == 'mlq_rs_mapping':
        mapping_func = mapping_module.mlq_rs_mapping(bits=min(bits, 2), alpha=0.8, beta=1.5)
    elif mapping_type == 'rs_ecc_mapping':
        mapping_func = mapping_module.rs_ecc_mapping(repeats=3, rs_alpha=0.8)
    elif mapping_type == 'nd_mapping':
        mapping_func = mapping_module.nd_mapping(bits=bits)
    else:
        mapping_func = getattr(mapping_module, mapping_type)(bits=bits)

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu)
    model = model.to(device)
    sampler = DPMSolverSampler(model)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    # prompt_path = f'/groupshare_1/text_prompt_dataset/{opt.test_prompts}_dataset.txt'
    prompt_path = opt.test_prompts
    with open(prompt_path, 'r') as f:
        prompts = f.readlines()
        prompts = [i.strip() for i in prompts]

    # image composition
    batch_size = opt.n_samples
    width = 512
    height = 512
    acc_records = []

    with torch.no_grad():
        with precision_scope("cuda"):
            for j in tqdm(range(len(prompts))):
                cur_prompt = prompts[j]
                print(cur_prompt)
                c, uc = load_model_and_get_prompt_embedding(model, opt, [cur_prompt])

                latent_shape = (batch_size, opt.C, int(height // opt.f), int(width // opt.f))

                # 第一步：随机生成秘密信息并通过映射函数编码为初始潜变量
                random_input = np.random.randint(0, 2 ** bits, latent_shape)  # 随机生成秘密信息
                random_input_ori_sample = None
                # 部分映射方法需要额外的均匀/高斯采样作为辅助输入
                if mapping_func.need_uniform_sampler:  # 连续均匀分布 U(0,1)
                    random_input_ori_sample = np.random.rand(*latent_shape)
                if mapping_func.need_gaussian_sampler:  # 标准正态分布 N(0,1)
                    random_input_ori_sample = np.random.randn(*latent_shape)
                if mapping_type in ['ours_mapping', 'bics_mapping', 'fale_mapping', 'combined_mapping',
                                     'qr_ds_mapping', 'ds_mapping', 'hadamard_mapping']:
                    seed_shuffle = np.random.randint(0, 2 ** 31 - 1, 1)
                    seed_kernel = np.random.randint(0, 2 ** 31 - 1, 1)
                    random_input_args = dict(seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)
                elif mapping_type == 'tdsc_mapping':
                    random_input_args = dict(key=np.random.randint(0, 2 ** 31 - 1, 1))
                else:
                    random_input_args = dict()
                init_latent = mapping_func.encode_secret(secret_message=random_input, ori_sample=random_input_ori_sample, **random_input_args).astype(np.float32)
                init_latent = torch.from_numpy(init_latent).to(device)

                # 第二步：以携带秘密信息的潜变量为起点，通过 DPM-Solver++ 反向扩散生成图像
                shape = init_latent.shape[1:]
                z_0, _ = sampler.sample(steps=opt.dpm_gen_steps,
                                        unconditional_conditioning=uc,
                                        conditioning=c,
                                        batch_size=opt.n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        eta=opt.ddim_eta,
                                        order=opt.dpm_order,
                                        x_T=init_latent,
                                        width=width,
                                        height=height,
                                        DPMencode=False,
                                        DPMdecode=True,
                                        )

                # 将潜变量解码为像素空间图像
                x0_samples = model.decode_first_stage(z_0)

                # 第三步：对生成的图像施加鲁棒性攻击（如JPEG压缩、模糊、噪声、缩放等）
                x0_samples = attack_func(x0_samples, factor=attack_factor, tmp_image_name=tmp_image_name).to(device)

                # 第四步：将攻击后的图像重新编码回潜空间，再通过 DPM 正向扩散反转回初始噪声
                init_latent_hat = model.get_first_stage_encoding(model.encode_first_stage(x0_samples))
                z_enc, _ = sampler.sample(steps=opt.dpm_inv_steps,
                                          unconditional_conditioning=uc,
                                          conditioning=c,
                                          batch_size=opt.n_samples,
                                          shape=shape,
                                          verbose=False,
                                          unconditional_guidance_scale=opt.scale,
                                          eta=opt.ddim_eta,
                                          order=opt.dpm_order,
                                          x_T=init_latent_hat,
                                          width=width,
                                          height=height,
                                          DPMencode=True,
                                          )
                distance = (init_latent - init_latent_hat).abs().mean()
                print("encoder-decode error:", distance)
                recon_distance = (init_latent - z_enc).abs().mean()
                print("recon-error", recon_distance)
                # 从反转后的噪声中解码出秘密信息，并与原始秘密计算准确率
                pred_noise = z_enc.clone().cpu().numpy()
                recon_latent = mapping_func.decode_secret(pred_noise=pred_noise, **random_input_args)
                if mapping_type == 'tdsc_mapping':
                    acc = mapping_func._compute_acc(random_input, recon_latent)
                else:
                    acc = cal_acc(recon_latent, random_input, gray_list=gray_list, bits=bits)

                print(f'mean:{acc}')
                acc_records.append(acc)

    print(f'average accuracy: {sum(acc_records) / len(acc_records)}')



if __name__ == "__main__":
    main()



