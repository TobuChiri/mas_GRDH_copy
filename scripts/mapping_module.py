import torch
import numpy as np
from scipy.linalg import qr
from scipy.stats import norm
from scipy.fft import dct, idct
# import Levenshtein
# from S2I_Transformation_scheme2 import embed_message, extract_message


# 抽象基类：定义秘密信息与潜变量噪声之间的映射接口
# 所有具体的映射方法（simple / tmm / ours）都继承自此类
class mapping_module:
    def __init__(self, need_uniform_sampler=False, need_gaussian_sampler=False, bits=1, seed=None):
        self.need_uniform_sampler = need_uniform_sampler
        self.need_gaussian_sampler = need_gaussian_sampler
        self.bits = bits
        self.bits_l = 2 ** bits
        self.seed = seed
        pass

    # 将秘密信息映射为噪声，修改式的方法往往需要借助额外的采样步骤
    def encode_secret(self, secret_message, ori_sample=None):
        pass

    # 从噪声中还原秘密信息
    def decode_secret(self, pred_noise):
        pass


# 最简单的映射方法：用潜变量的正负号来表示1 bit秘密信息
# bits=1 时与 TMM 映射等价
# 编码：secret=0 → 元素为负，secret=1 → 元素为正
# 解码：直接读取每个元素的符号位
class simple_mapping(mapping_module):
    # 简单的映射方法，将秘密信息映射为正负号
    def __init__(self, bits=1):
        assert bits == 1
        super(simple_mapping, self).__init__(need_gaussian_sampler=True)

    def encode_secret(self, secret_message, ori_sample=None):
        secret_re = secret_message * 2 - 1  # 嵌入信息：0 → -1, 1 → +1
        ori_sign = np.sign(ori_sample)  # 原始符号 -1或+1
        out = ori_sample * ori_sign * secret_re
        return out

    def decode_secret(self, pred_noise):
        out = np.round((np.sign(pred_noise) + 1) / 2)
        return out


# 基于正态分布分位数的映射方法（TMM 2023）
# 编码：将均匀采样值映射到秘密信息所在区间的正态分位数
# 解码：计算标准正态 CDF 后根据区间位置还原秘密
# 论文：https://ieeexplore.ieee.org/abstract/document/10306313
class tmm_mapping(mapping_module):
    def __init__(self, bits=1):
        super(tmm_mapping, self).__init__(need_uniform_sampler=True, bits=bits)

    def encode_secret(self, secret_message, ori_sample=None):
        out = norm.ppf((ori_sample + secret_message) / self.bits_l)
        # 以 bits=1 为例：
        # secret = 0  →  (ori_sample + 0) / 2  →  值落在 [0.0, 0.5)   →  norm.ppf → 负数区
        # secret = 1  →  (ori_sample + 1) / 2  →  值落在 [0.5, 1.0)   →  norm.ppf → 正数区
        return out

    def decode_secret(self, pred_noise):
        out = np.floor(self.bits_l * norm.cdf(pred_noise))
        return out

# From paper (TDSC 2022): https://ieeexplore.ieee.org/abstract/document/9931463
# class tdsc_mapping(mapping_module):
#     def __init__(self, bits, group_num=6, fixed=4096):
#         super(tdsc_mapping, self).__init__(need_gaussian_sampler=True, bits=bits)
#         self.group_num = group_num
#         self.fixed = fixed
#         self.cap = None
#
#     def encode_secret(self, secret_message, ori_sample=None, key=-1):
#         latent_shape = secret_message.shape
#         secret = ''.join([bin(int(num)).replace('0b', '').zfill(self.bits) for num in secret_message.flatten()])
#         z, c = embed_message(
#             ori_sample.flatten(), secret, self.group_num,
#             key=key, fixed_pos_list=self.fixed
#         )
#         self.cap = c
#         return z.reshape(latent_shape)
#
#     def decode_secret(self, pred_noise, key=-1):
#         out = extract_message(pred_noise.flatten(), group_num=self.group_num, key=key, fixed_pos_list=self.fixed)
#         return out
#
#     def _compute_acc(self, secret_message, received_message, length=-1):
#         secret_message = ''.join([bin(int(num)).replace('0b', '').zfill(self.bits) for num in secret_message.flatten()])
#         if length <= 0:
#             length = max(len(secret_message), len(received_message))
#         if len(secret_message) < length:
#             secret_message = secret_message + '0' * (length - len(secret_message))
#         else:
#             secret_message = secret_message[:length]
#         #     if len(str2) < length:
#         #         str2 = str2 + '0' * (length - len(str2))
#         #     else:
#         #         str2 = str2[:length]
#         return 1 - Levenshtein.distance(secret_message, received_message) / max(length, len(received_message))


# 本文提出的映射方法：通过正交随机变换 + 随机打乱实现秘密嵌入
# 不需要额外的随机采样器，直接对秘密信息做可逆变换使其分布接近高斯噪声
class ours_mapping(mapping_module):
    def __init__(self, bits=1):
        super(ours_mapping, self).__init__(bits=bits)
        # 计算秘密信息的均值和标准差，用于归一化/反归一化
        # 这是离散均匀分布的均值和方差公式
        # 编码时需要先对秘密信息做归一化，使其均值为 0、方差为 1（接近标准正态分布）：
        # secret_re = (secret_message - self.bits_mean) / self.bits_std
        self.bits_mean = (self.bits_l - 1) / 2
        self.bits_std = ((self.bits_l ** 2 - 1) / 12) ** 0.5

    # 用 QR 分解生成随机正交矩阵作为卷积核
    # 正交矩阵保证了变换前后能量守恒，避免引入可检测的统计特征
    def _get_random_kernel(self, seed_kernel, kernel_shape):
        ori_seed = np.random.get_state()[1][0]  # 获取原来的随机种子
        np.random.seed(seed_kernel)
        H = np.random.randn(*kernel_shape)
        Q, r = qr(H)  # Gram-Schmidt正交化过程
        kernel = Q  #
        np.random.seed(ori_seed)  # 恢复原来的种子
        return kernel

    # 基于种子的可逆随机打乱/恢复操作
    # 先保存当前随机状态，用种子控制打乱顺序以保证可复现，操作后恢复原始状态
    def _random_shuffle(self, ori_input, seed_shuffle, reverse=False):
        ori_seed = np.random.get_state()[1][0]  # 获取原来的随机种子
        np.random.seed(seed_shuffle)

        ori_shape = ori_input.shape
        ori_input = ori_input.flatten()
        ori_order = np.arange(0, len(ori_input))
        shuffle_order = ori_order.copy()
        np.random.shuffle(shuffle_order)  # 索引打乱
        if reverse:
            sorted_shuffle_order = np.argsort(shuffle_order)
            reverse_order = ori_order[sorted_shuffle_order]
            out = ori_input[reverse_order]
        else:
            out = ori_input[shuffle_order]
        out = out.reshape(*ori_shape)
        np.random.seed(ori_seed)  # 恢复原来的种子
        return out

    # 编码：将秘密信息通过正交变换 + 随机打乱映射为类高斯噪声
    # 不需要额外的随机采样器，直接对秘密做可逆变换
    # 输入：secret_message 形状为 (1, 4, 64, 64)，每个元素取值 {0, 1, ..., bits_l-1}
    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        # 获取一个 (64, 64) 的随机正交矩阵 K（K^T K = I）。用了 QR 分解保证正交性。这个 K 就是后面做矩阵乘法的"卷积核"
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=secret_message.shape[-2:])  # 获取随机kernel
        # 对秘密信息做归一化，使其均值为 0、方差为 1（接近标准正态分布）：
        secret_re = (secret_message - self.bits_mean) / self.bits_std
        out = np.matmul(np.matmul(kernel, secret_re), kernel.transpose(-1, -2))
        out = self._random_shuffle(out, seed_shuffle=seed_shuffle)  # 随机打乱
        return out

    # 解码：逆向操作 — 取消打乱 → 逆正交变换 → 取整还原秘密
    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)  # 取消随机打乱
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=pred_noise.shape[-2:])  # 获取随机kernel
        secret_hat = np.matmul(np.matmul(kernel.transpose(-1, -2), pred_noise), kernel)
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        out = np.round(secret_hat) % self.bits_l
        return out


# ============================================================
# BICS: Bit-Interleaved Coding with Spatial Error Spreading
# ============================================================
# Adds block-wise spatial interleaving to any base mapping.
# Converts burst errors (from JPEG/blur) into random isolated
# bit errors, significantly improving effective BER.


def _block_interleave(secret, block_size=8, reverse=False):
    """
    Block-wise spatial interleaver for (1, C, H, W) latent tensor.

    Divides the spatial grid (H, W) into (H//B × W//B) blocks of size B×B,
    then permutes blocks and within-block positions using a fixed deterministic
    pattern derived from the tensor shape (no seed needed — reproducibility
    is determined solely by block_size).

    Args:
        secret: np.ndarray of shape (1, C, H, W), typically (1, 4, 64, 64)
        block_size: int, side length of each square block (default 8)
        reverse: bool, if True performs de-interleaving

    Returns:
        Interleaved/de-interleaved array with same shape
    """
    _, C, H, W = secret.shape
    B = block_size
    assert H % B == 0 and W % B == 0, f"Spatial dims ({H},{W}) must be divisible by block_size ({B})"

    n_blocks_h = H // B
    n_blocks_w = W // B
    n_blocks = n_blocks_h * n_blocks_w

    # Flatten to (C, n_blocks, B*B)
    x = secret.reshape(C, H, W)
    # Split into blocks: (C, n_blocks_h, n_blocks_w, B, B)
    blocks = x.reshape(C, n_blocks_h, B, n_blocks_w, B)
    # Rearrange to (C, n_blocks, B*B)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(C, n_blocks, B * B)

    # Generate deterministic permutation based on shape+sizes
    rng = np.random.RandomState(seed=(hash((H, W, B, n_blocks)) % (2**31)))
    block_order = np.arange(n_blocks)
    rng.shuffle(block_order)
    # Per-block within-block permutations (same pattern for all blocks)
    inner_perm = np.arange(B * B)
    rng.shuffle(inner_perm)
    inner_inv = np.argsort(inner_perm)

    if not reverse:
        # Shuffle block order
        blocks = blocks[:, block_order, :]
        # Shuffle within each block
        blocks = blocks[:, :, inner_perm]
    else:
        # Inverse within-block shuffle
        blocks = blocks[:, :, inner_inv]
        # Inverse block order
        inv_order = np.argsort(block_order)
        blocks = blocks[:, inv_order, :]

    # Reshape back to original
    blocks = blocks.reshape(C, n_blocks_h, n_blocks_w, B, B)
    x_out = blocks.transpose(0, 1, 3, 2, 4).reshape(1, C, H, W)
    return x_out


class bics_mapping(ours_mapping):
    """
    Bit-Interleaved Coding with Spatial Error Spreading.

    Wraps ours_mapping with a block-wise spatial interleaver that
    spreads burst errors across the latent grid before QR-based mapping.
    JPEG/blur attacks tend to corrupt spatially contiguous regions;
    interleaving converts these burst errors into random isolated errors
    which the QR orthogonal mapping handles more effectively.

    Usage: pass --mapping_func bics_mapping --block_size 8
    """
    def __init__(self, bits=1, block_size=8):
        super().__init__(bits=bits)
        self.block_size = block_size
        self.need_uniform_sampler = False
        self.need_gaussian_sampler = False

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        # Interleave secret spatially before QR mapping
        secret_interleaved = _block_interleave(secret_message, self.block_size, reverse=False)
        return super().encode_secret(secret_interleaved, ori_sample=None,
                                     seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        # Decode normally, then de-interleave
        secret_hat = super().decode_secret(pred_noise,
                                           seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)
        return _block_interleave(secret_hat, self.block_size, reverse=True)


# ============================================================
# FALE: Frequency-Adaptive Latent Embedding
# ============================================================
# Analyzes per-frequency robustness of SD's VAE latent space
# and adaptively allocates bits/energy to robust coefficients.


def _latent_dct(latent):
    """Apply 2D DCT to each channel of the latent. Input: (1, C, H, W)."""
    _, C, H, W = latent.shape
    out = np.zeros_like(latent, dtype=np.float64)
    for c in range(C):
        out[0, c] = dct(dct(latent[0, c], axis=0, norm='ortho'), axis=1, norm='ortho')
    return out


def _latent_idct(latent_dct):
    """Apply inverse 2D DCT. Input: (1, C, H, W)."""
    _, C, H, W = latent_dct.shape
    out = np.zeros_like(latent_dct, dtype=np.float64)
    for c in range(C):
        out[0, c] = idct(idct(latent_dct[0, c], axis=0, norm='ortho'), axis=1, norm='ortho')
    return out


class fale_mapping(ours_mapping):
    """
    Frequency-Adaptive Latent Embedding.

    Uses a pre-computed frequency robustness profile to weight the
    orthogonal mapping. More robust frequency coefficients receive
    higher embedding energy, while fragile coefficients are attenuated.
    The robustness profile is loaded from a .npy file generated by
    `analyze_frequency_robustness.py`.

    Usage: pass --mapping_func fale_mapping --tau_a 0.3 --tau_b 0.7
    """
    def __init__(self, bits=1, tau_a=0.3, tau_b=0.7, profile_path=None):
        super().__init__(bits=bits)
        self.tau_a = tau_a  # Low-frequency threshold
        self.tau_b = tau_b  # High-frequency threshold
        self.profile = None
        if profile_path is not None:
            try:
                self.profile = np.load(profile_path)
            except Exception:
                pass
        self.need_uniform_sampler = False
        self.need_gaussian_sampler = False

    def _get_freq_mask(self, shape):
        """Build a frequency mask with shape (H, W), values in [tau_a, 1.0]."""
        _, _, H, W = shape
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        # Low freq → high weight, high freq → low weight
        freq_ratio = np.sqrt(yy**2 + xx**2) / np.sqrt(2.0)
        mask = self.tau_a + (1.0 - self.tau_a) * np.clip(1.0 - freq_ratio / self.tau_b, 0, 1)
        return mask  # (H, W)

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        _, C, H, W = secret_message.shape
        # Normalize secret
        secret_re = (secret_message - self.bits_mean) / self.bits_std
        # Transform to frequency domain
        secret_dct = np.zeros_like(secret_re, dtype=np.float64)
        for c in range(C):
            secret_dct[0, c] = dct(dct(secret_re[0, c], axis=0, norm='ortho'), axis=1, norm='ortho')
        # Apply frequency-adaptive weighting
        freq_mask = self._get_freq_mask(secret_message.shape)  # (H, W)
        secret_dct_weighted = secret_dct * freq_mask  # broadcast over channels
        # Transform back to spatial
        secret_weighted = np.zeros_like(secret_re, dtype=np.float64)
        for c in range(C):
            secret_weighted[0, c] = idct(idct(secret_dct_weighted[0, c], axis=0, norm='ortho'), axis=1, norm='ortho')
        # Now apply QR mapping on the frequency-shaped secret
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=(H, W))
        out = np.matmul(np.matmul(kernel, secret_weighted), kernel.transpose(-1, -2))
        out = self._random_shuffle(out, seed_shuffle=seed_shuffle)
        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        H, W = pred_noise.shape[-2], pred_noise.shape[-1]
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=(H, W))
        secret_hat = np.matmul(np.matmul(kernel.transpose(-1, -2), pred_noise), kernel)
        # Inverse frequency weighting
        freq_mask = self._get_freq_mask(pred_noise.shape)
        # Avoid division by zero
        inv_mask = np.where(freq_mask > 0.01, 1.0 / freq_mask, 0.0)
        secret_hat_dct = np.zeros_like(secret_hat, dtype=np.float64)
        for c in range(secret_hat.shape[1]):
            secret_hat_dct[0, c] = dct(dct(secret_hat[0, c], axis=0, norm='ortho'), axis=1, norm='ortho')
        secret_hat_dct_unweighted = secret_hat_dct * inv_mask
        for c in range(secret_hat.shape[1]):
            secret_hat[0, c] = idct(idct(secret_hat_dct_unweighted[0, c], axis=0, norm='ortho'), axis=1, norm='ortho')
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        out = np.round(secret_hat) % self.bits_l
        return out


# ============================================================
# Combined: BICS + FALE
# ============================================================
class combined_mapping(fale_mapping):
    """
    Combines BICS spatial interleaving with FALE frequency-adaptive weighting.

    Pipeline: secret → block interleave → frequency-adaptive weight → QR mapping
    → shuffle → (inverse) → de-shuffle → inverse QR → inverse freq weight → de-interleave

    This provides two complementary robustness improvements:
    1. FALE: protects low frequencies that survive JPEG compression
    2. BICS: spreads burst errors across spatial positions

    Usage: pass --mapping_func combined_mapping --block_size 8 --tau_a 0.3 --tau_b 0.7
    """
    def __init__(self, bits=1, block_size=8, tau_a=0.3, tau_b=0.7):
        super().__init__(bits=bits, tau_a=tau_a, tau_b=tau_b)
        self.block_size = block_size

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        # Step 1: Spatial interleaving (BICS)
        secret_interleaved = _block_interleave(secret_message, self.block_size, reverse=False)
        # Step 2: Frequency-adaptive weighting (FALE), then QR mapping
        return fale_mapping.encode_secret(self, secret_interleaved, ori_sample=None,
                                          seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        # Step 1: Inverse FALE (decode with freq unweighting)
        secret_hat = fale_mapping.decode_secret(self, pred_noise,
                                                seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)
        # Step 2: De-interleave
        return _block_interleave(secret_hat, self.block_size, reverse=True)


# ============================================================
# DS: Diffusion-Stego Message Projection (MC variant)
# ============================================================
# Direct message projection into latent noise, no QR decomposition.
# Based on: Kim et al., "Diffusion-Stego: Training-free Diffusion
# Generative Steganography via Message Projection", Inf. Sci. 2025
#
# MC (Centered Binary) projection:
#   bit=0 → value = 0  (mode of N(0,1), most "natural" value)
#   bit=1 → value = ±√2 (random sign, maintains variance=1)
#
# Decoding: threshold on |value| to distinguish 0 from ±√2.
# This is a fundamentally different paradigm from GRDH:
# - GRDH: transform secret → look like noise
# - DS: start with noise → tweak to encode secret


class ds_mapping(mapping_module):
    """
    Diffusion-Stego MC Message Projection.

    Instead of transforming the secret to look Gaussian (GRDH approach),
    starts with actual Gaussian noise and modifies each element to
    encode the secret bit while preserving mean=0, variance=1.

    Capacity: 1 bit per latent element = 16,384 bits per 512x512 image.
    """
    def __init__(self, bits=1, threshold=0.7):
        assert bits == 1, "ds_mapping only supports bits=1"
        super().__init__(need_gaussian_sampler=True, bits=bits)
        self.threshold = threshold  # Decision threshold for decoding
        self.sqrt2 = np.sqrt(2.0)

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        Encode secret into Gaussian noise using MC projection.

        Args:
            secret_message: (1,4,64,64) with values in {0, 1}
            ori_sample: Gaussian noise N(0, I) with same shape
            seed_kernel: used as seed for random ± signs on bit=1
            seed_shuffle: unused (for interface compatibility)
        Returns:
            Modified noise with same shape, mean≈0, var≈1
        """
        z = ori_sample.copy()
        # MC projection: bit=0→0, bit=1→±√2
        # Use seed_kernel for reproducible random signs
        if seed_kernel is not None:
            rng = np.random.RandomState(int(seed_kernel[0]) % (2**31))
        else:
            rng = np.random

        out = np.where(secret_message == 0, 0.0, self.sqrt2 * rng.choice([-1, 1], size=secret_message.shape))
        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        Decode by thresholding on absolute value.

        |value| < threshold → bit=0
        |value| >= threshold → bit=1

        Returns: decoded secret with same shape, values in {0, 1}
        """
        out = (np.abs(pred_noise) >= self.threshold).astype(np.float64)
        return out


# ============================================================
# ICDF: Non-Continuous Inverse CDF Mapping
# ============================================================
# Based on: Zhang et al., "Enhancing the communication reliability
# for generative image steganography with diffusion model",
# Information Processing and Management, 2025
#
# Uses the inverse normal CDF (Φ⁻¹) to create non-contiguous
# sub-intervals with gaps. The gaps serve as error detection margin:
# if decoded value falls in a gap, an error is detected.
#
# For 1-bit:
#   bit=0 → Φ⁻¹([0, p0))  where p0 is gap_start (default: 0.4)
#   bit=1 → Φ⁻¹((p1, 1])  where p1 is gap_end   (default: 0.6)
#   Gap: [Φ⁻¹(p0), Φ⁻¹(p1)] = error detection zone
#
# The gap width controls the robustness-vs-capacity tradeoff:
#   wider gap → more error detection but fewer valid encoding values


class icdf_mapping(mapping_module):
    """
    ICDF-based non-continuous interval mapping.

    Encodes secret bits into inverse-CDF sub-intervals with
    protective gaps for error detection.

    Args:
        bits: must be 1
        gap_start: CDF value where bit=0 interval ends (default 0.4)
        gap_end: CDF value where bit=1 interval begins (default 0.6)
    """
    def __init__(self, bits=1, gap_start=0.4, gap_end=0.6):
        assert bits == 1
        assert 0 < gap_start < gap_end < 1, "gap_start must be < gap_end"
        super().__init__(need_uniform_sampler=True, bits=bits)
        self.gap_start = gap_start
        self.gap_end = gap_end
        from scipy.stats import norm
        self._norm = norm

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        Encode secret using ICDF with gaps.

        Args:
            secret_message: (1,4,64,64) with values in {0, 1}
            ori_sample: Uniform [0,1] samples with same shape
        Returns:
            Latent noise with values from ICDF sub-intervals
        """
        u = ori_sample.copy()
        # Rescale uniform samples to the valid interval for each secret bit
        out = np.zeros_like(u)
        # bit=0: rescale [0,1] → [0, gap_start)
        mask0 = (secret_message == 0)
        out[mask0] = u[mask0] * self.gap_start
        # bit=1: rescale [0,1] → (gap_end, 1]
        mask1 = (secret_message == 1)
        out[mask1] = self.gap_end + u[mask1] * (1.0 - self.gap_end)
        # Apply inverse CDF to get Gaussian-distributed values
        return self._norm.ppf(out).astype(np.float64)

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        Decode by computing CDF and checking which interval the value falls in.

        p < gap_start → bit=0
        p > gap_end → bit=1
        p in [gap_start, gap_end] → treat as 0.5 (uncertain)
          (in practice, apply nearest-interval decoding)

        Returns: decoded secret with values in {0, 1}
        """
        p = self._norm.cdf(pred_noise)
        out = np.where(p >= self.gap_end, 1.0, np.where(p < self.gap_start, 0.0, 0.5))
        return out


# ============================================================
# Hadamard: Walsh-Hadamard Transform Mapping
# ============================================================
# Uses the Walsh-Hadamard transform (WHT) instead of QR decomposition
# for the orthogonal mapping. The Hadamard matrix has entries only ±1,
# making it computationally efficient (additions only, no multiplications).
#
# Unlike QR decomposition (which produces a random matrix), the Hadamard
# matrix is deterministic and fixed. This provides reproducible mixing
# without needing a seed for the kernel.
#
# Properties:
# - Orthogonal: H^T H = H H^T = nI (up to normalization)
# - Fixed: no randomness in the transform itself
# - Fast: can use FWHT (O(n log n)) instead of matrix multiply (O(n²))
# - Strong mixing: every output depends on every input


class hadamard_mapping(mapping_module):
    """
    Walsh-Hadamard Transform based orthogonal mapping.

    Replaces GRDH's random QR matrix with a fixed Hadamard matrix.
    The transform is: z = (1/64) * H @ secret @ H^T  (where H is 64×64)
    Normalization factor 1/64 ensures output ≈ N(0, 1).

    No seed_kernel needed (deterministic). Still uses seed_shuffle
    for the final random permutation.

    Capacity: same as ours_mapping (bits per element × 16384)
    """
    def __init__(self, bits=1):
        super().__init__(bits=bits)
        self.bits_mean = (self.bits_l - 1) / 2
        self.bits_std = ((self.bits_l ** 2 - 1) / 12) ** 0.5
        from scipy.linalg import hadamard
        self.H = hadamard(64).astype(np.float64)  # 64×64 Hadamard matrix

    def _random_shuffle(self, ori_input, seed_shuffle, reverse=False):
        """Same as ours_mapping._random_shuffle"""
        ori_seed = np.random.get_state()[1][0]
        np.random.seed(seed_shuffle)
        ori_shape = ori_input.shape
        flat = ori_input.flatten()
        order = np.arange(len(flat))
        np.random.shuffle(order)
        if reverse:
            out = flat[np.argsort(order)]
        else:
            out = flat[order]
        out = out.reshape(*ori_shape)
        np.random.seed(ori_seed)
        return out

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        Encode using Hadamard transform + random shuffle.

        z = (1/64) * H @ normalized_secret @ H^T
        Normalization: (secret - bits_mean) / bits_std
        """
        secret_re = (secret_message - self.bits_mean) / self.bits_std
        # H @ X @ H^T, normalized by 1/64
        out = np.matmul(np.matmul(self.H, secret_re), self.H.transpose(-1, -2)) / 64.0
        out = self._random_shuffle(out, seed_shuffle=seed_shuffle)
        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        Decode via inverse Hadamard transform.

        secret_hat = (1/64) * H @ de_shuffled @ H^T
        Then denormalize and round.
        """
        H = self.H
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)
        # Inverse Hadamard transform (same as forward since H is symmetric)
        secret_hat = np.matmul(np.matmul(H.transpose(-1, -2), pred_noise), H) / 64.0
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        out = np.round(secret_hat) % self.bits_l
        return out


# ============================================================
# Multi-ICDF: Multi-Bit Non-Continuous ICDF Mapping
# ============================================================
# Extension of icdf_mapping to support multi-bit encoding with
# non-contiguous intervals and protective gaps.
#
# For bits=2 (4 values per element):
#   '00' → Φ⁻¹([0,    0.20))   Gap: [0.20, 0.25)
#   '01' → Φ⁻¹([0.25, 0.45))   Gap: [0.45, 0.50)
#   '10' → Φ⁻¹([0.50, 0.70))   Gap: [0.70, 0.75)
#   '11' → Φ⁻¹([0.75, 0.95))   Gap: [0.95, 1.00)
#
# Each gap provides error detection margin. The gap size controls
# the robustness-vs-capacity tradeoff.


class multi_icdf_mapping(mapping_module):
    """
    Multi-bit ICDF-based non-continuous interval mapping.

    Args:
        bits: 1 or 2 (bits per element)
        gap: fraction of each interval used as protective gap (default 0.2)
    """
    def __init__(self, bits=1, gap=0.2):
        assert bits in (1, 2), "multi_icdf_mapping supports bits=1 or bits=2"
        super().__init__(need_uniform_sampler=True, bits=bits)
        self.gap = gap
        from scipy.stats import norm
        self._norm = norm
        self._build_intervals()

    def _build_intervals(self):
        """Build encoding intervals with gaps."""
        n = self.bits_l  # number of values (2 for bits=1, 4 for bits=2)
        total_width = 1.0 - n * self.gap  # total encoding width after gaps
        seg_width = total_width / n  # width per segment
        self.intervals = []
        for i in range(n):
            start = i * (seg_width + self.gap)
            end = start + seg_width
            self.intervals.append((float(start), float(end)))
        # Decode thresholds: midpoints between interval end and gap end
        self.decode_thresh = []
        for i in range(n - 1):
            thresh = (self.intervals[i][1] + self.intervals[i+1][0]) / 2.0
            self.decode_thresh.append(float(thresh))
        self.decode_thresh = np.array(self.decode_thresh)

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """Encode using ICDF sub-intervals."""
        u = ori_sample.copy()
        out = np.zeros_like(u)
        for val in range(self.bits_l):
            mask = (secret_message == val)
            if mask.any():
                start, end = self.intervals[val]
                out[mask] = start + u[mask] * (end - start)
        return self._norm.ppf(out).astype(np.float64)

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """Decode by CDF interval membership."""
        p = self._norm.cdf(pred_noise)
        out = np.zeros_like(p)
        for val in range(self.bits_l):
            start, end = self.intervals[val]
            mask = (p >= start) & (p < end)
            out[mask] = val
        # Handle values that fall in gaps: assign to nearest interval
        remaining = (out == 0) & (p >= self.intervals[0][1])  # not in interval 0
        if remaining.any():
            for val in range(1, self.bits_l):
                start, end = self.intervals[val]
                mid = (end + (self.intervals[val-1][1] if val > 0 else 0)) / 2.0
                if val == 1:
                    mask = remaining & (p < self.intervals[0][1] + (self.intervals[1][0] - self.intervals[0][1])/2)
                    out[mask] = 0
                    mask = remaining & (p >= self.intervals[0][1] + (self.intervals[1][0] - self.intervals[0][1])/2) & (p < end)
                    out[mask] = val
                else:
                    mask = remaining & (p >= start) & (p < end)
                    out[mask] = val
        return out


# ============================================================
# RS: Rejection Sampling Mapping (CGIS-style)
# ============================================================
# Implements the RS (Resampling-based) mapping from:
#   Wu et al., "Controllable Generative Image Steganography
#   Based on Denoising Diffusion Implicit Model", JKSU CIS 2026.
#
# For bit 0: sample N(0,1) until value > alpha  (positive large)
# For bit 1: sample N(0,1) until value < -alpha (negative large)
#
# Uses vectorized numpy for efficient rejection sampling.
# ============================================================


class rs_mapping(mapping_module):
    """
    Binary rejection sampling mapping (CGIS RS mapping).

    Args:
        bits: must be 1 (binary)
        alpha: rejection threshold (default 1.0). Higher = more robust but slower.
        max_iter: maximum rejection sampling iterations (default 100)
    """
    def __init__(self, bits=1, alpha=1.0, max_iter=100):
        assert bits == 1, "rs_mapping only supports bits=1"
        super().__init__(bits=bits)
        self.alpha = alpha
        self.max_iter = max_iter

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        Rejection sampling encode.

        For each element:
          bit=0 → sample N(0,1) until value > alpha
          bit=1 → sample N(0,1) until value < -alpha
        """
        out = np.random.randn(*secret_message.shape).astype(np.float64)
        mask_done = np.zeros(secret_message.shape, dtype=bool)

        for _ in range(self.max_iter):
            # Check conditions
            bit0 = (secret_message == 0) & ~mask_done
            bit1 = (secret_message == 1) & ~mask_done

            ok0 = bit0 & (out > self.alpha)
            ok1 = bit1 & (out < -self.alpha)

            mask_done = mask_done | ok0 | ok1

            if mask_done.all():
                break

            # Re-sample unsatisfied elements
            out = np.where(mask_done, out, np.random.randn(*secret_message.shape).astype(np.float64))

        if not mask_done.all():
            pct = 100.0 * mask_done.sum() / mask_done.size
            print(f"WARN: rs_mapping max_iter={self.max_iter} reached ({pct:.1f}% done)")

        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        Decode by threshold at 0.

        value > 0 → bit 0 (since RS encodes bit 0 as positive large)
        value < 0 → bit 1 (since RS encodes bit 1 as negative large)
        """
        out = np.where(pred_noise > 0, 0.0, 1.0)
        return out


# ============================================================
# MLQ-RS: Multi-Level Quantized Rejection Sampling
# ============================================================
# Extends binary RS to multi-bit encoding using multiple thresholds.
#
# For 2 bits per element (bits=2):
#   00 → value < -beta     (very negative)
#   01 → -beta < value < -alpha  (moderately negative)
#   10 → alpha < value < beta    (moderately positive)
#   11 → value > beta      (very positive)
#
# This directly increases embedding rate by log2(M) × for M intervals.
# ============================================================


class mlq_rs_mapping(mapping_module):
    """
    Multi-Level Quantized Rejection Sampling.

    Args:
        bits: bits per element (1 or 2)
        alpha: inner threshold (default 0.8)
        beta: outer threshold (default 1.5), must be > alpha for bits=2
        max_iter: maximum rejection sampling iterations (default 200)
    """
    def __init__(self, bits=1, alpha=0.8, beta=1.5, max_iter=200):
        assert bits in (1, 2), "mlq_rs_mapping supports bits=1 or bits=2"
        super().__init__(bits=bits)
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def _sample_interval(self, val):
        """
        Return the target interval for a given value.

        For bits=2 (values 0,1,2,3):
          0 or 4: (-inf, -beta)         → very negative
          1 or 5: (-beta, -alpha)       → moderately negative
          2 or 6: (alpha, beta)         → moderately positive
          3 or 7: (beta, +inf)          → very positive

        For bits=1 (values 0,1):
          0: (alpha, +inf)
          1: (-inf, -alpha)
        """
        n = self.bits_l  # 2 or 4
        half = n // 2  # 1 or 2
        if val < half:
            # Positive side
            idx = val  # 0 or 1 on positive side
            if n == 2:  # bits=1
                lo, hi = self.alpha, float('inf')
            else:  # bits=2
                if idx == 0:  # 2: (alpha, beta)
                    lo, hi = self.alpha, self.beta
                else:  # 3: (beta, +inf)
                    lo, hi = self.beta, float('inf')
        else:
            # Negative side
            idx = val - half
            if n == 2:  # bits=1
                lo, hi = float('-inf'), -self.alpha
            else:  # bits=2
                if idx == 0:  # 0: (-inf, -beta)
                    lo, hi = float('-inf'), -self.beta
                else:  # 1: (-beta, -alpha)
                    lo, hi = -self.beta, -self.alpha
        return lo, hi

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        Multi-level rejection sampling encode.
        """
        out = np.random.randn(*secret_message.shape).astype(np.float64)
        mask_done = np.zeros(secret_message.shape, dtype=bool)

        for _ in range(self.max_iter):
            remaining = ~mask_done
            if not remaining.any():
                break

            for val in range(self.bits_l):
                lo, hi = self._sample_interval(val)
                mask = (secret_message == val) & remaining
                if not mask.any():
                    continue

                if lo == float('-inf'):
                    ok = out < hi
                elif hi == float('inf'):
                    ok = out > lo
                else:
                    ok = (out > lo) & (out < hi)

                mask_done = mask_done | (mask & ok)

            if mask_done.all():
                break

            # Re-sample
            out = np.where(mask_done, out, np.random.randn(*secret_message.shape).astype(np.float64))

        if not mask_done.all():
            pct = 100.0 * mask_done.sum() / mask_done.size
            print(f"WARN: mlq_rs_mapping max_iter={self.max_iter} reached ({pct:.1f}% done)")

        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        Decode using threshold boundaries.
        """
        out = np.zeros(pred_noise.shape, dtype=np.float64)
        n = self.bits_l
        half = n // 2

        for val in range(n):
            lo, hi = self._sample_interval(val)
            if lo == float('-inf'):
                mask = pred_noise < hi
            elif hi == float('inf'):
                mask = pred_noise > lo
            else:
                mask = (pred_noise > lo) & (pred_noise < hi)
            out[mask] = val

        # Handle values that fall in gaps (between intervals)
        # Assign to nearest interval
        if n == 4:  # bits=2
            # Gap regions: (-alpha, alpha) and (-beta, -alpha) and (alpha, beta)
            # For values in the middle gap (-alpha, alpha):
            middle = (pred_noise > -self.alpha) & (pred_noise < self.alpha)
            # Assign to nearest: positive → 10 (value 2), negative → 01 (value 1)
            out[middle & (pred_noise >= 0)] = 2
            out[middle & (pred_noise < 0)] = 1

            # Gap between (-beta, -alpha) and (-inf, -beta): never happens with correct intervals
            # Gap between (alpha, beta) and (beta, +inf): never happens

        return out


# ============================================================
# RS-ECC: Rejection Sampling with Repetition Coding
# ============================================================
# Wraps rs_mapping with repetition coding + majority voting.
#
# Encoding:
#   N_data bits → repeat R times → N_total = N_data × R bits
#   → RS encode each repeated bit
#
# Decoding:
#   RS decode → group into R-tuples → majority vote → data bits
#
# The ECC corrects residual RS errors, allowing lower alpha
# (= fewer rejection iterations) at the same robustness level.
# ============================================================


class rs_ecc_mapping(mapping_module):
    """
    RS mapping with repetition coding and majority voting.

    Args:
        repeats: repetition factor R (3, 5, 7, ...)
        rs_alpha: RS threshold alpha (lower = fewer iterations)
        rs_max_iter: RS max iterations
        bits: must be 1
    """
    def __init__(self, repeats=3, rs_alpha=0.8, rs_max_iter=50, bits=1):
        assert bits == 1, "rs_ecc_mapping only supports bits=1"
        super().__init__(bits=bits)
        self.repeats = repeats
        self.rs = rs_mapping(alpha=rs_alpha, max_iter=rs_max_iter, bits=1)

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        Encode with repetition coding + RS.

        secret_message contains N_data data bits (values 0 or 1).
        Each bit is repeated R times to form the RS input.
        """
        shape = secret_message.shape
        n_total = np.prod(shape).astype(int)
        n_data = n_total // self.repeats

        # Flatten and repeat
        flat = secret_message.flatten().astype(np.float64)
        data_bits = flat[:n_data * self.repeats:self.repeats]  # check: take every R-th
        # Actually, we need to take first n_data bits and repeat them
        data_bits_actual = flat[:n_data]
        repeated = np.repeat(data_bits_actual, self.repeats)
        # Pad if needed
        if len(repeated) < n_total:
            repeated = np.pad(repeated, (0, n_total - len(repeated)), 'constant')
        secret_coded = repeated.reshape(shape)

        # RS encode the repeated bits
        return self.rs.encode_secret(secret_message=secret_coded)

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        RS decode → majority voting.
        """
        # RS decode
        decoded = self.rs.decode_secret(pred_noise=pred_noise)

        # Majority voting
        flat = decoded.flatten()
        n_total = len(flat)
        n_data = n_total // self.repeats
        groups = flat[:n_data * self.repeats].reshape(-1, self.repeats)
        data_bits = (np.sum(groups, axis=1) > self.repeats / 2).astype(np.float64)

        # Pad and reshape back
        out = np.pad(data_bits, (0, max(0, n_total - len(data_bits))), 'constant')[:n_total]
        return out.reshape(decoded.shape)


# ============================================================
# ND: Normal Distribution Mapping (CGIS-style)
# ============================================================
# Directly projects binary messages into normally-distributed
# latent variables using quantile-based encoding.
#
# Unlike RS (rejection sampling) which iteratively refines
# values until they pass a threshold, ND mapping is a single-pass
# deterministic projection — much faster, at the cost of
# lower robustness.
#
# For k bits per element, partitions N(0,1) into 2^k equal-
# probability intervals. Each k-bit value maps to the Φ⁻¹ of
# the midpoint of its CDF interval.
#
# bits=1:  0→Φ⁻¹(0.25)=-0.674,  1→Φ⁻¹(0.75)=0.674
# bits=2:  0→Φ⁻¹(0.125), 1→Φ⁻¹(0.375), 2→Φ⁻¹(0.625), 3→Φ⁻¹(0.875)
# bits=3:  8 intervals, etc.
#
# This directly enables multi-bit embedding: 2 bits → 2× rate,
# 3 bits → 3× rate, etc., all without rejection sampling.
# ============================================================


class nd_mapping(mapping_module):
    """
    Normal Distribution (ND) mapping — CGIS paper Algorithm 3 & 4.

    Two-level hierarchical mapping:
      Level 1 — n_main main intervals of width Δ = (b-a)/n_main cover [a,b].
      Level 2 — each main interval is subdivided into n_sub = 2^bits sub-intervals.

    Encoding (Algorithm 3, extended for multi-bit):
      For secret value d (0 .. 2^bits-1):
        1. Sub-interval li = d + 1  (which sub-interval within a main interval)
        2. Gaussian center at sub-interval midpoint:
           μ = a + (ji-1)·Δ + (li-0.5)·Δ/n_sub
        3. Sample y ~ N(μ, σ²)
      The main interval ji is a free parameter (default: ji=1).

    Decoding (Algorithm 4, two-level):
      1. Find main interval: ji = floor((y-a)/Δ) + 1
      2. Find sub-interval within ji: li = floor((y-a-(ji-1)Δ) · n_sub / Δ) + 1
      3. Decoded value = li - 1  (converted from sub-interval index)

    Args:
        bits: bits per element (1-10). Creates 2^bits sub-intervals per main interval.
        sigma: std of the normal distribution for sampling (default 0.5).
        a: ciphertext range start (default -3.0).
        b: ciphertext range end (default 3.0).
        n_main: number of main intervals (default 1). Larger = narrower main intervals.
    """
    def __init__(self, bits=1, sigma=0.5, a=-3.0, b=3.0, n_main=1):
        assert 1 <= bits <= 10, "nd_mapping supports bits=1..10"
        assert n_main >= 1, "n_main must be >= 1"
        super().__init__(bits=bits)
        self.sigma = sigma
        self.a = a
        self.b = b
        self.n_intervals = 2 ** bits    # total possible values (sub-intervals per main interval)
        self.n_main = n_main            # number of main intervals
        self.n_sub = self.n_intervals   # sub-intervals per main interval = 2^bits
        self.delta_main = (b - a) / n_main
        self.delta_sub = self.delta_main / self.n_sub
        self.delta = self.delta_sub  # backward compat: sub-interval width

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        """
        ND encode — Algorithm 3 extended for multi-bit.

        Centers Gaussian at the SUB-INTERVAL midpoint, not the main interval midpoint.
        For k-bit value d (0..2^k-1):
          li = d + 1  (sub-interval index, 1-indexed)
          ji = n_main // 2  (center main interval, or fixed at 1)
          μ = a + (ji-1)·Δ + (li-0.5)·Δ_sub
        """
        d = secret_message.astype(np.float64)
        li = d + 1.0
        ji = float(self.n_main // 2) if self.n_main > 1 else 1.0
        mu = self.a + (ji - 1.0) * self.delta_main + (li - 0.5) * self.delta_sub
        out = mu + self.sigma * np.random.randn(*secret_message.shape).astype(np.float64)
        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        ND decode — Algorithm 4 (two-level: ji → li).

        For each noise value y:
          1. ji = floor((y-a)/Δ) + 1             — main interval index
          2. li = floor((y-a-(ji-1)Δ)/Δ_sub) + 1 — sub-interval index within ji
          3. decoded = li - 1                    — k-bit secret value
        """
        # Step 4: main interval index ji
        ji = np.floor((pred_noise - self.a) / self.delta_main).astype(np.int64) + 1
        ji = np.clip(ji, 1, self.n_main)

        # Step 5: sub-interval index li within ji
        offset = pred_noise - self.a - (ji.astype(np.float64) - 1.0) * self.delta_main
        li = np.floor(offset / self.delta_sub).astype(np.int64) + 1
        li = np.clip(li, 1, self.n_sub)

        # Step 6-7: li → k-bit binary → decoded value
        out = (li - 1).astype(np.float64)
        return out


if __name__ == '__main__':
    bits = 1
    # # 我们的映射方法
    # # args = dict(seed_kernel=100, seed_shuffle=101)
    # # f = ours_mapping(bits=bits)
    #
    # # simple映射方法
    # # args = dict()
    # # f = simple_mapping()
    #
    # # tmm的映射方法
    # args = dict()
    # f = tmm_mapping(bits=bits)
    #
    # tdsc的映射方法
    init_args = dict(group_num=6, fixed=4096, bits=bits)
    f = tdsc_mapping(**init_args)
    args = dict(key=1001)
    ori_sample = None
    if f.need_uniform_sampler:
        ori_sample = np.random.rand(*(1, 4, 64, 64))
    if f.need_gaussian_sampler:
        ori_sample = np.random.randn(*(1, 4, 64, 64))
    secret = np.random.randint(0, 2**bits, (1, 4, 64, 64))  # 随机生成秘密信息
    z = f.encode_secret(secret_message=secret, ori_sample=ori_sample, **args)
    z_hat = z + np.random.randn(*(1, 4, 64, 64)) * 0.65
    secret_recon = f.decode_secret(pred_noise=z_hat, **args)
    #
    print('bpp:', f.cap/(64*64*4))
    # print(f._compute_acc(secret, secret_recon))
    # Test new mappings
    print("\n=== Testing BICS + FALE mappings ===")
    np.random.seed(42)
    secret = np.random.randint(0, 2, (1, 4, 64, 64))
    args_new = dict(seed_kernel=np.array([12345]), seed_shuffle=np.array([67890]))

    for name, cls in [("ours_mapping", ours_mapping),
                      ("bics_mapping(8)", bics_mapping),
                      ("fale_mapping(0.3,0.7)", fale_mapping),
                      ("combined_mapping", combined_mapping)]:
        if name == "bics_mapping(8)":
            f = bics_mapping(bits=1, block_size=8)
        elif name == "fale_mapping(0.3,0.7)":
            f = fale_mapping(bits=1, tau_a=0.3, tau_b=0.7)
        elif name == "combined_mapping":
            f = combined_mapping(bits=1, block_size=8, tau_a=0.3, tau_b=0.7)
        else:
            f = ours_mapping(bits=1)

        z = f.encode_secret(secret_message=secret.astype(np.float64), **args_new)
        recon = f.decode_secret(pred_noise=z.copy(), **args_new)
        lossless_acc = np.mean(recon == secret)

        # With noise
        z_noisy = z + np.random.randn(*z.shape) * 0.3
        recon_noisy = f.decode_secret(pred_noise=z_noisy, **args_new)
        noisy_acc = np.mean(recon_noisy == secret)

        print(f"  {name:25s} | lossless: {lossless_acc:.4f} | noisy: {noisy_acc:.4f}")
