from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from transformers import AutoModel
from enum import Enum


@dataclass
class Dinov2Config:
    model: str = "dinov2_vits14"
    output_size: Tuple[int, int, int] = (384, 7, 12)
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    @property
    def frames_features_dim(self):
        return self.output_size[0]


class Dinov2(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        self.model = torch.hub.load("facebookresearch/dinov2", cfg.model)
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (224, 384), f"Expected shape is 224x384; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x[:, :, :, 3:-3])["x_norm_patchtokens"]
            B, _, D = x.shape  # Bx432x384
            x = x.permute(0, 2, 1)  # Bx384x432
            x = x.reshape(B, D, 16, 27)
            x = self.pool(x)
            return x


@dataclass
class SigLIPConfig(Dinov2Config):
    model: str = "ViT-B-16-SigLIP-256"
    output_size: Tuple[int, int, int] = (768, 7, 12)
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)


class SigLIP(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        siglip_full_model = create_model_from_pretrained(
            "hf-hub:timm/{}".format(cfg.model)
        )
        self.model = siglip_full_model[0].visual.trunk
        self.context_length = siglip_full_model[0].context_length
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (256, 256), f"Expected shape is 256x256; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x)
            B, _, D = x.shape  # Bx256x768
            x = x.permute(0, 2, 1)  # Bx768x256
            x = x.reshape(B, D, 16, 16)
            x = self.pool(x)
            return x


class SigLIPNoPool(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        siglip_full_model = create_model_from_pretrained(
            "hf-hub:timm/{}".format(cfg.model)
        )
        self.model = siglip_full_model[0].visual.trunk
        self.context_length = siglip_full_model[0].context_length
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (256, 256), f"Expected shape is 256x256; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x)
            B, _, D = x.shape  # Bx256x768
            x = x.permute(0, 2, 1)  # Bx768x256
            x = x.reshape(B, D, 16, 16)
            return x


@dataclass
class SigLIP2BaseConfig(Dinov2Config):
    """Config for SigLIP2 Base with pooling to (7, 12) spatial dims."""

    model: str = "google/siglip2-base-patch16-256"
    output_size: Tuple[int, int, int] = (768, 7, 12)  # After pooling
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    image_size: int = 256
    patch_size: int = 16


@dataclass
class SigLIP2BaseNoPoolConfig(Dinov2Config):
    """Config for SigLIP2 Base without pooling - preserves full 16x16 spatial resolution."""

    model: str = "google/siglip2-base-patch16-256"
    output_size: Tuple[int, int, int] = (768, 16, 16)  # No pooling: 256/16 = 16
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    image_size: int = 256
    patch_size: int = 16


class SigLIP2Base(nn.Module):
    """SigLIP2 Base image encoder with spatial pooling.

    Input: (B, 3, 256, 256)
    Output: (B, 768, 7, 12) after adaptive pooling
    """

    def __init__(self, cfg: SigLIP2BaseConfig):
        super().__init__()
        self.cfg = cfg

        self.full_model = AutoModel.from_pretrained(cfg.model)

        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

        self.num_patches_side = cfg.image_size // cfg.patch_size  # 256/16 = 16
        self.num_patches = self.num_patches_side**2  # 256

    def forward(self, x):
        expected_size = self.cfg.image_size
        assert x.shape[-2:] == (expected_size, expected_size), (
            f"Expected shape is {expected_size}x{expected_size}; got {x.shape}"
        )

        with torch.no_grad():
            # Use pixel_values kwarg for HuggingFace models
            vision_outputs = self.full_model.vision_model(pixel_values=x)
            x = vision_outputs.last_hidden_state  # Bx256x768

            B, N, D = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(B, D, self.num_patches_side, self.num_patches_side)
            x = self.pool(x)
            return x


class SigLIP2BaseNoPool(nn.Module):
    """SigLIP2 Base image encoder without pooling - preserves full spatial resolution.

    Input: (B, 3, 256, 256)
    Output: (B, 768, 16, 16) - full patch grid preserved
    """

    def __init__(self, cfg: SigLIP2BaseNoPoolConfig):
        super().__init__()
        self.cfg = cfg

        self.full_model = AutoModel.from_pretrained(cfg.model)
        self.eval()

        self.num_patches_side = cfg.image_size // cfg.patch_size
        self.num_patches = self.num_patches_side**2

    def forward(self, x):
        expected_size = self.cfg.image_size
        assert x.shape[-2:] == (expected_size, expected_size), (
            f"Expected shape is {expected_size}x{expected_size}; got {x.shape}"
        )

        with torch.no_grad():
            # Use pixel_values kwarg for HuggingFace models
            vision_outputs = self.full_model.vision_model(pixel_values=x)
            x = vision_outputs.last_hidden_state

            B, N, D = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(
                B, D, self.num_patches_side, self.num_patches_side
            )  # Bx768x16x16
            return x


# ==================== SigLIP 2 SO400M ====================
# https://huggingface.co/google/siglip2-so400m-patch14-384


@dataclass
class SigLIP2Config:
    model: str = "google/siglip2-so400m-patch14-384"
    output_size: Tuple[int, int, int] = (1152, 7, 12)  # 1152 dim for SO400M
    image_size: int = 384  # SigLIP2 SO400M uses 384x384
    patch_size: int = 14  # SO400M uses 14x14 patches
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)

    @property
    def frames_features_dim(self):
        return self.output_size[0]


class SigLIP2(nn.Module):
    """SigLIP 2 image encoder using HuggingFace transformers.

    Reference: https://huggingface.co/google/siglip2-so400m-patch14-384

    Note: SigLIP2 vision models do NOT include a CLS token in last_hidden_state,
    so num_patches = (image_size / patch_size)^2 exactly.
    """

    def __init__(self, cfg: SigLIP2Config):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model)
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

        # Precompute expected spatial dimensions
        self.num_patches_side = (
            cfg.image_size // cfg.patch_size
        )  # 384/14 = 27 (rounded)

    def forward(self, x):
        # x is already preprocessed tensor (B, 3, H, W)
        expected_size = self.cfg.image_size
        assert x.shape[-2:] == (expected_size, expected_size), (
            f"Expected shape is {expected_size}x{expected_size}; got {x.shape}"
        )

        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=x)
            x = vision_outputs.last_hidden_state  # B, num_patches, D

            B, num_patches, D = x.shape
            h = w = int(round(num_patches**0.5))

            x = x.permute(0, 2, 1)  # B, D, num_patches
            x = x.reshape(B, D, h, w)
            x = self.pool(x)
            return x


class IMAGE_ENCODER(Enum):
    Dinov2Small = (Dinov2, Dinov2Config())
    Dinov2Base = (Dinov2, Dinov2Config(model="dinov2_vitb14", output_size=(768, 7, 12)))
    SigLIPBase = (SigLIP, SigLIPConfig())
    SigLIPNoPool = (SigLIPNoPool, SigLIPConfig())
    SigLIPLarge = (
        SigLIP,
        SigLIPConfig(model="ViT-L-16-SigLIP-256", output_size=(1024, 7, 12)),
    )
    # SigLIP 2 variants (newest, Feb 2025)
    SigLIP2_SO400M = (SigLIP2, SigLIP2Config())  # 1152 dim, 1B params - best quality
    SigLIP2Base = (SigLIP2Base, SigLIP2BaseConfig())  # 768 dim, output (768, 7, 12)
    SigLIP2BaseNoPool = (
        SigLIP2BaseNoPool,
        SigLIP2BaseNoPoolConfig(),
    )  # 768 dim, output (768, 16, 16)


if __name__ == "__main__":
    encoder_cls, cfg = IMAGE_ENCODER.SigLIP2Base.value
    model = encoder_cls(cfg)

    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {output.shape}")
    # Expected: Input: torch.Size([2, 3, 256, 256]) -> Output: torch.Size([2, 768, 7, 12])
