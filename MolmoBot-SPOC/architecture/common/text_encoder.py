from typing import Literal, Tuple
from dataclasses import dataclass
from open_clip import create_model_from_pretrained
from transformers import T5EncoderModel, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from enum import Enum


# Text encoder output dimensions reference:
# T5-small: 512
# SigLIP-Base: 768, SigLIP-Large: 1024, SigLIP-SO400M: 1152

T5_DIM_MAP = {
    "t5-small": 512,
}

SIGLIP_DIM_MAP = {
    "ViT-B-16-SigLIP-256": 768,
    "ViT-B-16-SigLIP-384": 768,
    "ViT-L-16-SigLIP-256": 1024,
    "ViT-L-16-SigLIP-384": 1024,
    "ViT-SO400M-14-SigLIP-384": 1152,
}

# SigLIP 2 models (Feb 2025) - use HuggingFace transformers
SIGLIP2_DIM_MAP = {
    "google/siglip2-so400m-patch14-384": 1152,
    "google/siglip2-base-patch16-224": 768,
    "google/siglip2-base-patch16-256": 768,
    "google/siglip2-large-patch16-384": 1024,
}


def create_text_encoder(encoder_name: str):
    if "siglip" in encoder_name.lower():
        encoder = create_model_from_pretrained(f"hf-hub:timm/{encoder_name}")[0].text
        encoder.output_tokens = True
        return encoder
    elif "t5" in encoder_name.lower():
        return T5EncoderModel.from_pretrained(encoder_name)
    else:
        raise NotImplementedError(
            f"Text encoder {encoder_name} not supported. Use SigLIP or T5 variants."
        )


@dataclass
class Config:
    model_name: str = "t5-small"
    is_siglip2: bool = (
        False  # Flag for SigLIP 2 models (use transformers instead of open_clip)
    )

    @property
    def goal_text_features_dim(self):
        # Check SigLIP 2 variants first
        if self.model_name in SIGLIP2_DIM_MAP:
            return SIGLIP2_DIM_MAP[self.model_name]
        # Check T5 variants
        if self.model_name in T5_DIM_MAP:
            return T5_DIM_MAP[self.model_name]
        # Check SigLIP variants
        for key, dim in SIGLIP_DIM_MAP.items():
            if key.lower() in self.model_name.lower():
                return dim
        raise NotImplementedError(f"Unknown text encoder: {self.model_name}")


class TextEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if "siglip" in cfg.model_name.lower() and not cfg.is_siglip2:
            self.encoder = create_model_from_pretrained(
                f"hf-hub:timm/{cfg.model_name}"
            )[0].text
            self.encoder.output_tokens = True
        elif "t5" in cfg.model_name.lower():
            self.encoder = T5EncoderModel.from_pretrained(cfg.model_name)
        else:
            raise NotImplementedError(f"Text encoder {cfg.model_name} not supported.")

    def forward(self, goal):
        # goal is the output from tokenizer, including input_ids, attention_mask, etc.
        if "siglip" in self.cfg.model_name.lower():
            with torch.no_grad():
                cls_feats, text_feats = self.encoder(goal)
            return torch.cat([text_feats, cls_feats.unsqueeze(1)], dim=1)
        elif "t5" in self.cfg.model_name.lower():
            with torch.no_grad():
                feats = self.encoder(**goal).last_hidden_state
            return feats
        else:
            raise NotImplementedError(
                f"Text encoder {self.cfg.model_name} not supported."
            )


class SigLIP2TextEncoder(nn.Module):
    """SigLIP 2 text encoder using HuggingFace transformers.
    Reference: https://huggingface.co/google/siglip2-so400m-patch14-384
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.eval()

    def forward(self, goal):
        # goal is tokenized input dict with input_ids, attention_mask
        with torch.no_grad():
            if isinstance(goal, dict):
                # Get full sequence of text embeddings (B, L, D), not just pooled
                text_outputs = self.model.text_model(**goal)
                # Return last_hidden_state which has shape (B, L, D)
                return text_outputs.last_hidden_state
            else:
                # Raw tensor - wrap in dict
                text_outputs = self.model.text_model(input_ids=goal)
                return text_outputs.last_hidden_state


class TEXT_ENCODER(Enum):
    # T5 (use with T5 tokenizer)
    T5Small = (TextEncoder, Config(model_name="t5-small"))  # 512 dim

    # SigLIP variants (use with SigLIP tokenizer via open_clip)
    SigLIPBase = (TextEncoder, Config(model_name="ViT-B-16-SigLIP-256"))  # 768 dim
    SigLIPLarge = (TextEncoder, Config(model_name="ViT-L-16-SigLIP-256"))  # 1024 dim
    SigLIPSO400M = (
        TextEncoder,
        Config(model_name="ViT-SO400M-14-SigLIP-384"),
    )  # 1152 dim

    SigLIP2Base = (
        SigLIP2TextEncoder,
        Config(model_name="google/siglip2-base-patch16-256", is_siglip2=True),
    )  # 768 dim, 256x256 images
    SigLIP2_SO400M = (
        SigLIP2TextEncoder,
        Config(model_name="google/siglip2-so400m-patch14-384", is_siglip2=True),
    )  # 1152 dim, best semantic understanding
