from dataclasses import dataclass
from typing import Tuple
from molmobot_spoc.architecture.action_spaces.abstract import AbstractActionSpace
from molmobot_spoc.architecture.common.image_encoder import IMAGE_ENCODER
from molmobot_spoc.architecture.common.text_encoder import TEXT_ENCODER


@dataclass
class PreprocessorConfig:
    image_size: Tuple[int, int] = (224, 384)
    max_steps: int = None
    pad: bool = True

    action_space: AbstractActionSpace = None
    data_augmentation: bool = True
    augmentation_version: str = "v2"
    model_version: str = ""
    text_encoder_context_length: int = None

    image_encoder: IMAGE_ENCODER = IMAGE_ENCODER.Dinov2Small
    text_encoder: TEXT_ENCODER = TEXT_ENCODER.T5Small

    warp_images: bool = True
    warp_points: bool = True

    @property
    def mean(self):
        return self.image_encoder.value[1].mean

    @property
    def stdev(self):
        return self.image_encoder.value[1].stdev


@dataclass
class SigLipPreprocessorConfig(PreprocessorConfig):
    image_size: Tuple[int, int] = (256, 256)  # 256x256 for SigLIP

    model_version: str = "hf-hub:timm/ViT-B-16-SigLIP-256"
    text_encoder_context_length: int = 64

    image_encoder: IMAGE_ENCODER = IMAGE_ENCODER.SigLIP2BaseNoPool
    text_encoder: TEXT_ENCODER = TEXT_ENCODER.SigLIPBase
