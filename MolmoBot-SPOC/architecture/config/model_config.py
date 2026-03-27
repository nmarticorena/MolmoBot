from dataclasses import dataclass, field
from typing import Literal, Optional, List, Union, Callable, Any
import torch.nn as nn
from molmobot_spoc.architecture.common.utils import TransformerConfig
from molmobot_spoc.architecture.config.preproc_config import SigLipPreprocessorConfig
from molmobot_spoc.architecture.action_spaces.abstract import AbstractActionSpace
import torch
from molmobot_spoc.architecture.common.goal_cond_llama import (
    ModelArgs as GoalCondLlamaConfig,
)
from molmobot_spoc.utils.constants.sensor_constants import (
    is_a_goal_features,
    is_a_visual_features,
)


GOAL_POINT_UUIDS = ["pickup_obj_image_points"]


@dataclass
class GoalCondVisualEncoderConfig:
    dim: int = 512
    visual_features: List[str] = None
    frames_features_dim: int = None
    goal_text_features_dim: int = None
    goal_text_uuid: str = None
    goal_point_uuid: Optional[Literal["pickup_obj_image_points"]] = None


@dataclass
class ActionDecoderConfig:
    decoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(3, 512, 8, True)
    )
    max_length: int = 1000
    action_space: AbstractActionSpace = None
    use_proprioception: bool = False
    proprio_additional_dim: int = (
        2  # Additional dimensions added to proprioception input
    )
    # QFormer visual token sampling
    use_qformer_sampling: bool = False  # If True, use QFormer to sample visual tokens
    qformer_num_query: int = 32  # Number of query tokens to sample
    qformer_num_layers: int = 3  # Number of transformer decoder layers
    qformer_num_heads: int = 8  # Number of attention heads


@dataclass
class GoalCondLlamaActionDecoderConfig:
    n_layers: int
    dim: int
    n_heads: int
    output_size: int
    max_batch_size: int
    max_seq_len: int
    an_object_is_in_hand: bool = True
    action_space: AbstractActionSpace = None
    dropout: float = 0.1
    use_rms_norm: bool = True
    norm_first: bool = True
    activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.functional.silu
    use_proprioception: bool = False
    proprio_additional_dim: int = (
        2  # Additional dimensions added to proprioception input
    )
    # QFormer visual token sampling
    use_qformer_sampling: bool = False  # If True, use QFormer to sample visual tokens
    qformer_num_query: int = 32  # Number of query tokens to sample
    qformer_num_layers: int = 3  # Number of transformer decoder layers
    qformer_num_heads: int = 8  # Number of attention heads

    @property
    def decoder(self) -> GoalCondLlamaConfig:
        return GoalCondLlamaConfig(
            n_layers=self.n_layers,
            dim=self.dim,
            n_heads=self.n_heads,
            output_size=self.output_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_rms_norm=self.use_rms_norm,
            norm_first=self.norm_first,
            activation=self.activation,
        )


@dataclass
class SpocModelConfig:
    # observations is a list of observation uuids that are used in the model
    observations: List[str] = None

    # Store the action space class (not instance) - will be initialized in build_model()
    action_space_cls: type = None
    action_space: Any = None  # Will be set when build_model() is called

    separate_visual_encoder: bool = False

    # ParallelActionDecoder parameters
    parallel_decoder_num_heads: int = 8
    parallel_decoder_num_layers: int = 2

    use_proprioception: bool = False
    proprio_additional_dim: int = (
        2  # Additional dimensions added to proprioception input
    )

    # Visual tokens for action decoder cross-attention
    # When True, visual encoder returns spatial tokens for the decoder to attend to
    use_visual_tokens: bool = False

    # QFormer visual token sampling
    # When True, uses QFormer to sample a fixed number of visual tokens
    use_qformer_sampling: bool = False

    @property
    def preproc_config(self):
        if self.action_space is None:
            raise ValueError(
                "action_space must be initialized before accessing preproc_config. Call build_model() first."
            )
        return SigLipPreprocessorConfig(action_space=self.action_space)

    @property
    def frames_features_dim(self):
        # frames_features_dim is used in GoalCondVisualEncoder
        # It should be determined by which image encoder is used
        # For example, SigLIP's frames_features_dim is 768, while
        # DINOv2's frames_features_dim is 384
        return self.preproc_config.image_encoder.value[1].frames_features_dim

    @property
    def goal_text_features_dim(self):
        # goal_text_features_dim is used in GoalCondVisualEncoder
        # It should be determined by which text encoder is used
        # For example, SigLIP's goal_text_features_dim is 768, while
        # T5's goal_text_features_dim is 512
        return self.preproc_config.text_encoder.value[1].goal_text_features_dim

    @property
    def visual_features(self) -> List[str]:
        return [obs for obs in self.observations if is_a_visual_features(obs)]

    @property
    def goals_features(self) -> List[str]:
        return [obs for obs in self.observations if is_a_goal_features(obs)]

    @property
    def goal_text_uuid(self) -> str:
        if "goal_text_features" in self.observations:
            return "goal_text_features"
        else:
            return None

    @property
    def goal_point_uuid(
        self,
    ) -> Optional[Literal["pickup_obj_image_points"]]:
        goal_point_obs = set(self.observations).intersection(set(GOAL_POINT_UUIDS))
        if len(goal_point_obs) == 0:
            return None
        return goal_point_obs.pop()

    @property
    def visual_encoder_config(self) -> GoalCondVisualEncoderConfig:
        return GoalCondVisualEncoderConfig(
            visual_features=self.visual_features,
            goal_text_uuid=self.goal_text_uuid,
            goal_point_uuid=self.goal_point_uuid,
            frames_features_dim=self.frames_features_dim,
            goal_text_features_dim=self.goal_text_features_dim,
        )

    @property
    def action_decoder_config(self) -> ActionDecoderConfig:
        return ActionDecoderConfig(
            action_space=self.action_space,
            use_proprioception=self.use_proprioception,
            proprio_additional_dim=self.proprio_additional_dim,
            use_qformer_sampling=self.use_qformer_sampling,
        )


@dataclass
class SpocGoalCondLlamaModelConfig(SpocModelConfig):
    batch_size: int = 224
    max_seq_len: int = 320
    dropout: float = 0.1
    use_rms_norm: bool = True
    norm_first: bool = True
    activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.functional.silu

    @property
    def action_decoder_config(self) -> GoalCondLlamaActionDecoderConfig:
        return GoalCondLlamaActionDecoderConfig(
            action_space=self.action_space,
            n_layers=3,
            dim=512,
            n_heads=8,
            output_size=512,
            max_batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_rms_norm=self.use_rms_norm,
            norm_first=self.norm_first,
            activation=self.activation,
            use_proprioception=self.use_proprioception,
            proprio_additional_dim=self.proprio_additional_dim,
            use_qformer_sampling=self.use_qformer_sampling,
        )


@dataclass
class SpocGoalCondLlamaModelConfigXXL(SpocGoalCondLlamaModelConfig):
    """Extra large version of SpocGoalCondLlamaModelConfig with ~150M parameters."""

    @property
    def action_decoder_config(self) -> GoalCondLlamaActionDecoderConfig:
        return GoalCondLlamaActionDecoderConfig(
            action_space=self.action_space,
            n_layers=24,  # 3 → 24 (8x layers)
            dim=512,
            n_heads=32,  # 8 → 32 (4x heads)
            output_size=512,
            max_batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_rms_norm=self.use_rms_norm,
            norm_first=self.norm_first,
            activation=self.activation,
            use_proprioception=self.use_proprioception,
            proprio_additional_dim=self.proprio_additional_dim,
            use_qformer_sampling=self.use_qformer_sampling,
        )
