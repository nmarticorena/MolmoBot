import torch
import torch.nn as nn
from typing import Literal, Optional

from molmobot_spoc.architecture.config.model_config import SpocGoalCondLlamaModelConfig
from molmobot_spoc.architecture.common.action_decoder import ParallelActionDecoder
from molmobot_spoc.architecture.common.goal_cond_visual_encoder import (
    GoalCondVisualEncoder,
    GoalCondVisualEncoderWPointAsGoal,
)
from molmobot_spoc.architecture.common.utils import LinearActorHead
from molmobot_spoc.architecture.common.preprocessors import SigLipPreprocessor
from molmobot_spoc.architecture.config.model_config import SpocModelConfig
from molmobot_spoc.utils.train_utils import load_ckpt


class SpocContinuousActionModel(nn.Module):
    def __init__(
        self, cfg: SpocGoalCondLlamaModelConfig, train_mode: Literal["IL", "RL"] = "IL"
    ):
        super().__init__()
        self.cfg = cfg
        self.train_mode = train_mode
        self.visual_encoder = self.build_visual_encoder()
        self.action_decoder = self.build_action_decoder()
        self.actor = self.build_actor()
        self.preproc: Optional[SigLipPreprocessor] = None  # Will be built at train time

    @classmethod
    def build_preproc(cls, preproc_config, train_mode: Literal["IL", "RL"] = "IL"):
        if train_mode == "IL":
            return SigLipPreprocessor(preproc_config, device=torch.device("cpu"))
        else:
            raise NotImplementedError("Currently only IL train mode is implemented")

    @classmethod
    def build_agent(
        cls,
        cfg: SpocModelConfig,
        ckpt_pth: Optional[str] = None,
        device: str = "cpu",
    ):
        spoc_model = cls(cfg)
        load_ckpt(spoc_model, ckpt_pth)
        spoc_model = spoc_model.to(torch.device(device))
        spoc_model.preproc = spoc_model.build_preproc(cfg.preproc_config)
        spoc_model.preproc.to(torch.device(device))
        return spoc_model

    def build_action_decoder(self):
        return ParallelActionDecoder(self.cfg.action_decoder_config)

    def build_visual_encoder(self):
        if self.cfg.goal_point_uuid is None:
            return GoalCondVisualEncoder(self.cfg.visual_encoder_config)
        else:
            return GoalCondVisualEncoderWPointAsGoal(self.cfg.visual_encoder_config)

    def build_actor(self):
        return LinearActorHead(
            self.action_decoder.cfg.decoder.dim,
            self.cfg.action_space.get_num_actions(),
        )

    def forward_decoder(
        self,
        observations: dict[str, torch.Tensor],
        adapted_text_features: torch.Tensor,
        visual_tokens: torch.Tensor = None,
        point_tokens: torch.Tensor = None,
    ):
        proprioception = observations.get("proprioception", None)

        decoder_output = self.action_decoder(
            goals_features=adapted_text_features,
            proprioception=proprioception,
            visual_tokens=visual_tokens,
            point_tokens=point_tokens,
        )
        return decoder_output

    def get_visual_tokens(self, observations):
        frames_features_dict = {
            key: value
            for (key, value) in observations.items()
            if key in self.cfg.visual_features
        }
        B, T, _, _, _ = frames_features_dict[self.cfg.visual_features[0]].shape

        visual_tokens_list = []
        for k in self.visual_encoder.visual_features:
            frames_features = self.visual_encoder.adapt_frames_features(
                frames_features_dict[k]
            )  # BT, HW, D
            corresponding_camera_token = getattr(
                self.visual_encoder, f"visual_feature_token_{k}"
            )
            frames_features = frames_features + corresponding_camera_token
            visual_tokens_list.append(frames_features)
        all_visual_tokens = torch.cat(
            visual_tokens_list, dim=1
        )  # BT, num_visual_tokens, D
        return all_visual_tokens.reshape(
            B, T, -1, all_visual_tokens.shape[-1]
        )  # B, T, num_visual_tokens, D

    def forward(self, obs):
        observations = {key: obs[key] for key in self.cfg.observations}
        visual_tokens = self.get_visual_tokens(observations)
        adapted_text_features = self.visual_encoder.adapt_text_features(
            observations[self.cfg.goal_text_uuid]
        )
        if self.cfg.goal_point_uuid is not None:
            point_tokens = self.visual_encoder.adapt_goal_points_features(
                observations[self.cfg.goal_point_uuid]
            )
        else:
            point_tokens = None

        decoder_output = self.forward_decoder(
            observations, adapted_text_features, visual_tokens, point_tokens
        )

        action_logits = self.actor(decoder_output)
        return action_logits
