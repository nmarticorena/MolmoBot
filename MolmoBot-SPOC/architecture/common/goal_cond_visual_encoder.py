from typing import List, Optional, Literal
from enum import Enum

import torch
import torch.nn as nn
from molmobot_spoc.architecture.common.utils import PositionalEncoder
from molmobot_spoc.architecture.config.model_config import GoalCondVisualEncoderConfig


class GoalCondVisualEncoder(nn.Module):
    def __init__(self, cfg: GoalCondVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize the visual compressor and visual adapter,
        # where the visual compressor is a CNN that compresses the visual tokens spatially,
        # while the visual adapter project the feature to the transformer dimension
        self.visual_compressor = nn.Sequential(
            nn.Conv2d(self.cfg.frames_features_dim, self.cfg.dim, 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.dim, self.cfg.dim, 1),
            nn.ReLU(),
        )
        self.visual_adapter = nn.Sequential(
            nn.Linear(self.cfg.dim, self.cfg.dim),
            nn.LayerNorm(self.cfg.dim),
            nn.ReLU(),
        )

        # Initialize the text adapter, which projects the text features to the transformer dimension
        self.text_adapter = nn.Sequential(
            nn.Linear(cfg.goal_text_features_dim, self.cfg.dim),
            nn.LayerNorm(self.cfg.dim),
            nn.ReLU(),
        )

        # Initialize the visual sensor tokens.
        # The goal of this token is to allow the model to distinguish between different visual sensors.
        self.visual_features = sorted(self.cfg.visual_features)
        for feature in self.visual_features:
            setattr(
                self,
                f"visual_feature_token_{feature}",
                nn.Parameter(0.1 * torch.rand(self.cfg.dim)),
            )

    def adapt_text_features(self, text_features: torch.Tensor):
        return self.text_adapter(text_features)

    def adapt_frames_features(self, frames_features: torch.Tensor):
        B, T, C, H, W = frames_features.shape
        feats = self.visual_compressor(
            frames_features.reshape(B * T, C, H, W)
        )  # BTxC_xH_xW_
        _, C_, H_, W_ = feats.shape
        feats = feats.reshape(B * T, C_, H_ * W_).permute(0, 2, 1)  # BTxH_W_xC_
        return self.visual_adapter(feats)


class GoalCondVisualEncoderWPointAsGoal(GoalCondVisualEncoder):
    def __init__(self, cfg: GoalCondVisualEncoderConfig):
        super().__init__(cfg)
        # Support up to 2 points (pick and place): 2 points * 2 coords * 1 camera = 4
        # For pick tasks: 1 point * 2 coords * 1 camera = 2
        max_num_points = 2
        num_cameras = 1
        self.len_points_goals = max_num_points * 2 * num_cameras  # 4 for max support
        self.goal_as_point_pos_encoder = nn.Sequential(
            PositionalEncoder(32),
            nn.Linear(32, self.cfg.dim),
            nn.LayerNorm(self.cfg.dim),
            nn.ReLU(),
        )
        self.coord_pos_enc = nn.Embedding(self.len_points_goals, self.cfg.dim)

    def adapt_goal_points_features(self, goal_as_point: torch.Tensor):
        B, T, N = goal_as_point.shape
        # N can be 2 (pick: x, y) or 4 (pick-and-place: x1, y1, x2, y2)

        goal_as_point = goal_as_point.reshape(B * T, N)
        pos_encoded_points = self.goal_as_point_pos_encoder(goal_as_point)
        # Use only the first N indices from the positional encoding
        pos_encoded_points = pos_encoded_points + self.coord_pos_enc(
            torch.tensor(
                [[i for i in range(N)]],
                device=pos_encoded_points.device,
            ).tile(B * T, 1)
        )
        return pos_encoded_points


class RegisteredGoalCondVisualEncoder(Enum):
    GoalCondVisualEncoder = (GoalCondVisualEncoder, GoalCondVisualEncoderConfig())
    GoalCondVisualEncoderWPointAsGoal = (
        GoalCondVisualEncoderWPointAsGoal,
        GoalCondVisualEncoderConfig(
            frames_features_dim=768, goal_text_features_dim=768
        ),
    )
