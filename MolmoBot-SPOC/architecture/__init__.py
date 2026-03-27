from dataclasses import dataclass
from typing import List, Type, Any, Literal
from molmobot_spoc.architecture.action_spaces.quantile_based_binned_continuous import (
    QuantileBasedBinnedContinuousActionSpace,
)
from molmobot_spoc.architecture.config.model_config import (
    SpocGoalCondLlamaModelConfigXXL,
)
from molmobot_spoc.architecture.spoc_model import SpocContinuousActionModel

REGISTERED_MODELS = {}


@dataclass
class SPOCModelPackage:
    model_cls: Type
    config: Any
    input_sensors: List[str]

    def build_model(
        self, train_mode: Literal["IL", "RL"] = "IL", action_space_kwargs: dict = {}
    ):
        """Build model with action space initialization.

        Args:
            train_mode: Training mode ("IL" or "RL")
            action_space_kwargs: Optional kwargs to pass to action_space_cls constructor.
                                If provided, builds action space from action_space_cls.
                                If None, uses action_space if already set, or builds from action_space_cls with no args.
        """
        if self.config.action_space is None:
            self.config.action_space = self.config.action_space_cls(
                **action_space_kwargs
            )

        return self.model_cls(self.config, train_mode=train_mode)


# RB-Y1 MODELS

REGISTERED_MODELS["SpocLlamaModelWBinnedActionRBY1ArticulatedManipXXL"] = (
    SPOCModelPackage(
        model_cls=SpocContinuousActionModel,
        config=SpocGoalCondLlamaModelConfigXXL(
            observations=[
                "head_camera_features",
                "wrist_camera_r_features",
                "wrist_camera_l_features",
                "first_target_frame_repeated_features",
                "goal_text_features",
                "padding_mask",
                "proprioception",
                "pickup_obj_image_points",
            ],
            action_space_cls=QuantileBasedBinnedContinuousActionSpace,
            use_proprioception=True,
            use_qformer_sampling=False,
            use_visual_tokens=True,
        ),
        input_sensors=[
            "head_camera",
            "wrist_camera_r",
            "wrist_camera_l",
            "first_target_frame_repeated",
            "goal",
            "proprioception",
            "pickup_obj_image_points",
        ],
    )
)

REGISTERED_MODELS["SpocLlamaModelWBinnedActionRBY1RigidManipXXL"] = SPOCModelPackage(
    model_cls=SpocContinuousActionModel,
    config=SpocGoalCondLlamaModelConfigXXL(
        observations=[
            "head_camera_features",
            "wrist_camera_r_features",
            "wrist_camera_l_features",
            "goal_text_features",
            "padding_mask",
            "proprioception",
        ],
        action_space_cls=QuantileBasedBinnedContinuousActionSpace,
        use_proprioception=True,
        use_qformer_sampling=False,
        use_visual_tokens=True,
    ),
    input_sensors=[
        "head_camera",
        "wrist_camera_r",
        "wrist_camera_l",
        "goal",
        "proprioception",
    ],
)

# FRANKA MODELS

REGISTERED_MODELS["SpocLlamaModelWBinnedActionFrankaPickPlaceXXL"] = SPOCModelPackage(
    model_cls=SpocContinuousActionModel,
    config=SpocGoalCondLlamaModelConfigXXL(
        observations=[
            # "exo_camera_1_features",
            # "wrist_camera_features",
            "droid_shoulder_light_randomization_features",
            "wrist_camera_zed_mini_features",
            "goal_text_features",
            "padding_mask",
            "proprioception",
        ],
        action_space_cls=QuantileBasedBinnedContinuousActionSpace,
        use_proprioception=True,
        use_qformer_sampling=False,
        use_visual_tokens=True,
        proprio_additional_dim=1,
    ),
    input_sensors=[
        # "exo_camera_1",
        # "wrist_camera",
        "droid_shoulder_light_randomization",
        "wrist_camera_zed_mini",
        "goal",
        "proprioception",
    ],
)

REGISTERED_MODELS["SpocLlamaModelWBinnedActionFrankaPnPObjectPointsXXL"] = (
    SPOCModelPackage(
        model_cls=SpocContinuousActionModel,
        config=SpocGoalCondLlamaModelConfigXXL(
            observations=[
                "droid_shoulder_light_randomization_features",
                "wrist_camera_zed_mini_features",
                "goal_text_features",
                "padding_mask",
                "proprioception",
                "first_target_frame_repeated_features",
                "pickup_obj_image_points",
            ],
            action_space_cls=QuantileBasedBinnedContinuousActionSpace,
            use_proprioception=True,
            use_qformer_sampling=False,
            use_visual_tokens=True,
            proprio_additional_dim=1,
        ),
        input_sensors=[
            "droid_shoulder_light_randomization",
            "wrist_camera_zed_mini",
            "goal",
            "proprioception",
            "first_target_frame_repeated",
            "pickup_obj_image_points",
        ],
    )
)
