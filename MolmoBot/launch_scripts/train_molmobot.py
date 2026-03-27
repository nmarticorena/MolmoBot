"""
Training script for MolmoBot SynthManip model using SynthManip-format robot demonstration data.

Usage:
    python launch_scripts/train_molmobot.py <checkpoint> \\
        --data_paths /path/to/TaskConfig1 /path/to/TaskConfig2 \\
        --action_preset RBY1_full \\
        --camera_preset RBY1_right_arm \\
        --seq_len 2048 \\
        --weighted_sampling  # Optional: upweight grasps, downweight failures

Data paths:
    Each --data_paths argument must point to a task type directory containing:
        {data_path}/train/house_*/*.h5
        {data_path}/val/house_*/*.h5
    
    Multiple data paths can be provided to aggregate data from multiple task types.

Configuration:
    Camera and action configuration must be explicitly specified via presets or direct args:
    - --action_preset: Action preset (e.g., RBY1_full, franka_joint)
    - --camera_preset: Camera preset (e.g., RBY1_right_arm)
    - --camera_names: Explicit camera names (alternative to preset)
    - --action_move_groups: Explicit move groups (alternative to preset)

Validation:
    - Automatically runs validation on the 'val' split at intervals specified by --val_interval
    - Computes validation loss metrics (action loss, language loss, total loss)
    - Logs validation curves to wandb
    - Configure with --val_max_examples and --val_interval

Environment variables:
    WANDB_PROJECT: W&B project name
    WANDB_ENTITY: W&B entity
"""

import os
import argparse
from os.path import join
from typing import Optional
from dataclasses import asdict, fields

from omegaconf import omegaconf, OmegaConf

from olmo.data.data_loader import DataLoaderConfig
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.data.synthmanip_dataset import synthmanip_config_registry, SynthmanipDatasetConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.io import file_exists
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmobot.molmobot import MolmoBotConfig
from olmo.models.video_olmo.video_preprocessor import MultiModalVideoPreprocessorConfig
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.torch_util import get_world_size
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.run_trainer import run_trainer
from olmo.train.trainer_config import (
    TrainConfig,
    CompilerConfig,
    FSDPConfig,
    BatchDivisor,
    SpeedMonitorConfig,
    WandbConfig,
)
from olmo.util import prepare_torchrun_environment, clean_opt, select_checkpoint

from olmo.train_init_utils import compute_state_action_normalization_stats, get_synthmanip_training_data, validate_data_path


def get_model(checkpoint: Optional[str], model_type: str) -> MolmoBotConfig:
    """Build MolmoBotConfig from checkpoint or default configuration."""
    if checkpoint and file_exists(join(checkpoint, "model.yaml")):
        model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
    elif checkpoint:
        model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")
    else:
        raise ValueError("Must provide a checkpoint path or '8b'")

    # Video preprocessor for handling robot observation frames
    video_pre_processor_cfg = MultiModalVideoPreprocessorConfig(
        use_col_tokens=False,
        max_crops=1,
        pooling_h=3,
        pooling_w=3,
        high_res_pooling_h=None,
        high_res_pooling_w=None,
        periodic_high_res_frame=None,
        time_mode="per-frame-compact",
        max_frames=1,
        time_sampling=True,
        loading_method="torchcodec_exact",
        frame_sample_mode="uniform_last_frame",
        max_fps=[2],
    )

    if isinstance(model_cfg.mm_preprocessor, MultiCropConfig):
        image_preprocessor_args = asdict(model_cfg.mm_preprocessor)
        image_preprocessor = MultiCropConfig(
            **{k.name: image_preprocessor_args[k.name] for k in fields(MultiCropConfig)}
        )
        image_preprocessor.high_res_max_crops = 24
        image_preprocessor.p_high_res = 0
        video_pre_processor_cfg.image = image_preprocessor
    else:
        video_pre_processor_cfg.image = model_cfg.mm_preprocessor.image

    model_cfg = MolmoBotConfig(
        llm=model_cfg.llm,
        vision_backbone=model_cfg.vision_backbone,
        data_formatter=model_cfg.data_formatter,
        mm_preprocessor=video_pre_processor_cfg,
        bi_directional_attn=model_cfg.bi_directional_attn,
    )

    # Fine-tuning settings
    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model_v2"
    model_cfg.data_formatter.message_format = "qwen3"
    model_cfg.data_formatter.system_prompt = "demo_or_style_v2"
    model_cfg.data_formatter.pointing_format = "html-v2"
    model_cfg.data_formatter.p_multi_point_all_image = 0.5
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments_root_tokens"
    model_cfg.data_formatter.p_choice_content_in_mc = 1.0
    model_cfg.vision_backbone.pooling_attention_mask = True

    # Multi-image settings (for multiple camera views)
    model_cfg.mm_preprocessor.image.max_multi_image_crops = 1
    model_cfg.mm_preprocessor.image.max_images = 2

    model_cfg.llm.max_sequence_length = 4096 * 2

    # Reduce shared memory requirements
    model_cfg.vision_backbone.normalize_on_gpu = True

    # Action expert settings
    model_cfg.action_expert.num_layers = model_cfg.llm.n_layers
    model_cfg.action_expert.hidden_size = 768
    model_cfg.action_expert.num_heads = 8
    model_cfg.action_expert.max_horizon = max(
        model_cfg.action_expert.max_horizon, model_cfg.action_horizon
    )

    return model_cfg


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train SynthManip MolmoBot")
    parser.add_argument("checkpoint", help="Path to checkpoint or '8b' for base model")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="Paths to task type directories. Each must contain train/ and val/ subdirs with house_*/*.h5 files.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", default="video")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--device_batch_size", default=2, type=int)
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # Action configuration
    parser.add_argument("--action_dim", default=14, type=int)
    parser.add_argument("--action_horizon", default=16, type=int)
    parser.add_argument("--n_obs_steps", default=1, type=int)
    parser.add_argument("--obs_step_delta", default=1, type=int)
    parser.add_argument("--n_action_steps", default=8, type=int)
    parser.add_argument("--action_expert_layer_mode", default="per_layer", type=str)

    # Robot configuration - must be explicitly specified via preset or direct args
    parser.add_argument(
        "--action_preset", type=str, help="Action preset name (e.g. RBY1_full, franka_joint). Required if --action_move_groups not provided.")
    parser.add_argument(
        "--camera_preset", type=str, help="Camera preset name (e.g. RBY1_right_arm). Required if --camera_names not provided.")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=None,
        help="List of camera names. Required if --camera_preset not provided.",
    )
    parser.add_argument(
        "--action_move_groups",
        nargs="+",
        default=None,
        help="List of action move groups. Required if --action_preset not provided.",
    )
    parser.add_argument(
        "--action_type", 
        default="delta_actions", 
        help="Action type suffix (e.g. delta_actions, absolute_actions)"
    )

    # Training options
    parser.add_argument("--img_aug", action="store_true", help="Enable image augmentation")
    parser.add_argument("--no_stats", action="store_true", help="Skip normalization stats computation")
    parser.add_argument("--stats_path", type=str, default="synthmanip_norm_stats.yaml", help="Path to save/load normalization stats")
    parser.add_argument("--weighted_sampling", action="store_true", help="Use grasp-aware weighted timestep sampling (upweights final grasps, downweights failed grasps)")
    parser.add_argument("--randomize_prompts", action="store_true", help="Randomize prompts for each action sample")
    parser.add_argument("--furthest_camera_prob", type=float, default=0.0, help="Probability of explicitly selecting the furthest camera")
    parser.add_argument(
        "--use_point_prompts",
        action="store_true",
        help=(
            "Append object point annotations to goal string (Molmo html-v2 <points> format). "
            "Uses head_camera by default; override with --point_prompt_camera."
        ),
    )
    parser.add_argument(
        "--point_prompt_camera",
        type=str,
        default="head_camera",
        help="Camera to use for point prompts (default: head_camera). Only used with --use_point_prompts.",
    )
    parser.add_argument(
        "--conditioning_frame",
        type=str,
        default="random_first_10",
        help="Conditioning frame for object image points: an integer frame index or 'random_first_10'.",
    )
    parser.add_argument(
        "--cameras_to_warp",
        nargs="*",
        default=[],
        help="Cameras to apply GoPro fisheye warping (resize to 640x480 4:3 + barrel distortion). E.g. --cameras_to_warp head_camera",
    )
    parser.add_argument(
        "--max_points_in_conditioning_frame",
        type=int,
        default=1,
        help="Max number of points per object per camera from the conditioning frame (default: 10).",
    )
    parser.add_argument(
        "--use_point_prompts_per_dataset",
        nargs="+",
        type=int,
        default=None,
        help="Per-dataset point prompt toggle (0 or 1, one per --data_paths). Overrides --use_point_prompts when set.",
    )

    # Checkpoint resume options
    parser.add_argument("--reset_optimizer", action="store_true", help="Reset optimizer state when resuming from checkpoint (use if optimizer state is incompatible)")

    # Validation options
    parser.add_argument("--no_val", action="store_true", help="Disable validation entirely.")
    parser.add_argument("--val_data_paths", nargs="+", default=None, help="Explicit path(s) to val data dir(s) (each must contain val/house_*/*.h5). If not set, uses val/ from first --data_paths entry.")
    parser.add_argument("--val_max_examples", type=int, default=2000, help="Max examples for validation")
    parser.add_argument("--val_interval", type=int, default=1000, help="Validation interval")
    parser.add_argument("--dataset_sample_rates", nargs="+", type=float, default=None, help="Per-dataset sample rates (one per --data_paths). Equal weighting if not set.")

    parser.add_argument("--max_exo_views", type=int, default=1, help="Maximum number of exo views per frame (default: 1)")
    parser.add_argument("--exp_name", type=str, default="molmobot_training", help="Unique experiment name.")

    args, other_args = parser.parse_known_args()

    # Handle action preset
    action_spec_from_preset = None
    state_spec_from_preset = None
    state_indices_from_preset = {}
    if args.action_preset:
        from olmo.data.synthmanip_presets import ACTION_SPECS, ACTION_DATASET_KEYS, STATE_SPECS, STATE_INDICES
        if args.action_preset in ACTION_SPECS:
            print(f"Using action preset: {args.action_preset}")
            action_spec_from_preset = ACTION_SPECS[args.action_preset]
            args.action_move_groups = list(action_spec_from_preset.keys())
            state_spec_from_preset = STATE_SPECS.get(args.action_preset)
            state_indices_from_preset = STATE_INDICES.get(args.action_preset, {})
        else:
            raise ValueError(f"Unknown action preset: {args.action_preset}")
            
    # Handle camera preset
    if args.camera_preset:
        from olmo.data.synthmanip_presets import CAMERA_PRESETS
        if args.camera_preset in CAMERA_PRESETS:
            print(f"Using camera preset: {args.camera_preset}")
            args.camera_names = CAMERA_PRESETS[args.camera_preset]
        else:
            raise ValueError(f"Unknown camera preset: {args.camera_preset}")

    # Validate required configuration
    if args.camera_names is None:
        raise ValueError(
            "Camera names must be specified. Use --camera_names or --camera_preset.\n"
            "Example: --camera_names head_camera wrist_camera\n"
            "Example: --camera_preset RBY1_right_arm"
        )
    
    if args.action_move_groups is None:
        raise ValueError(
            "Action move groups must be specified. Use --action_move_groups or --action_preset.\n"
            "Example: --action_move_groups base head right_arm right_gripper\n"
            "Example: --action_preset RBY1_full"
        )
    
    print(f"Using camera_names: {args.camera_names}")
    print(f"Using action_move_groups: {args.action_move_groups}")
    
    # Build action spec - must come from preset, no fallback guessing
    if action_spec_from_preset is None:
        raise ValueError(
            "Action spec must be specified via --action_preset.\n"
            "Available presets: RBY1_full, franka_joint, franka_jointdelta, franka_eedelta, franka_ee"
        )
    action_spec = {mg: action_spec_from_preset[mg] for mg in args.action_move_groups}

    # Build state spec/indices (falls back to action_spec when no state preset exists)
    state_spec = {mg: state_spec_from_preset[mg] for mg in args.action_move_groups} if state_spec_from_preset else None
    state_indices = state_indices_from_preset or None

    # Build action keys from preset
    from olmo.data.synthmanip_presets import ACTION_DATASET_KEYS
    action_key = ACTION_DATASET_KEYS.get(args.action_preset, args.action_type)
    if isinstance(action_key, dict):
        action_keys = {mg: action_key[mg] for mg in args.action_move_groups}
    else:
        action_keys = {mg: action_key for mg in args.action_move_groups}

    # Compute actual action dim from spec
    actual_action_dim = sum(action_spec[mg] for mg in args.action_move_groups)
    if args.action_dim != actual_action_dim:
        print(f"Overriding action_dim from {args.action_dim} to {actual_action_dim} based on move groups")
        args.action_dim = actual_action_dim

    # Build model config
    checkpoint = select_checkpoint(args.checkpoint)
    model_cfg = get_model(checkpoint, args.model)

    model_cfg.mm_preprocessor.max_subtitle_tokens = None
    model_cfg.action_dim = args.action_dim
    model_cfg.action_expert.action_dim = args.action_dim
    model_cfg.action_horizon = args.action_horizon
    model_cfg.n_obs_steps = args.n_obs_steps
    model_cfg.obs_step_delta = args.obs_step_delta
    model_cfg.n_action_steps = args.n_action_steps
    model_cfg.action_expert.max_horizon = max(
        model_cfg.action_expert.max_horizon, args.action_horizon
    )
    model_cfg.vision_backbone.use_image_augmentation = args.img_aug
    model_cfg.action_expert_layer_mode = args.action_expert_layer_mode

    data_paths = args.data_paths
    dataset_sample_rates = args.dataset_sample_rates

    # Load or compute normalization stats
    proc_cfg: Optional[RobotProcessorConfig] = None
    if not args.no_stats:
        stats_path = args.stats_path
        if os.path.exists(stats_path):
            print(f"Loading existing normalization stats from {stats_path}")
            proc_cfg = RobotProcessorConfig.load(stats_path)
            existing_stats = proc_cfg.stats_by_repo.get("synthmanip", {})

            # Check if state stats are missing
            if "observation.state" not in existing_stats or "action" not in existing_stats:
                print("Computing missing normalization statistics...")

                # Compute state stats only
                state_stats, action_norm_mode, state_norm_mode = compute_state_action_normalization_stats(
                    data_paths=data_paths,
                    camera_names=args.camera_names,
                    action_move_group_names=args.action_move_groups,
                    action_spec=action_spec,
                    action_keys=action_keys,
                    action_horizon=args.action_horizon,
                    input_window_size=args.n_obs_steps,
                    num_workers=args.num_workers,
                    state_spec=state_spec,
                    state_indices=state_indices,
                )
                if "action" in existing_stats:
                    state_stats.pop("action", None)
                if "observation.state" in existing_stats:
                    state_stats.pop("observation.state", None)

                # Merge with existing stats (preserving action stats)
                final_stats = {**existing_stats, **state_stats}

                # Create final processor config
                proc_cfg = RobotProcessorConfig.from_stats(
                    stats_by_repo={"synthmanip": final_stats},
                    default_repo_id=proc_cfg.default_repo_id if proc_cfg is not None else "synthmanip",
                    action_norm_mode=action_norm_mode if "action" not in existing_stats else proc_cfg.action_norm_mode,
                    state_norm_mode=state_norm_mode if "observation.state" not in existing_stats else proc_cfg.state_norm_mode,
                )

                # Save updated stats
                print(f"Saving updated normalization stats to {stats_path}")
                proc_cfg.save(stats_path)

        else:
            print(f"Computing normalization statistics from dataset (saving to {stats_path})...")
            stats, action_norm_mode, state_norm_mode = compute_state_action_normalization_stats(
                data_paths=data_paths,
                camera_names=args.camera_names,
                action_move_group_names=args.action_move_groups,
                action_spec=action_spec,
                action_keys=action_keys,
                action_horizon=args.action_horizon,
                input_window_size=args.n_obs_steps,
                num_workers=args.num_workers,
                state_spec=state_spec,
                state_indices=state_indices,
            )
            proc_cfg = RobotProcessorConfig.from_stats(
                stats_by_repo={"synthmanip": stats},
                default_repo_id="synthmanip",
                action_norm_mode=action_norm_mode,
                state_norm_mode=state_norm_mode,
            )

            if stats_path:
                print(f"Saving normalization stats to {stats_path}")
                proc_cfg.save(stats_path)

        model_cfg.robot_preprocessor = proc_cfg
        model_cfg.robot_postprocessor = proc_cfg

    if args.debug:
        checkpoint = None
        model_cfg.llm.init_path = None
        model_cfg.llm.n_layers = 4
        model_cfg.vision_backbone.vit.init_path = None
        model_cfg.vision_backbone.vit.image_num_layers = 2
        model_cfg.vision_backbone.vit_layers = [-1, -2]
        model_cfg.action_expert.num_layers = model_cfg.llm.n_layers
        args.num_workers = 2
        args.prefetch_factor = 2

    # Parse conditioning_frame: int string → int, else keep as string (e.g. "random_first_10")
    conditioning_frame = int(args.conditioning_frame) if args.conditioning_frame.isdigit() else args.conditioning_frame

    has_separate_val_paths = args.val_data_paths is not None

    # Build data mixture - pass the already-loaded processor config
    action_mixture = get_synthmanip_training_data(
        data_paths=data_paths,
        dataset_sample_rates=dataset_sample_rates,
        camera_names=args.camera_names,
        action_move_group_names=args.action_move_groups,
        action_spec=action_spec,
        action_keys=action_keys,
        action_horizon=args.action_horizon,
        input_window_size=args.n_obs_steps,
        obs_step_delta=args.obs_step_delta,
        robot_processor_config=proc_cfg,
        weighted_sampling=args.weighted_sampling,
        randomize_prompts=args.randomize_prompts,
        furthest_camera_prob=args.furthest_camera_prob,
        use_point_prompts=args.use_point_prompts,
        point_prompt_camera=args.point_prompt_camera,
        conditioning_frame=conditioning_frame,
        cameras_to_warp=args.cameras_to_warp,
        max_points_in_conditioning_frame=args.max_points_in_conditioning_frame,
        use_point_prompts_per_dataset=[bool(x) for x in args.use_point_prompts_per_dataset] if args.use_point_prompts_per_dataset else None,
        require_val=not has_separate_val_paths and not args.no_val,
        state_spec=state_spec,
        state_indices=state_indices,
        max_exo_views=args.max_exo_views,
    )

    # Register separate val datasets if --val_data_paths is provided
    val_dataset_names = ["synthmanip/task_0"]  # default: val from first training path
    if has_separate_val_paths:
        val_dataset_names = []
        for i, val_path in enumerate(args.val_data_paths):
            validate_data_path(val_path, require_val=True)
            val_config = SynthmanipDatasetConfig(
                data_path=os.path.abspath(val_path),
                camera_names=args.camera_names,
                action_move_group_names=args.action_move_groups,
                action_spec=action_spec,
                action_keys=action_keys,
                state_spec=state_spec,
                state_indices=state_indices,
                action_horizon=args.action_horizon,
                input_window_size=args.n_obs_steps,
                obs_step_delta=args.obs_step_delta,
                robot_processor_config=proc_cfg,
                conditioning_frame=conditioning_frame,
                cameras_to_warp=args.cameras_to_warp or [],
                use_point_prompts=args.use_point_prompts,
                point_prompt_camera=args.point_prompt_camera,
                max_points_in_conditioning_frame=args.max_points_in_conditioning_frame,
                max_exo_views=args.max_exo_views,
            )
            name = f"synthmanip/val_{i}"
            synthmanip_config_registry.register(name, val_config)
            val_dataset_names.append(name)
            print(f"Registered val dataset '{name}' for: {val_path}")

    # Build validation evaluators
    val_evaluators = []
    if not args.no_val and not args.debug:
      for vi, val_ds_name in enumerate(val_dataset_names):
        val_data_config = DataLoaderConfig(
            dataset=val_ds_name,
            split="val",
            shuffle=False,
            drop_last=False,
            sequence_length=args.seq_len,
            max_text_seq_len=None,
            num_workers=max(2, args.num_workers // 2),
            pad="to_max",
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            seed=691203,
        )

        val_label = f"synthmanip_val_{vi}" if len(val_dataset_names) > 1 else "synthmanip_val"
        val_evaluator = LossDatasetEvaluatorConfig(
            label=val_label,
            data=val_data_config,
            device_batch_size=args.device_batch_size,
            max_examples=args.val_max_examples,
            console_log_interval=10,
            response_logits_only=True,
        )
        val_evaluators.append(val_evaluator)

    log_interval = 1 if args.debug else args.log_interval
    cfg = TrainConfig(
        run_name=args.exp_name,
        save_folder=omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None
        if args.debug
        else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval,
            allow_resume=True,
            finish_on_sigterm=True,
        ),
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=True,
        data=DataLoaderConfig(
            mixture=action_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=args.seq_len,
            max_text_seq_len=None,
            num_workers=args.num_workers,
            timeout=0 if args.num_workers == 0 else 300,
            pad="to_max",
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            seed=50189,
            packing=None,
        ),
        action_data=None,  # Single loader for action data
        action_loader_rate=None,
        ft_llm=False,
        ft_vit=False,
        ft_connector=False,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
            action_expert_learning_rate=1e-4,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            action_expert_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            action_expert_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            action_expert_eps=1e-6,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=2000,
            action_expert_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0,
        ),
        fsdp=FSDPConfig(fsdp2=True),
        load_path=None,
        reset_optimizer_state=args.reset_optimizer,
        initial_model_checkpoint=checkpoint,
        save_interval=2000,
        save_num_checkpoints_to_keep=1,
        checkpoint_retention_frequency=10000,
        global_train_batch_size=get_world_size() if args.debug else args.global_batch_size,
        device_train_microbatch_size=args.device_batch_size,
        time_limit=None,
        max_duration=200000,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        inf_evaluators=[],
        evaluators=val_evaluators,
        inf_eval_interval=args.val_interval,
        eval_interval=args.val_interval if val_evaluators else 0,
        save_final_optim=False,
        response_logits_only=True,
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    conf = OmegaConf.to_object(conf)
    run_trainer(conf)


if __name__ == "__main__":
    main()
