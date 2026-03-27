"""Run this script with 'torchrun'."""

import logging
import os
import re
import signal
import socket
import sys
import time
from datetime import datetime
from os.path import join
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import wandb
from beaker import Beaker
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from wandb.sdk.wandb_run import Run

from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.io import file_exists, write_file, resource_path
from olmo.train.checkpointer import (
    Checkpointer,
    is_unsharded_checkoint,
    MODEL_FILENAME,
)
from olmo.torch_util import (
    barrier,
    get_global_rank,
    get_local_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    freeze_module,
)
from olmo.dist_util import (
    parallelize_model,
    build_world_mesh
)
from olmo.train.remote_filesystem import RemoteFileSystemReader
from olmo.train.trainer import Trainer, BeakerLogger
from olmo.train.trainer_config import TrainConfig, RuntimeData
from olmo.util import (
    clean_opt,
    log_extra_field,
    prepare_torchrun_environment,
)

log = logging.getLogger("train")


def _generate_action_expert_init_state(model: torch.nn.Module) -> dict:
    """Create properly initialized action expert weights on CPU (rank 0 only).

    This avoids calling ``nn.init`` directly on FSDP2 DTensor parameters which
    can deadlock.  Instead we build a temporary, non-sharded ``ActionExpert`` on
    CPU, run :meth:`reset_parameters`, and return its state dict prefixed with
    ``"action_expert."``.
    """
    from olmo.nn.action_expert import ActionExpert

    ae = model.action_expert
    config = ae.config
    llm_dim = ae.context_proj.in_features
    temp_expert = ActionExpert(config, llm_dim=llm_dim, device="cpu")
    temp_expert.reset_parameters()
    return {"action_expert." + k: v for k, v in temp_expert.state_dict().items()}


def _load_unsharded_checkpoint_allowing_missing_action_expert(path: str, model: torch.nn.Module) -> bool:
    """
    Load an unsharded checkpoint into ``model`` while allowing the checkpoint to omit
    the action_expert weights (e.g., when initializing from a VLM-only checkpoint).

    When action_expert weights are missing, properly initialized values are
    generated on rank 0 and injected into the state dict so they are loaded via
    the FSDP-safe ``set_model_state_dict`` path (avoiding ``nn.init`` calls on
    sharded DTensors which can hang).

    Returns:
        bool: True if the checkpoint included action_expert weights, False otherwise.
    """
    if get_global_rank() == 0:
        state_dict = torch.load(
            resource_path(path, MODEL_FILENAME),
            map_location="cpu",
            weights_only=True,
        )
        has_action_expert = any(key.startswith("action_expert.") for key in state_dict.keys())
        if not has_action_expert:
            filtered_state = {
                key: value for key, value in state_dict.items() if not key.startswith("action_expert.")
            }
            if "_metadata" in state_dict:
                filtered_state["_metadata"] = {
                    k: v for k, v in state_dict["_metadata"].items() if not k.startswith("action_expert.")
                }
            # Generate init values for the missing action expert so we can
            # load everything through set_model_state_dict (FSDP-safe).
            if hasattr(model, "action_expert") and model.action_expert is not None:
                log.info("Checkpoint lacks action expert weights; generating init values on rank 0.")
                init_state = _generate_action_expert_init_state(model)
                filtered_state.update(init_state)
        else:
            filtered_state = state_dict
    else:
        filtered_state = {}
        has_action_expert = False

    if torch.cuda.is_available():
        broadcast_device = torch.device("cuda", torch.cuda.current_device())
    else:
        broadcast_device = torch.device("cpu")
    flag = torch.tensor(int(has_action_expert), device=broadcast_device)
    dist.broadcast(flag, src=0)
    has_action_expert = bool(flag.item())

    dist_cp_sd.set_model_state_dict(
        model=model,
        model_state_dict=filtered_state,
        options=dist_cp_sd.StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=has_action_expert,
        ),
    )

    if get_global_rank() == 0:
        del state_dict
    del filtered_state
    return has_action_expert


def run_trainer(cfg: TrainConfig) -> None:
    if cfg.run_name is None:
        log_extra_field("run_name", cfg.run_name)

    # Additional environment setup
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    seed_all(cfg.seed)
    barrier()

    # Display the configuration.
    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)

    # Figure out what checkpoint we are starting from, if any
    start_from = None
    reset_opt, reset_train = False, False
    is_resuming = False
    if cfg.allow_resume:
        # Check if there is a checkpoint for us to resume from in our save folder, in which
        # case we ignore `cfg.load_from` and use it
        try:
            lastest_checkpoint = Checkpointer.latest_checkpoint(cfg.save_folder)
        except FileNotFoundError:
            lastest_checkpoint = None
        if lastest_checkpoint:
            print(f"Resuming from {lastest_checkpoint}", flush=True)
            log.info(f"Resuming from {lastest_checkpoint}")
            if get_global_rank() == 0:
                saved_config: TrainConfig = TrainConfig.load(join(cfg.save_folder, "config.yaml"))
                if saved_config.model != cfg.model:
                    log.warning("Model config does not match the one resuming from")
                    import dataclasses as _dc
                    for _f in _dc.fields(cfg.model):
                        _sv = getattr(saved_config.model, _f.name, None)
                        _cv = getattr(cfg.model, _f.name, None)
                        if _sv != _cv:
                            print(f"  MODEL DIFF [{_f.name}]: saved={_sv!r}  current={_cv!r}", flush=True)
                if saved_config.optimizer != cfg.optimizer:
                    log.warning("Optimizer config does not match the one resuming from")
                if saved_config.data != cfg.data:
                    log.warning("Data config does not match the one resuming from")
            start_from = str(lastest_checkpoint)
            reset_opt = cfg.reset_optimizer_state
            reset_train = cfg.reset_trainer_state
            is_resuming = True
        else:
            log.info("Not resuming since no latest checkpoint found")

    if start_from is None and cfg.load_path:
        start_from = cfg.load_path
        reset_train, reset_opt = cfg.reset_trainer_state, cfg.reset_optimizer_state
    elif start_from is None and cfg.initial_model_checkpoint is not None:
        start_from = cfg.initial_model_checkpoint
        reset_train, reset_opt = True, True
    start_from_unsharded = start_from and is_unsharded_checkoint(start_from)
    if start_from_unsharded:
        assert reset_opt and reset_train, "Unshared checkpoints do not support optim/train state loading"

    # Fail fast if we would be overwriting another save directory
    if not cfg.dry_run and not is_resuming and not cfg.save_overwrite:
        save_path = join(cfg.save_folder, "config.yaml")
        if file_exists(save_path):
            raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")

    barrier()

    # Init the model
    model_cfg = cfg.model
    with torch.device("meta"):
        olmo_model = model_cfg.build_model()

    # Freeze parameters depending on what we are tuning
    if not cfg.ft_connector:
        log.info(f"Freezing connector")
        for param in olmo_model.get_connector_parameters():
            param.requires_grad = False
    if not cfg.ft_vit:
        log.info(f"Freezing vision backbone")
        for param in olmo_model.get_vit_parameters():
            param.requires_grad = False
    if not cfg.ft_llm:
        log.info(f"Freezing LLM")
        for param in olmo_model.get_llm_parameters():
            param.requires_grad = False
    elif cfg.ft_embedding != "all":
        freeze_wte, freeze_out, freeze_ln_f = True, True, True
        if cfg.ft_embedding == "ln_f":
            freeze_ln_f = False
        elif cfg.ft_embedding == "lm_head":
            freeze_ln_f = False
            freeze_out = False
        elif cfg.ft_embedding == "wte":
            freeze_wte = False
        elif cfg.ft_embedding == "ae":
            # The action expert loss will drive the training
            pass
        else:
            raise NotImplementedError(cfg.fsdp)
        if freeze_ln_f:
            log.info(f"Freezing LLM: ln_f")
            freeze_module(olmo_model.transformer.ln_f)
        if freeze_out and hasattr(olmo_model.transformer, "ff_out"):
            log.info(f"Freezing LLM: ff_out")
            freeze_module(olmo_model.transformer.ff_out)
        if freeze_wte:
            log.info(f"Freezing LLM: wte")
            olmo_model.transformer.wte.embedding.requires_grad = False

    cp_enabled = cfg.parallelism.context_parallel_config.degree > 1

    # Do some other model setup
    if cfg.activation_checkpointing:
        olmo_model.apply_activation_checkpointing()
    # Stops the compiler get confused due to cache modifications
    olmo_model.warmup_cache(device, cp_enabled=cp_enabled)

    if cfg.compile:
        assert cfg.parallelism.context_parallel_config.degree == 1, "Model compilation is not supported with context parallelism yet."
        olmo_model.apply_compile(**cfg.compile.compile_args())

    world_mesh = None
    if not cp_enabled:
        # Shard the model, and initialize if we are not loading a checkpoint
        if cfg.fsdp and not cfg.fsdp.fsdp2:
            log.info("Wrapping model with FSDP...")
            if start_from is None:
                # Just run our `reset_with_pretrained_weights` on rank0 and broadcast so we
                # don't have to port all the init logic to a FSDP param_init_fn function
                if get_global_rank() == 0:
                    # Load on CPU in case model doesn't fit on a single GPU
                    olmo_model = olmo_model.to_empty(device="cpu")
                    olmo_model.reset_with_pretrained_weights()
                sync_module_states = True
            else:
                sync_module_states = False

            # meta-device parameters can just become empty since we are either broadcasting from rank0
            # or going to load a checkpoint anyway
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=device, recurse=False)

            fsdp_model = FSDP(
                olmo_model,
                **cfg.fsdp.get_fsd_args(cfg.autocast_precision),
                param_init_fn=dummy_init_fn,
                auto_wrap_policy=olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy),
                device_id=get_local_rank(),
                sync_module_states=sync_module_states,
            )

        elif cfg.fsdp.fsdp2:
            log.info("Wrapping model with FSDP2...")
            olmo_model.apply_fsdp2(**cfg.fsdp.get_fsd2_args(cfg.autocast_precision))
            olmo_model.to_empty(device=device)
            if start_from is None:
                olmo_model.reset_with_pretrained_weights()
            fsdp_model = olmo_model
        else:
            raise NotImplementedError()
    else:
        parallelism_config = cfg.parallelism

        world_mesh = build_world_mesh(
            tp=parallelism_config.tensor_parallel_config,
            cp=parallelism_config.context_parallel_config,
            dp=parallelism_config.data_parallel_config,
        )
        olmo_model = parallelize_model(
            olmo_model,
            world_mesh=world_mesh,
            float8_config=None,
            cp_config=parallelism_config.context_parallel_config,
            dp_config=parallelism_config.data_parallel_config,
            tp_config=parallelism_config.tensor_parallel_config,
        )
        olmo_model.to_empty(device=device)
        if start_from is None:
            olmo_model.reset_with_pretrained_weights()
        fsdp_model = olmo_model
 
    torch.cuda.empty_cache()

    log.info("Model:")
    log.info(fsdp_model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"VLM number of parameters: {olmo_model.num_params_vlm():,d}")
    log.info(f"Action expert number of parameters: {olmo_model.num_params_action_expert():,d}")
    if olmo_model.config.llm.block_type == "moe":
        log.info(f"Number of active parameters: {olmo_model.num_params(include_inactive_params=False):,d}")
    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")

    # Construct optimizer/scheduler/checkpointer
    optim = cfg.optimizer.build_optimizer(cfg.max_grad_norm, cfg.max_grad_norm_ratio, fsdp_model)
    scheduler = cfg.scheduler.build()
    checkpointer = cfg.checkpointer_config.build(cfg.save_overwrite)

    # Construct data loader and evaluators
    log.info(f"[rank {get_global_rank()}] Building train dataloader with {cfg.data.num_workers} workers...")
    train_loader = cfg.data.build_train_dataloader(
        model_config=cfg.model,
        mesh=world_mesh,
        global_batch_size=cfg.global_train_batch_size,
    )
    log.info(f"[rank {get_global_rank()}] Train dataloader built successfully")
    action_loader = None
    if cfg.action_data is not None:
        log.info("Building action-only dataloader...")
        action_loader = cfg.action_data.build_train_dataloader(
            model_config=cfg.model,
            mesh=world_mesh,
            global_batch_size=cfg.global_train_batch_size,
        )
    if cfg.eval_interval > 0 or cfg.eval_on_load:
        evaluators = [v.build_dataset_evaluator(
            model_config=cfg.model, 
            mesh=world_mesh,
            device=device) for v in cfg.evaluators]
    else:
        evaluators = None
    if cfg.inf_eval_interval > 0 or cfg.eval_on_load:
        inf_evaluators = [v.build_dataset_evaluator(
            model_config=cfg.model, 
            mesh=None,  # disable mesh for inference as it's not supported
            default_save_dir=None, 
            device=device) for v in cfg.inf_evaluators]
    else:
        inf_evaluators = None

    # Maybe build the BeakerLogger
    if "BEAKER_EXPERIMENT_ID" in os.environ and "BEAKER_TOKEN" in os.environ:
        if get_global_rank() == 0:
            experiment_id = os.environ["BEAKER_EXPERIMENT_ID"]
            client = Beaker.from_env()
            beaker_logger = BeakerLogger(client, experiment_id, cfg.beaker_log_interval)
            beaker_logger.log_init()
        else:
            beaker_logger = None
    else:
        if cfg.beaker_log_interval > 0 and "BEAKER_EXPERIMENT_ID" in os.environ:
            logging.info(f"Beaker log interval set to {cfg.beaker_log_interval}, but beaker "
                         f"token is missing, so beaker logging will turned off")
        beaker_logger = None

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_cfg = cfg.asdict(exclude=["wandb"])

        if "BEAKER_EXPERIMENT_ID" in os.environ:
            wandb_cfg["beaker_experiment_id"] = os.environ["BEAKER_EXPERIMENT_ID"]
            if beaker_logger is not None:
                wandb_cfg["beaker_url"] = beaker_logger.get_beaker_url()

        if is_resuming:
            wandb_cfg["resuming_from"] = start_from
        if is_resuming and cfg.wandb.allow_resume:
            resume_run_id = saved_config.runtime_data.wandb_id
            resume_step = int(re.match(r".*step([0-9]+).*", lastest_checkpoint).group(1))
            resume_from = f"{resume_run_id}?_step={resume_step}"
            run_id = resume_run_id
        else:
            run_id = None
            resume_from = None

        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=wandb_cfg,
            id=run_id,
            resume_from=resume_from,
            settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", 600)))
        )
        wandb_url = wandb.run.get_url() if wandb.run is not None else None
        if beaker_logger is not None and wandb_url is not None:
            beaker_logger.add_wandb(wandb_url)  # add wandb url to beaker description

        if cfg.wandb.finish_on_sigterm:
            # Try to make sure wandb will always finish cleanly if we get preempted
            # This is a bit of hack, but its useful since we can't use wandb.resume if wandb
            # did not finish cleanly

            def _signal_handler(signum, frame):
                # The dataloader workers might die and send us SIGCHLD which can interrupt this
                # method, so we ignore SIGCHLD here to avoid this
                # We are exiting anyway so its probably fine?
                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                log.warning(f"Getting {signum}, finish wandb then exiting")
                if wandb.run:
                    try:
                        wandb.finish(1)
                    except Exception as e:
                        log.warning(f"Unable to finish wandb {e}")
                exit(1)
            signal.signal(signal.SIGTERM, _signal_handler)

    # Fill in some runtime data so it will be recorded when we save the config
    cfg.runtime_data = RuntimeData(
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m/%d/%Y, %H:%M"),
        world_size=get_world_size(),
        beaker_experiment_id=os.environ.get("BEAKER_EXPERIMENT_ID"),
        beaker_experiment_url=(None if beaker_logger is None else
                               beaker_logger.get_beaker_url()),
        wandb_url=wandb.run.get_url() if wandb.run else None,
        wandb_id=wandb.run.id if wandb.run else None,
        args=" ".join(sys.argv),
        resuming_from=start_from if is_resuming else None,
    )

    # Save the config in a top-level file, note if we are resuming
    # the current config will still be saved next to new checkpoints
    if not cfg.dry_run and not is_resuming:
        if get_global_rank() == 0:
            write_file(cfg.save_folder, "config.yaml",
                       OmegaConf.to_yaml(cfg, resolve=True), cfg.save_overwrite)
    barrier()

    with Trainer(
        cfg=cfg,
        mesh=world_mesh,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=fsdp_model,
        checkpointer=checkpointer,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        action_loader=action_loader,
        device=device,
        evaluators=evaluators,
        inference_evaluators=inf_evaluators,
        beaker_logger=beaker_logger,
    ) as trainer:

        if start_from:
            # Load the starting checkpoint if there is one
            t0 = time.perf_counter()
            if start_from_unsharded:
                print(f"Loading unshared model from {start_from}", flush=True)
                log.info(f"Loading unshared model from {start_from}")
                has_action_expert = _load_unsharded_checkpoint_allowing_missing_action_expert(
                    start_from, fsdp_model
                )
                if (
                    not has_action_expert
                    and hasattr(fsdp_model, "action_expert")
                    and hasattr(fsdp_model.action_expert, "reset_parameters")
                ):
                    print("Checkpoint lacks action expert weights; reinitializing action expert.", flush=True)
                    log.info("Checkpoint lacks action expert weights; reinitializing action expert.")
                    fsdp_model.action_expert.reset_parameters()
            else:
                if reset_train and reset_opt:
                    print(f"Loading model from {start_from}", flush=True)
                    log.info(f"Loading model from {start_from}")
                elif not reset_opt and not reset_train:
                    print(f"Resuming from checkpoint {start_from}", flush=True)
                    log.info(f"Resuming from checkpoint {start_from}")
                else:
                    print(f"Restoring checkpoint {start_from}, but resetting "
                          f"{'Trainer' if reset_train else 'Optimizer'}", flush=True)
                    log.info(f"Restoring checkpoint {start_from}, but resetting "
                             f"{'Trainer' if reset_train else 'Optimizer'}")
                trainer.restore_checkpoint(
                    start_from,
                    load_optimizer_state=not reset_opt,
                    load_trainer_state=not reset_train,
                )
            print(f"Checkpoint successfully loaded in {time.perf_counter()-t0:0.1f} seconds", flush=True)
            log.info(f"Checkpoint successfully loaded in {time.perf_counter()-t0:0.1f} seconds")
            barrier()

        for name, param in fsdp_model.named_parameters():
            if not torch.all(torch.isfinite(param)):
                raise ValueError(name)

        # Ready to start training
        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    prepare_torchrun_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    run_trainer(cfg)
