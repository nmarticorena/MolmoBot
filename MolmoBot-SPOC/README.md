![MolmoBot-SPOC Logo](docs/images/molmobot_spoc.png)

# Overview

MolmoBot-SPOC is a lightweight, goal-conditioned transformer policy for robot manipulation. It takes RGB observations, a natural-language task description, and optional image points as input and outputs continuous joint positions for execution on a robot platform.


# Installation

Set up a conda environment with Python 3.11:

```bash
conda create -n molmobot python=3.11
conda activate molmobot
```

Clone the repo and install from `MolmoBot-SPOC`:

```bash
git clone git@github.com:allenai/MolmoBot.git
cd MolmoBot/MolmoBot-SPOC
pip install -e .
```

# Training

Training uses PyTorch Lightning and a config registry. 

To use MolmoBot-Data for training experiments, follow the instructions in the top level README to download and postprocess the data.


To train the default RBY1 rigid manipulation model:

```bash
python -m molmobot_spoc.training.train_spoc_pl SPOCRBY1RigidManipTrainingConfig
```

and to train the default Franka Droid pick-and-place model:

```bash
python -m molmobot_spoc.training.train_spoc_pl SPOCDroidPickPlaceTrainingConfig
```


To train a custom model:
1. Define a matching `SpocGoalCondLlamaModelConfig` (model architecture + action space).
2. Define a new `TrainingConfig` subclass and register it.
3. Point your `TrainingConfig` at your dataset directory (HDF5 trajectories) and `SpocGoalCondLlamaModelConfig`.

The dataset loader (`training/dataset.py`) reads HDF5 trajectory files and supports phase-aware upsampling to rebalance underrepresented task phases.

Training checkpoints and metrics are logged to [Weights & Biases](https://wandb.ai).

# Sim Eval

Run evaluation against a JSON benchmark directory in MuJoCo simulation:

```bash
python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir /path/to/benchmarks/pick_minibenchmark
```

Swap `RBY1RigidManipEvalConfig` for `RBY1ArticulatedManipEvalConfig` to evaluate door-opening tasks.
Each eval config specifies:
- **Robot config**: RBY1M with joint-relative position control for arms, holonomic base, and height-controlled torso.
- **Camera config**: GoPro + D455 camera system.
- **Policy config**: W&B run ID and checkpoint step to load from as well as inference specific configuration.

```bash
python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.franka_eval_config:DroidPickPlaceMultitaskEvalConfig \
  --benchmark_dir /path/to/benchmarks/pick_minibenchmark
```

Each eval config specifies:
- **Robot config**: Franka Panda arm with joint‑position control, fixed base; end‑effector gripper.
- **Camera config**: Exocentric overhead (exo_camera_1) + wrist‑mounted (wrist_camera) RGB views.

**Warning**: The Franka Droid pick‑and‑place policy is trained with a fixed shoulder camera; camera pose randomization is not supported.

# Real Eval

Start the WebSocket policy server by specifying an eval config and optionally a task type:

```bash
python -m molmobot_spoc.eval.spoc_server \
  --config RBY1ArticulatedManipEvalConfig \
  --port 8000
```

Available `--config` options:
- `RBY1ArticulatedManipEvalConfig` — RBY1 door-opening (articulated manipulation), task type `open`
- `RBY1RigidManipEvalConfig` — RBY1 pick tasks (rigid manipulation), task type `pick`
- `DroidPickPlaceMultitaskEvalConfig` - Franka droid pick-and-place tasks

Use `--task_type` to override the task type string embedded in the config:

```bash
python -m molmobot_spoc.eval.spoc_server \
  --config RBY1RigidManipEvalConfig \
  --task_type pick_and_place \
  --port 8000
```

Once the server is running, a robot client can connect via WebSocket and exchange observations/actions using the msgpack-encoded protocol. The server logs the address it is listening on and emits per-step timing breakdowns (preprocess, infer, postprocess).
