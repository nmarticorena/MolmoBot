![MolmoBot-Pi0 Logo](assets/MolmoBot-Pi0.png)

# Installation

Clone this repo and install the MolmoBot-Pi0 package:

```bash
git clone https://github.com/allenai/MolmoBot.git
cd MolmoBot/MolmoBot-Pi0
uv sync --extra eval
uv run scripts/install_openpi_extras.py
```

# Using MolmoBot-Pi0

This is a short example on how to load and run inference with MolmoBot-Pi0-DROID.
Note that the official policy configuration used in our evals is fully detailed [here](config/eval/policy/paligemma.yaml).

```python
import numpy as np
from huggingface_hub import snapshot_download
from molmobot_pi0.eval.policies.pi import PiJointPosPolicy

ckpt_dir = snapshot_download("allenai/MolmoBot-Pi0-DROID")
policy = PiJointPosPolicy(checkpoint_dir=ckpt_dir)
policy.prepare_model()

obs = {
    "task": "put the mug in the bowl",
    "qpos": {
        "arm": np.zeros(7),
        "gripper": np.zeros(2),
    },
    "exo_camera_1": np.zeros((360, 640, 3), dtype=np.uint8),
    "wrist_camera": np.zeros((360, 640, 3), dtype=np.uint8),
}

action = policy.get_action(obs)
print(action)
```

# Serving a policy for real evaluation

To run our official MolmoBot-Pi0-DROID checkpoint, simply run:

```bash
uv run molmobot_pi0/eval/real/serve.py
```

The `model_dir=` and `checkpoints=[]` arguments can be used to serve other models, including from remote sources such as S3 or Huggingface.

## Running official Pi models

This codebase can also be used to run `pi0_droid` and `pi05_droid`. To do so, run:

```bash
uv run molmobot_pi0/eval/real/serve.py -cn pi model_name=pi05_droid  # or pi0_droid
```

# Running simulation evals

To run simulation evals, use:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu  # optional, used to reduce MolmoSpaces VRAM usage
python molmobot_pi0/eval/eval.py \
    benchmark=<benchmark_path> \
    policy=paligemma \
    policy.model_dir=<model_dir> \
    policy.checkpoints=<checkpoints> \
    policy/cameras=sim_exo_shoulder \
    filament.enabled=<use_filament> \
    filament.light_intensity=12000
```

`<benchmark_path>` is the path to the MolmoSpaces benchmark to evaluate on. Furthermore, `filament.enabled` should be set to `true` or `false` depending on if the optional filement renderer is installed, and `filament.light_intensity` controls the global light intensity.

To evaluate the official MolmoBot-Pi0-DROID model, set `policy.model_dir=hf://allenai/MolmoBot-Pi0-DROID` and `policy.checkpoints=null`.

For further information, see the [MolmoSpaces](https://github.com/allenai/molmospaces) documentation.

## Using the filament renderer

To use the improved filament renderer for simulation evaluation, install it before running eval:

```bash
uv pip uninstall molmo-spaces
uv pip install "git+https://github.com/allenai/mujoco-thor.git@c54ed49ac64b612522f85272766bb5c54461ba16#egg=molmo-spaces[mujoco-filament]"
```

# Training a model

MolmoBot-Data can be downloaded from Huggingface with [this script](https://huggingface.co/datasets/allenai/molmobot-data/blob/main/bulk_download.py).

To download the dataset used to train MolmoBot-Pi0-DROID:

```bash
python bulk_download.py --config FrankaPickAndPlaceOmniCamConfig --part 1 <data_root>
python bulk_download.py --config FrankaPickOmniCamConfig --part 0 <data_root>
python bulk_download.py --config FrankaPickAndPlaceOmniCamConfig --part 0 <data_root>
python bulk_download.py --config FrankaPickAndPlaceNextToOmniCamConfig --part 2 <data_root>
python bulk_download.py --config FrankaPickAndPlaceColorOmniCamConfig --part 0 <data_root>
```

The following command can be used to replicate the training process for MolmoBot-Pi0-DROID.

```bash
python molmobot_pi0/train_openpi_pytorch.py molmobot_pi0_droid \
    --data.repo-id <data_root>/FrankaPickAndPlaceOmniCamConfig/part1/train \
    --data.exo-camera randomized_zed2_analogue_1 \
    --data.wrist-camera wrist_camera_zed_mini \
    --data.auxiliary-dataset-paths <data_root>/FrankaPickOmniCamConfig/part0/train 0.2 \
    --data.auxiliary-dataset-paths <data_root>/FrankaPickAndPlaceOmniCamConfig/part0/train 0.1 \
    --data.auxiliary-dataset-paths <data_root>/FrankaPickAndPlaceNextToOmniCamConfig/part2/train 0.2 \
    --data.auxiliary-dataset-paths <data_root>/FrankaPickAndPlaceColorOmniCamConfig/part0/train 0.15 \
    --exp-name train_molmobot_pi0_droid \
    --checkpoint-base-dir train_output \
    --project-name molmobot_pi0 \
    --batch-size 1024 \
    --num-workers 48
```

Note that the above training run was used with `torchrun` to train on 32 H100s, so you will need to scale as necessary, and set up multinode as needed.
