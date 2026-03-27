import os
import sys
import logging

os.environ["JAX_ENABLE_X64"] = "0"  # 64-bit jax messes with openpi inference

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.evaluation.policy_server import WebsocketPolicyServer

from molmobot_pi0.eval.real.utils import maybe_download


logging.basicConfig(level=logging.INFO)


def load_pi(policy_cfg: DictConfig, device_id: int = 0) -> InferencePolicy:
    # disable cudagraphs to avoid threading issues
    torch._inductor.config.triton.cudagraphs = False

    from molmobot_pi0.eval.policies.pi import PiJointVelPolicy, PiJointPosPolicy

    if policy_cfg.model_dir is not None and policy_cfg.checkpoints is not None:
        assert len(policy_cfg.checkpoints) == 1
        with open_dict(policy_cfg):
            policy_cfg.checkpoint = policy_cfg.checkpoints[0]

        model_url: str = policy_cfg.model_dir
        ckpt_url = model_url.rstrip("/") + "/" + str(policy_cfg.checkpoints[0])
        ckpt_dir = maybe_download(ckpt_url, exclude="optimizer.pt")
    elif policy_cfg.model_dir is not None:
        ckpt_dir = maybe_download(policy_cfg.model_dir)
    else:
        ckpt_dir = None

    init_kwargs = dict(
        model_name=policy_cfg.model_name,
        checkpoint_dir=ckpt_dir,
        use_torch=policy_cfg.use_torch,
        buffer_length=policy_cfg.buffer_length,
        cameras=policy_cfg.cameras,
        device_id=device_id,
        compile_mode=policy_cfg.compile_mode,
    )

    if policy_cfg.action_type == "jointvel":
        policy = PiJointVelPolicy(**init_kwargs)
    elif policy_cfg.action_type == "jointpos":
        policy = PiJointPosPolicy(**init_kwargs)
    else:
        raise ValueError(f"Invalid action type: {policy_cfg.action_type}")

    return policy, policy.model_name


@hydra.main(version_base=None, config_path="../../../config/eval/policy", config_name="paligemma")
def main(policy_cfg: DictConfig):
    if policy_cfg == {}:
        print("Reading policy config from stdin")
        policy_cfg_str = sys.stdin.read()
        policy_cfg = OmegaConf.create(policy_cfg_str)

    if missing_keys := OmegaConf.missing_keys(policy_cfg):
        raise ValueError(f"Missing keys: {missing_keys}")

    print(OmegaConf.to_yaml(policy_cfg))

    num_policies_per_gpu = policy_cfg.get("num_policies_per_gpu", 1)
    print(f"Using {num_policies_per_gpu} policies per GPU")

    policies: list[InferencePolicy] = []
    model_names: list[str] = []
    for _ in range(num_policies_per_gpu):
        # have gpu in the inner loop to improve balancing between GPUs
        for device_id in range(torch.cuda.device_count()):
            if policy_cfg.type == "pi":
                policy, model_name = load_pi(policy_cfg, device_id)
                policies.append(policy)
                model_names.append(model_name)
            else:
                raise ValueError(f"Invalid policy type: {policy_cfg.type}")

    print(f"Serving {len(policies)} policies on {torch.cuda.device_count()} GPUs")

    assert len(policies) == len(model_names)
    assert all(mn == model_names[0] for mn in model_names)

    port = int(os.getenv("POLICY_SERVER_PORT", "8000"))

    server = WebsocketPolicyServer(policies, model_names[0], port=port, metadata=OmegaConf.to_container(policy_cfg))
    server.serve_forever()


if __name__ == "__main__":
    main()
