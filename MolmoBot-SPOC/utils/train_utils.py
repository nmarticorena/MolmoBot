import torch


def load_ckpt(model, ckpt_pth, verbose=False):
    """Load a checkpoint from either a .ckpt (PyTorch Lightning) or .safetensors file."""
    print(f"Loading ckpt {ckpt_pth} ...")
    if str(ckpt_pth).endswith(".safetensors"):
        from safetensors.torch import load_file

        ckpt_state_dict = load_file(ckpt_pth, device="cpu")
        ckpt_prefix = ""
    else:
        ckpt_state_dict = torch.load(ckpt_pth, map_location="cpu")["state_dict"]
        ckpt_prefix = "model."

    new_state_dict = model.state_dict()

    params_to_load = [
        k for k in new_state_dict.keys() if ckpt_prefix + k in ckpt_state_dict
    ]
    for k in params_to_load:
        new_state_dict[k] = ckpt_state_dict[ckpt_prefix + k]

    model.load_state_dict(new_state_dict)
    if verbose:
        print("-" * 80)
        print("Parameters found and loaded from the checkpoint:", params_to_load)
        print("-" * 80)

    params_in_model_not_in_ckpt = [
        k for k in new_state_dict.keys() if ckpt_prefix + k not in ckpt_state_dict
    ]
    if params_in_model_not_in_ckpt:
        print("-" * 80)
        print("Parameters in model but not in checkpoint:", params_in_model_not_in_ckpt)
        print("-" * 80)

    params_in_ckpt_not_in_model = [
        k[len(ckpt_prefix) :]
        for k in ckpt_state_dict
        if k[len(ckpt_prefix) :] not in new_state_dict
    ]
    if params_in_ckpt_not_in_model:
        print("-" * 80)
        print("Parameters in checkpoint but not in model:", params_in_ckpt_not_in_model)
        print("-" * 80)


def load_pl_ckpt(model, ckpt_pth, ckpt_prefix="model.", verbose=False):
    print(f"Loading ckpt {ckpt_pth} using ckpt_prefix='{ckpt_prefix}' ...")
    ckpt_state_dict = torch.load(ckpt_pth, map_location="cpu")["state_dict"]

    # init new state dict with curr state dict
    new_state_dict = model.state_dict()

    params_to_load = [
        k for k in new_state_dict.keys() if ckpt_prefix + k in ckpt_state_dict
    ]
    for k in params_to_load:
        new_state_dict[k] = ckpt_state_dict[ckpt_prefix + k]

    model.load_state_dict(new_state_dict)
    if verbose:
        print("-" * 80)
        print(
            "Parameters found and loaded from the checkpoint:",
            params_to_load,
        )
        print("-" * 80)

    params_in_model_not_in_ckpt = [
        k for k in new_state_dict.keys() if ckpt_prefix + k not in ckpt_state_dict
    ]
    if len(params_in_model_not_in_ckpt) > 0:
        print("-" * 80)
        print(
            "Parameters that are present in model but not present in checkpoint:",
            params_in_model_not_in_ckpt,
        )
        print("-" * 80)

    params_in_ckpt_not_in_model = [
        k[len(ckpt_prefix) :]
        for k in ckpt_state_dict
        if k[len(ckpt_prefix) :] not in new_state_dict
    ]
    if len(params_in_ckpt_not_in_model):
        print("-" * 80)
        print(
            f"Parameters present in checkpoint not present in model (removing ckpt_prefix='{ckpt_prefix}'):",
            params_in_ckpt_not_in_model,
        )
        print("-" * 80)
