import argparse
import math
import os

import torch
import yaml


def setup_config_dict(
    config_dict: dict[str, dict[str, any]], args: argparse.Namespace, resume=False
) -> dict[str, dict[str, any]]:
    """
    Clean the configuration file, adjust based on environmental variables and overwritten options.
    """
    if hasattr(args, "skip_validation"):
        config_dict["skip_validation"] = args.skip_validation
    # --resume wins: a saved run's config.yaml always carries resume_path (null
    # unless that run was itself resumed).
    config_dict["resume_path"] = getattr(args, "resume", None) or config_dict.get(
        "resume_path"
    )

    config_dict["training"]["log_interval"] = config_dict["training"].get(
        "log_interval", 50
    )
    config_dict["training"]["dsync_interval"] = config_dict["training"].get(
        "dsync_interval", 200
    )

    # EWB config
    config_dict["data"]["ewb_events_path"] = config_dict["data"].get(
        "ewb_events_path", "data/ewb_events"
    )
    config_dict["data"]["ewb_case_ids"] = config_dict["data"].get("ewb_case_ids", [])
    config_dict["data"]["ewb_event_types"] = config_dict["data"].get(
        "ewb_event_types", []
    )

    # Handle mesh dims
    config_dict = overwrite_config(config_dict, args.configargs)

    world_size = int(os.getenv("SLURM_NTASKS", 1))
    mesh_dims = config_dict["mesh_dims"]
    if -1 in config_dict["mesh_dims"]:
        index = mesh_dims.index(-1)
        config_dict["mesh_dims"][index] = (
            world_size // -torch.prod(torch.tensor(mesh_dims)).item()
        )

    if not resume:
        ### Data Stuff ###
        WORKDIR = os.getenv("WORKDIR")

        if os.getenv("DATA_DIR") is not None:
            config_dict["data"]["data_path"] = (
                os.getenv("DATA_DIR") + config_dict["data"]["data_path"]
            )
            config_dict["data"]["constant_masks_path"] = (
                os.getenv("DATA_DIR") + config_dict["data"]["constant_masks_path"]
            )

        if "normalization_dir" in config_dict["data"]:
            config_dict["data"]["normalization_dir"] = (
                WORKDIR + "/" + config_dict["data"]["normalization_dir"]
            )

        # default case: load only one timestep
        if config_dict["data"]["n_in_timesteps"] not in [1, 2]:
            config_dict["data"]["n_in_timesteps"] = 1

        # Only for beast: append n_masks for odd number of masks (the dataset does this, too)
        n_masks = len(config_dict["data"]["constant_masks"])
        if math.prod(config_dict["mesh_dims"][-2:]) > 1 and n_masks % 2 == 1:
            n_masks += 1

        n_input_vars = (
            config_dict["data"]["n_in_timesteps"] * config_dict["data"]["n_variables"]
        )

        # we'll set this automatically, to avoid confusion when defining the config,
        # as n_input_channels depends on how many timesteps we use as concatenated input
        config_dict["model"]["in_channels"] = n_input_vars + n_masks
        config_dict["model"]["out_channels"] = (
            config_dict["data"]["n_variables"] * config_dict["data"]["n_out_timesteps"]
        )

    return config_dict


def overwrite_config(
    config_dict: dict[str, dict[str, any]], configargs: list[str], resume=False
) -> dict[str, dict[str, any]]:
    """
    Change configuration file values based on cmd line arguments passed as '<section>_<key>=<value>' after all the optional flags. Arrays are passed with the elements connected by comas.
    """
    for i in range(len(configargs)):
        section_key, new_value = configargs[i].split("=")

        # Special case for mesh dim, as it is not part of a section (though it should be parallelism)
        if section_key == "mesh_dims":
            config_dict["mesh_dims"] = [int(x) for x in new_value.split(",")]
            continue

        section, key = section_key.split("_", maxsplit=1)
        if section not in config_dict:
            print(
                f"Section for keyword argument {configargs[i]} does not exist in the configuration file"
            )
            continue
        elif key not in config_dict[section] and section == "model":
            if section == "model":
                print(f"Overriding unknown configuration option {key} from model.")
                if "," in new_value:
                    formated_new_value = [int(x) for x in new_value.split(",")]
                else:
                    formated_new_value = int(new_value)
                config_dict[section][key] = formated_new_value
                continue
            else:
                print(
                    f"Configuration option {key} does not exist in the configuration file"
                )
                continue

        original_value = config_dict[section][key]
        if isinstance(original_value, str):
            config_dict[section][key] = new_value
        elif isinstance(original_value, bool):
            config_dict[section][key] = new_value == "True"
        elif isinstance(original_value, int):
            config_dict[section][key] = int(new_value)
        elif isinstance(original_value, list):
            new_values = new_value.split(",")
            try:
                formated_new_value = [int(x) for x in new_values]
            except Exception:
                formated_new_value = [x for x in new_values]

            config_dict[section][key] = formated_new_value

    return config_dict


def load_configuration_yaml(directory):
    yaml_path = os.path.join(directory, "config.yaml")
    with open(f"{yaml_path}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return config_dict
