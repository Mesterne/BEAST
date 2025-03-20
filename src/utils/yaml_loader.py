import os

import yaml


def get_project_root():
    # Return the absolute path to the project root (one above 'models')
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def read_yaml(filename):
    with open(f"{filename}", "r") as f:
        output = yaml.safe_load(f)

        output["dataset_args"]["directory"] = os.path.join(
            get_project_root(), output["dataset_args"]["directory"]
        )
        validate_config(output)
    return output


def validate_config(config):
    model_name = None
    try:
        model_name = config["model_args"]["feature_model_args"]["model_name"]
    except:
        model_name = "undefined_model"

    match model_name:
        case "correlation_model":
            assert (
                config["dataset_args"]["use_one_hot_encoding"] == True
            ), "The correlation model does not support setting use_one_hot_encoding to True"
