import yaml

import os


def get_project_root():
    # Return the absolute path to the project root (one above 'models')
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def read_yaml(filename):
    with open(f"{filename}", "r") as f:
        output = yaml.safe_load(f)

        output["dataset_args"]["directory"] = os.path.join(
            get_project_root(), output["dataset_args"]["directory"]
        )
    return output
