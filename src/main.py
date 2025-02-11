import os
import sys
import argparse

# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.yaml_loader import read_yaml

# Load the configuration file
config = read_yaml(args["config_path"])

print(config["dataset_args"]["timeseries_to_use"])
