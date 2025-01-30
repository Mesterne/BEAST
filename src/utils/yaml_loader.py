import yaml


def read_yaml(filename):
    with open(f"{filename}", "r") as f:
        output = yaml.safe_load(f)
    return output


config = read_yaml("experiments/gridloss/feedforward.yml")
print(config["dataset_args"])
