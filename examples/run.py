import os
import time
import yaml
import subprocess
from argparse import ArgumentParser
from roomexplorer import generate_dataset


def get_git_hash():
    try:
        binary_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        hash_ = binary_hash.decode("utf-8")
    except Exception:
        hash_ = "no git commit"
    return hash_


def append_time(d):
    base, ext = os.path.splitext(d)
    t = time.strftime("%Y-%m-%d--%H-%M-%S",
                      time.localtime())
    base = "-".join((base, t))
    new_d = base + ext
    print("experiment name updated to: {}".format(new_d))
    return new_d


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="path to config file",
                        default="config/config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    config["dataset"]["git_commit"] = get_git_hash()

    if os.path.splitext(config["dataset"]["save_directory"])[1] == "":
        config["dataset"]["save_directory"] += ".hdf5"
    if os.path.exists(config["dataset"]["save_directory"]):
        config["dataset"]["save_directory"] = append_time(
            config["dataset"]["save_directory"])

    generate_dataset(config)
