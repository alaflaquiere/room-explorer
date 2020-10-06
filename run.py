import os
import time
import yaml
import numpy as np
from argparse import ArgumentParser
import subprocess

from environment.RoomEnvironment import Room
from agent.Agent import MobileArm


def get_git_hash():
    try:
        binary_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        hash_ = binary_hash.decode("utf-8")
    except Exception:
        hash_ = "no git commit"
    return hash_


def append_time(d):
    t = time.strftime("%Y-%m-%d--%H-%M-%S",
                      time.localtime())
    d = "-".join((d, t))
    print("experiment name updated to: {}".format(d))
    return d


def explore_and_save(agent, room, mode, k, directory):
    assert mode in ["dynamic_base", "static_base", "hopping_base"]
    motors_t, shifts_t, states_t, motors_tp, shifts_tp, states_tp = \
        agent.generate_random_transitions(mode, k)
    sensors_t = room.get_sensations(states_t)
    sensors_tp = room.get_sensations(states_tp)
    # save the data
    transitions = {"motors_t": motors_t,
                   "sensors_t": sensors_t,
                   "motors_tp": motors_tp,
                   "sensors_tp": sensors_tp}
    np.savez_compressed(os.path.join(directory, "data_{}.npz".format(mode)),
                        **transitions)


def save_regular_grid(agent, resolution, directory):
    motor_grid, state_grid = agent.generate_regular_states(resolution)
    grid = {"motor_grid": motor_grid,
            "state_grid": state_grid}
    np.savez_compressed(os.path.join(directory, "data_regular_grid.npz"),
                        **grid)


def generate_data(conf):
    """TODO"""

    os.makedirs(conf["save_directory"])

    with open(os.path.join(conf["save_directory"], "config.yml"), "w") as f:
        yaml.dump(conf, f)

    # iterate over the runs
    for r in range(conf["n_runs"]):

        # subdirectory for the run
        sub_dir = os.path.join(conf["save_directory"],
                               "run{:03}".format(r))
        os.makedirs(sub_dir)
        print("run {} >> {}".format(r, sub_dir))

        # create the agent
        agent = MobileArm(**conf["agent"])
        agent.save(sub_dir)

        # create an environment
        room = Room(resolution=16)
        room.save(sub_dir)

        # save the regular exploration of the motor space
        save_regular_grid(agent, 7, sub_dir)

        # generate_data with dynamic base
        explore_and_save(agent, room, "dynamic_base",
                         conf["n_transitions"],
                         sub_dir)
        # generate_data with static base
        explore_and_save(agent, room, "static_base",
                         conf["n_transitions"],
                         sub_dir)
        # generate_data with static base
        explore_and_save(agent, room, "hopping_base",
                         conf["n_transitions"],
                         sub_dir)
        # clean
        room.destroy()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="path to config file",
                        default="config\\config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    config["git_commit"] = get_git_hash()
    config["save_directory"] = append_time(config["save_directory"])

    generate_data(config)
