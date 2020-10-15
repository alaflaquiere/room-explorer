import os
import sys
import time
import yaml
import numpy as np
import h5py
import subprocess
from argparse import ArgumentParser
from flatten_dict import flatten

sys.path.append(os.path.join(os.getcwd(), "..\\src"))
import RoomExplorer


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


def explore(agent, room, mode, k):
    assert mode in ["dynamic_base", "static_base", "hopping_base"]
    print("mode: {}".format(mode))
    motors_t, shifts_t, states_t, motors_tp, shifts_tp, states_tp = \
        agent.generate_random_transitions(mode, k)
    sensors_t = room.get_sensations(states_t)
    sensors_tp = room.get_sensations(states_tp)
    return motors_t, sensors_t, motors_tp, sensors_tp


def save_regular_grid(agent, resolution, directory):
    #todo: necessary?!
    motor_grid, state_grid = agent.generate_regular_states(resolution)
    grid = {"motor_grid": motor_grid,
            "state_grid": state_grid}
    np.savez_compressed(os.path.join(directory, "data_regular_grid.npz"),
                        **grid)


def save_hdf5_datasets(filename, names, datasets):
    with h5py.File(filename, "a") as file:
        for name, dataset in zip(names, datasets):
            file.create_dataset(name, data=dataset)


def generate_dataset(conf):
    filename = conf["dataset"]["save_directory"]

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # save the metadata
    metadata = flatten(config, reducer="dot")
    with h5py.File(filename, "a") as file:
        for k, v in metadata.items():
            file.attrs.create(k, data=v)

    # create the agent
    agent = RoomExplorer.MobileArm(**conf["agent"])

    # save the regular motor exploration
    motor_grid, state_grid = agent.generate_regular_states()
    save_hdf5_datasets(filename,
                       names=["agent/motor_grid", "agent/state_grid"],
                       datasets=[motor_grid, state_grid])

    # iterate over the runs
    for r in range(conf["dataset"]["n_runs"]):
        name = "env{:02}".format(r)
        print(name)

        # create an environment
        room = RoomExplorer.Room(**conf["environment"])

        # # create the env group, save the data and metadata
        # with h5py.File(filename, "a") as file:
        #     file.create_group(name)

        # generate_dataset with dynamic base
        data = explore(agent, room, "dynamic_base",
                       conf["dataset"]["n_transitions"])
        # save the datasets
        # with h5py.File(filename, "a") as file:
        #     file.create_dataset(name + "/dynamic_base/motor_t", data=data[0])
        #     file.create_dataset(name + "/dynamic_base/sensor_t", data=data[1])
        #     file.create_dataset(name + "/dynamic_base/motor_tp", data=data[2])
        #     file.create_dataset(name + "/dynamic_base/sensor_tp", data=data[3])
        save_hdf5_datasets(filename,
                           names=[name + "/dynamic_base/motor_t",
                                  name + "/dynamic_base/sensor_t",
                                  name + "/dynamic_base/motor_tp",
                                  name + "/dynamic_base/sensor_tp"],
                           datasets=data)
        del data

        # generate_dataset with static base
        data = explore(agent, room, "static_base",
                       conf["dataset"]["n_transitions"])
        # save the datasets
        # with h5py.File(filename, "a") as file:
        #     file.create_dataset(name + "/static_base/motor_t", data=data[0])
        #     file.create_dataset(name + "/static_base/sensor_t", data=data[1])
        #     file.create_dataset(name + "/static_base/motor_tp", data=data[2])
        #     file.create_dataset(name + "/static_base/sensor_tp", data=data[3])
        save_hdf5_datasets(filename,
                           names=[name + "/static_base/motor_t",
                                  name + "/static_base/sensor_t",
                                  name + "/static_base/motor_tp",
                                  name + "/static_base/sensor_tp"],
                           datasets=data)
        del data

        # generate_dataset with static base
        data = explore(agent, room, "hopping_base",
                       conf["dataset"]["n_transitions"])
        # save the datasets
        # with h5py.File(filename, "a") as file:
        #     file.create_dataset(name + "/hopping_base/motor_t", data=data[0])
        #     file.create_dataset(name + "/hopping_base/sensor_t", data=data[1])
        #     file.create_dataset(name + "/hopping_base/motor_tp", data=data[2])
        #     file.create_dataset(name + "/hopping_base/sensor_tp", data=data[3])
        save_hdf5_datasets(filename,
                           names=[name + "/hopping_base/motor_t",
                                  name + "/hopping_base/sensor_t",
                                  name + "/hopping_base/motor_tp",
                                  name + "/hopping_base/sensor_tp"],
                           datasets=data)
        del data

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

    config["dataset"]["git_commit"] = get_git_hash()

    if os.path.splitext(config["dataset"]["save_directory"])[1] == "":
        config["dataset"]["save_directory"] += ".hdf5"
    if os.path.exists(config["dataset"]["save_directory"]):
        config["dataset"]["save_directory"] = append_time(
            config["dataset"]["save_directory"])

    generate_dataset(config)
