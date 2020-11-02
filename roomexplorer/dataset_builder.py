import os
import h5py
from flatten_dict import flatten
import roomexplorer


def explore(agent, room, mode, k, progress_bar=True):
    assert mode in ["dynamic_base", "static_base", "hopping_base"]
    # print("mode: {}".format(mode))
    motors_t, shifts_t, states_t, motors_tp, shifts_tp, states_tp = \
        agent.generate_random_transitions(mode, k)
    sensors_t = room.get_sensations(states_t, progress_bar)
    sensors_tp = room.get_sensations(states_tp, progress_bar)
    return motors_t, sensors_t, motors_tp, sensors_tp


def save_hdf5_datasets(filename, names, datasets):
    with h5py.File(filename, "a") as file:
        for name, dataset in zip(names, datasets):
            file.create_dataset(name, data=dataset)


def generate_dataset(conf):
    filename = conf["dataset"]["save_directory"]

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # save the metadata
    metadata = flatten(conf, reducer="dot")
    with h5py.File(filename, "a") as file:
        for k, v in metadata.items():
            file.attrs.create(k, data=v)

    # create the agent
    agent = roomexplorer.MobileArm(**conf["agent"])

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
        room = roomexplorer.Room(**conf["environment"])

        # generate_dataset with dynamic base
        data = explore(agent, room, "dynamic_base",
                       conf["dataset"]["n_transitions"])
        # save the datasets
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
        save_hdf5_datasets(filename,
                           names=[name + "/hopping_base/motor_t",
                                  name + "/hopping_base/sensor_t",
                                  name + "/hopping_base/motor_tp",
                                  name + "/hopping_base/sensor_tp"],
                           datasets=data)
        del data

        # clean
        room.destroy()
