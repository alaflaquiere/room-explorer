import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import _pickle as cpickle


def process_yaml_list(x):
    if type(x) is np.ndarray:
        return x
    else:
        return np.array([eval(i) if type(i) is str else i for i in x])


class MobileArm:
    """TODO
    type: str
            type of agent
        n_motors : int
            number of independent motor components in [-1, 1]
        size_regular_grid: int
            number of samples that form the regular sampling of the motor space

        generate_random_sampling(k):
            randomly explores the motor space and returns the motor samples and corresponding sensor egocentric positions
        generate_regular_sampling():
            returns a regular sampling of the motor space and the corresponding sensor egocentric positions
        display(motor):
            displays the agent's configuration associated with an input motor command
        log(dir):
            logs the agent's parameters
    """

    def __init__(self,
                 amplitudes=np.array([np.pi, np.pi, np.pi, np.pi, 0.8]),
                 lengths=np.array([0.5, 0.5, 0.5, 0]),
                 fixed_orientation=False):
        self.n_motors = 5
        self.amps = process_yaml_list(amplitudes)
        self.lens = process_yaml_list(lengths)
        self.fixed_orientation = fixed_orientation

    @staticmethod
    def check_motor(m):
        assert type(m) is np.ndarray
        assert (-1 <= m).all() and (m <= 1).all(), "motor not in range (-1, 1)"
        if m.ndim == 1:
            m = m.reshape(1, -1)
        return m

    def check_shift(self, sh):
        assert type(sh) is np.ndarray
        if sh.ndim == 1:
            sh = sh.reshape(1, -1)
        assert (-np.sum(self.lens) <= sh[:, 0:2]).all()\
               and (sh[:, 0:2] <= np.sum(self.lens)).all(),\
               "shift[0:2] not in range (-arm_reach, arm_reach)"
        assert (0 <= sh[:, 2]).all() and (sh[:, 2] <= 2 * np.pi).all(), \
               "shift[2] not in range (0, 2*pi)"
        return sh

    def get_state(self, m, shift):
        """Get the position/orientation/aperture of the sensor.
        """
        m = self.check_motor(m)
        shift = self.check_shift(shift)
        # relative angles
        rel_a = self.amps[:4].reshape(1, 4) * m[:, :4]
        # fix sensor orientation if necessary
        if self.fixed_orientation:
            total_a = np.sum(rel_a[:, :4], axis=1)
            assert (np.fmod(total_a, 2 * np.pi) < 1e-8).all(),\
                "the sensor orientation is expected to be fixed"
        # update base angle
        rel_a[:, 0] = rel_a[:, 0] + shift[:, 2]
        # sensor positions
        x = np.sum(
            np.multiply(
                self.lens,
                np.cos(
                    np.cumsum(rel_a, axis=1))),
            axis=1, keepdims=True)
        y = np.sum(
            np.multiply(
                self.lens,
                np.sin(
                    np.cumsum(rel_a, axis=1))),
            axis=1, keepdims=True)
        yaw = np.sum(rel_a[:, :4], axis=1, keepdims=True)
        yaw = np.mod(yaw, 2 * np.pi)
        aperture = self.amps[4] / 2 * (m[:, [4]] - 1) + 1
        # update the position
        x += shift[:, [0]]
        y += shift[:, [1]]
        return np.hstack((x, y, yaw, aperture))

    def fix_orientation(self, m):
        rel_a = self.amps[:4].reshape(1, 4) * m[:, :4]
        rel_a[:, 3] = -np.sum(rel_a[:, :3], axis=1)
        rel_a[:, 3] = np.mod(rel_a[:, 3] + np.pi, 2 * np.pi) - np.pi
        m[:, 3] = rel_a[:, 3] / self.amps[3]
        return m

    def generate_random_motors(self, k=1):
        """Draw a set of k random motor commands in [-1, 1]
        Returns:
            motors - (k, 5) array
        """
        motors = 2 * np.random.rand(k, self.n_motors) - 1
        if self.fixed_orientation:
            motors = self.fix_orientation(motors)
        return motors

    def generate_random_shifts(self, k=1):
        """Draw a set of k random shifts with a max amplitude
        equal to the arm reach.
        Returns:
            shifts - (k, 3) array
        """
        xy = np.sum(self.lens) * (2 * np.random.rand(k, 2) - 1)
        if self.fixed_orientation:
            a = 2 * np.pi * np.zeros((k, 1))
        else:
            a = 2 * np.pi * np.random.rand(k, 1)
        return np.hstack((xy, a))

    def generate_random_states(self, k=1):
        """Draw a set of k randomly selected motor configurations and associated egocentric sensor positions
        Returns:
            motors - (k, self.n_motors) array
            states - (k, 2) array
        """
        # draw random motor components in [-1, 1]
        motors = self.generate_random_motors(k)
        # draw random base shifts
        shifts = self.generate_random_shifts(k)
        # get the associated egocentric positions
        states = self.get_state(motors, shifts)
        return motors, shifts, states

    def generate_random_transitions(self, mode, k=1):
        """TODO
        Draw a set of k randomly selected motor configurations and associated egocentric sensor positions
        MODE
        Returns:
            motors - (k, self.n_motors) array
            states - (k, 2) array
        """
        assert mode in ["dynamic_base", "static_base", "hopping_base"]
        # draw base shifts
        if mode is "dynamic_base":
            shifts_t = self.generate_random_shifts(k)
            shifts_tp = self.generate_random_shifts(k)
        elif mode is "static_base":
            shifts_t = np.tile(np.random.rand(1, 3), (k, 1))
            shifts_tp = shifts_t.copy()
        elif mode is "hopping_base":
            shifts_t = self.generate_random_shifts(k)
            shifts_tp = shifts_t.copy()
        # draw random motor components in [-1, 1]
        motors_t = self.generate_random_motors(k)
        motors_tp = self.generate_random_motors(k)
        # get the associated egocentric positions
        states_t = self.get_state(motors_t, shifts_t)
        states_tp = self.get_state(motors_tp, shifts_tp)
        return motors_t, shifts_t, states_t, motors_tp, shifts_tp, states_tp

    def generate_regular_states(self, resolution=7):
        """Generates a regular grid of motor configurations in the motor space."""
        # create a grid of coordinates
        coordinates = np.array(
            np.meshgrid(
                *list([np.linspace(-1, 1, resolution)]) * self.n_motors
            )
        )
        # reshape the coordinates into matrix of size (resolution**n_motors, n_motors)
        motor_grid = np.array(
            [coord.reshape(-1) for coord in coordinates]
        ).T
        # fix_orientation if necessary
        if self.fixed_orientation:
            motor_grid = self.fix_orientation(motor_grid)
        # no shift
        shifts = np.zeros((motor_grid.shape[0], 3))
        # get the corresponding positions
        state_grid = self.get_state(motor_grid, shifts)
        return motor_grid, state_grid

    def display(self, m, shift, new_fig=True):
        """Displays the position associated with a motor configuration"""
        m = self.check_motor(m)
        shift = self.check_shift(shift)
        # relative angles
        rel_a = self.amps[:4].reshape(1, 4) * m[:, :4]
        if self.fixed_orientation:
            total_a = np.sum(rel_a[:, :4], axis=1)
            assert (np.fmod(total_a, 2 * np.pi) < 1e-8).all(), \
                "the sensor orientation is expected to be fixed"
        # update base angle
        rel_a[:, 0] = rel_a[:, 0] + shift[:, 2]
        # joints positions
        x = np.cumsum(
            np.multiply(
                self.lens,
                np.cos(
                    np.cumsum(rel_a, axis=1))),
            axis=1)
        y = np.cumsum(
            np.multiply(
                self.lens,
                np.sin(
                    np.cumsum(rel_a, axis=1))),
            axis=1)

        # add the agent's base
        x = np.hstack((np.zeros((x.shape[0], 1)), x))
        y = np.hstack((np.zeros((y.shape[0], 1)), y))

        yaw = np.sum(rel_a, axis=1, keepdims=True)
        aperture = self.amps[4] / 2 * (m[:, [4]] - 1) + 1

        # update the positions with the shift
        x += shift[:, [0]]
        y += shift[:, [1]]

        # display the different motor configurations
        if new_fig:
            plt.figure()
        plt.cla()
        for x_, y_, yaw_, ap_ in zip(x, y, yaw, aperture):
            plt.plot(x_, y_, '-o')
            plt.quiver(x_[-1], y_[-1],
                       np.cos(yaw_), np.sin(yaw_),
                       color=(ap_ + self.amps[4] - 1) * [1, 1, 1]  # gray proportional to aperture
                       )
            circ = plt.Circle((x_[-1], y_[-1]),
                              0.4 * ap_,
                              edgecolor=(ap_ + self.amps[4] - 1) * [1, 1, 1],  # gray proportional to aperture
                              facecolor="none",
                              linewidth=3)
            plt.gca().add_artist(circ)
        rect = plt.Rectangle((-3.5, -3.5), 7, 7,
                             edgecolor="k",
                             linewidth=3,
                             facecolor="none")
        plt.gca().add_artist(rect)
        plt.axis("equal")
        r = np.sum(self.lens) * 2.4
        plt.axis([-r, r, -r, r])

    def save(self, directory):
        """TODO save as yaml file
        Save the agent on disk.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        serializable_dict = self.__dict__.copy()
        for key, value in serializable_dict.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()  # make the np.arrays serializable

        with open(os.path.join(directory, "agent_parameters.yml"), "w") as f:
            yaml.dump(serializable_dict, f)

        with open(os.path.join(directory, "agent.pkl"), "wb") as f:
            cpickle.dump(self, f)
