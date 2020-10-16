import os
import numpy as np
import pybullet
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import yaml
import _pickle as cpickle
from tqdm import tqdm

import roomexplorer

"""
Collection of environments that can be used by generate-sensorimotor-data.py.
Environments are used to generate a sensory input for each position of the sensor (the environment thus implicitly includes information about the 
agent's sensor).
"""


def is_serialable(x):
    try:
        yaml.dump(x)
        return True
    except (TypeError, OverflowError):
        return False


def apply_aperture(img, aperture):
    assert 0. <= aperture <= 1.
    img = rgb_to_hsv(img.reshape(-1, 3) / 255)
    img[:, 2] *= aperture
    img = hsv_to_rgb(img) * 255
    return img.astype(np.uint8)


class Room:
    """ A 3D room filled with random objects. Position (0,0)
    corresponds to the center of the room. At each position,
    the environment generates a sensory input corresponding
    to the reading of a RGB camera with a fixed orientation.
    Attributes
    ----------
    resolution : int
        camera resolution (square field of view)
    resolution : list, tuple
        room side length
    n_obstacles : int
        number of obstacles in the environment
    """

    def __init__(self, resolution=16, size=(7, 7), n_obstacles=16):
        self.resolution = resolution
        self.size = np.array(size)
        self.n_obstacles = n_obstacles

        # Build the scene
        roomexplorer.bullet_tools.build_scene(fix_light_position=True)

        # Create the objects
        roomexplorer.bullet_tools.place_objects(
            roomexplorer.bullet_tools.get_colors(12),
            min_num_objects=self.n_obstacles,
            max_num_objects=self.n_obstacles,
            discrete_position=False,
            rotate_object=True)

        # Create the camera
        self._camera = roomexplorer.Camera(70,
                                           roomexplorer.CameraResolution(resolution,
                                                                         resolution))  # former fov: 45
        self._camera_height = 1.6
        self._pitch = 0.62

    def get_sensations(self, states):
        """
        Returns the sensations at a given set of input positions.
        Inputs:
            positions - (N, 4) array of positions [x, y, yaw, aperture]
        Returns:
            sensations - (N, self.resolution * self.resolution * 3) array
        """
        # Deal with the case of a single position
        if states.ndim == 1:
            states = states.reshape(1, -1)
        assert states.shape[1] == 4, "positions should be of shape (N, 4)"
        N = states.shape[0]

        sensations = np.empty((N, self.resolution * self.resolution * 3), dtype=np.uint8)
        for i, state in enumerate(tqdm(states, desc="Room exploration", mininterval=1)):
            assert (-self.size / 2 < state[0:2]).all() \
                   and (state[0:2] < self.size / 2).all(), \
                   "sensor outside of the room"
            # set the camera position
            camera_position = [state[0], state[1], self._camera_height]
            self._camera.setPosition(translation=camera_position,
                                     quaternion=pybullet.getQuaternionFromEuler([0.0,
                                                                                 self._pitch,
                                                                                 state[2]]))

            # render
            image = self._camera.getFrame()
            # aperture
            image = apply_aperture(image, state[3])
            # save sensation
            sensations[i, :] = image.reshape(-1)
        return sensations

    @staticmethod
    def destroy():
        """
        Disconnect the pybullet scene.
        """
        roomexplorer.bullet_tools.tear_down_scene()

    @staticmethod
    def overview():
        camera_position = [-7.5, -7.5, 7.5]  # [8, 8, 8]
        resolution = 512
        overview_camera = roomexplorer.Camera(45,
                                              roomexplorer.CameraResolution(resolution,
                                                                            resolution))
        # set the camera orientation and position
        yaw = np.pi / 4
        pitch = np.pi / 5.5
        overview_camera.setPosition(translation=camera_position,
                                    quaternion=pybullet.getQuaternionFromEuler([0.0, pitch, yaw]))

        image = overview_camera.getFrame()
        return image

    def save(self, directory):
        """
        Save the environment on disk.
        """
        serializable_dict = self.__dict__.copy()
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                serializable_dict[key] = value.tolist()  # make the np.arrays serializable
                continue
            if not is_serialable(value):
                del serializable_dict[key]

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, "environment_parameters.yml"), "w") as f:
            yaml.dump(serializable_dict, f, indent=1)

        # save the object on disk
        with open(os.path.join(directory, "environment.pkl"), "wb") as f:
            cpickle.dump(self, f)

        # save an image of the environment
        image = self.overview()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.tight_layout()
        ax.imshow(image, interpolation="none")
        ax.axis("off")
        fig.savefig(os.path.join(directory, "environment_image.png"))
        plt.close(fig)
