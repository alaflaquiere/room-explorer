import os
import numpy as np
import pybullet
import matplotlib.pyplot as plt
import json
import _pickle as cpickle
from tqdm import tqdm
import renderer.bullet.bullet_tools as bullet_tools
from renderer.bullet.camera import Camera, CameraResolution

"""
Collection of environments that can be used by generate-sensorimotor-data.py.
Environments are used to generate a sensory input for each position of the sensor (the environment thus implicitly includes information about the 
agent's sensor).
"""


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class Room:
    """
    TODO-------------------------------------------------------------------------------
    A 3D room filled with random objects. The position (0,0) corresponds to the center of the room.
    At each position, the environment generates a sensory input corresponding to the reading of a RGB camera with a fixed orientation.
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
        self.size = size
        self.n_obstacles = n_obstacles

        # Build the scene
        bullet_tools.build_scene(fix_light_position=True)

        # Create the objects
        bullet_tools.place_objects(
            bullet_tools.get_colors(12),
            min_num_objects=self.n_obstacles,
            max_num_objects=self.n_obstacles,
            discrete_position=False,
            rotate_object=True)

        # Create the camera
        self._camera = Camera(45, CameraResolution(resolution, resolution))
        # Set the camera position and orientation
        self._camera.setTranslation([0, -1, 1])
        self._camera_height = 1.6
        camera_direction = np.array([2.5, 1.8, 0])
        yaw, pitch = bullet_tools.compute_yaw_and_pitch(camera_direction)
        self._camera.setOrientation(pybullet.getQuaternionFromEuler([0.0, pitch, yaw]))

    def get_sensations(self, positions):
        """
        Returns the sensations at a given set of input positions.
        Inputs:
            positions - (N, 2) array
        Returns:
            sensations - (self.n_sensations, 4) array
        """
        # Deal with the case of a single position
        if positions.ndim == 1:
            positions = np.expand_dims(positions, 0)
        assert positions.shape[1] == 2, "positions should be of shape (N, 2)"
        N = positions.shape[0]

        sensations = np.empty((N, self.resolution * self.resolution * 3), dtype=np.float)
        for i, position in enumerate(tqdm(positions, desc="Room exploration", mininterval=1)):
            # set the camera position
            camera_position = [position[0], self._camera_height, position[1]]
            self._camera.setTranslation(bullet_tools.transform_pos_for_bullet(camera_position))
            # render
            image = self._camera.getFrame()
            # save sensation
            sensations[i, :] = image.reshape(-1)
        return sensations

    def DEPRECATED_generate_shift(self, k=1, static=False):
        """
        TODO this outside of the class
        Returns k random shifts for the environment in [-1.75, 1.75]^2.
        If static=True, returns the default shift which is [0, 0].
        """
        if static:
            shift = np.zeros((k, 2))
        else:
            shift = np.array(self.environment_size)/2 * np.random.rand(k, 2) - np.array(self.environment_size)/4
        return shift

    @staticmethod
    def destroy():
        """
        Disconnect the pybullet scene.
        """
        bullet_tools.tear_down_scene()

    @staticmethod
    def overview():
        camera_position = [8, 8, 8]
        camera_direction = np.array((5, 4.7, 5))
        resolution = 512
        overview_camera = Camera(45, CameraResolution(resolution, resolution))
        # set the camera orientation and position
        yaw, pitch = bullet_tools.compute_yaw_and_pitch(camera_direction)
        overview_camera.setOrientation(pybullet.getQuaternionFromEuler([0.0, pitch, yaw]))
        overview_camera.setTranslation(bullet_tools.transform_pos_for_bullet(camera_position))
        image = overview_camera.getFrame()
        return image

    def save(self, directory):
        """
        Save the environment on disk.
        """
        try:
            serializable_dict = self.__dict__.copy()
            for key, value in self.__dict__.items():
                if type(value) is np.ndarray:
                    serializable_dict[key] = value.tolist()  # make the np.arrays serializable
                    continue
                if not is_jsonable(value):
                    del serializable_dict[key]

            if not os.path.exists(directory):
                os.mkdir(directory)

            with open(os.path.join(directory, "environment_params.txt"), "w") as f:
                json.dump(serializable_dict, f, indent=1)

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

        except Exception:
            print("ERROR: saving the environment in {} failed".format(directory))
            return False
