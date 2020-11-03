import os
import numpy as np
import pybullet
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import yaml
import _pickle as cpickle
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor

from roomexplorer.renderer.bullet import bullet_tools
from roomexplorer.renderer.bullet.camera import Camera, CameraResolution

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


def create_environment(conf):
    assert conf["kind"] in ("room", "ether"),\
        "valid environment types: 'room', 'ether'"
    if conf["kind"] == "room":
        return Room(**conf)
    elif conf["kind"] == "ether":
        return Ether(**conf)
    else:
        return None


class Room:
    """ A 3D room filled with random objects. Position (0,0)
    corresponds to the center of the room. At each position,
    the environment generates a sensory input corresponding
    to the reading of a RGB camera with a fixed orientation.
    Attributes
    ----------
    resolution : int
        camera resolution (square field of view)
    size : list, tuple
        room side length
    n_obstacles : int
        number of obstacles in the environment
    """

    def __init__(self, resolution=16, size=(7, 7), n_obstacles=16, kind="room"):
        self.kind = kind
        self.resolution = resolution
        self.size = np.array(size)
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
        self._camera = Camera(70,
                              CameraResolution(resolution,
                                               resolution))  # former fov: 45
        self._camera_height = 1.6
        self._pitch = 0.62

    def get_sensations(self, states, progress_bar=True):
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
        assert np.all(-self.size / 2 <= states[:, 0:2]) and np.all(states[:, 0:2] <= self.size / 2), \
            "sensor outside of the room"
        assert np.all(-np.pi <= states[:, 2]) and np.all(states[:, 2] <= np.pi), \
            "orientation outside of [-pi, pi] range"
        assert np.all(0 <= states[:, 3]) and np.all(states[:, 3] <= 1), \
            "aperture outside of [0, 1] range"

        sensations = np.empty((N, self.resolution * self.resolution * 3), dtype=np.uint8)
        for i, state in enumerate(tqdm(states, desc="Room exploration", mininterval=1, disable=(not progress_bar))):
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
        bullet_tools.tear_down_scene()

    @staticmethod
    def overview():
        camera_position = [-7.5, -7.5, 7.5]
        resolution = 512
        overview_camera = Camera(45,
                                 CameraResolution(resolution, resolution))
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


class Ether:
    """ TODO
    Attributes
    ----------
    resolution : int
    size : list, tuple
        room side length
    """

    def __init__(self, resolution=6, size=(7, 7), kind="ether"):
        self.kind = kind
        self.resolution = resolution
        self.size = np.array(size)

        # create the mapping
        N = 5 * self.resolution  # number of anchors
        X = 2 * (np.random.rand(N, 4) - 0.5)
        Y = 2 * np.random.rand(N, resolution) - 1
        self.gp = GaussianProcessRegressor()
        self.gp.fit(X, Y)

    def get_sensations(self, states, progress_bar=False):
        """
        Returns the sensations at a given set of input positions.
        Inputs:
            states - (N, 4) array of positions [x, y, yaw, aperture]
        Returns:
            sensations - (N, self.resolution) array
        """
        # Deal with the case of a single position
        if states.ndim == 1:
            states = states.reshape(1, -1)
        assert states.shape[1] == 4, "positions should be of shape (N, 4)"
        assert np.all(-self.size / 2 <= states[:, 0:2]) and np.all(states[:, 0:2] <= self.size / 2),\
            "sensor outside of the room"
        assert np.all(-np.pi <= states[:, 2]) and np.all(states[:, 2] <= np.pi), \
            "orientation outside of [-pi, pi] range"
        assert np.all(0 <= states[:, 3]) and np.all(states[:, 3] <= 1), \
            "aperture outside of [0, 1] range"
        # rescale everything into unitary hypercube
        states[:, 0:2] /= self.size / 2
        states[:, 2] /= np.pi
        states[:, 3] = 2 * states[:, 3] - 1
        sensations = self.gp.predict(states)
        return sensations

    def overview(self, dims=(0, 1)):
        reso = 50
        temp_X = np.vstack(map(np.ravel,
                               np.meshgrid(np.linspace(-1, 1, reso),
                                           np.linspace(-1, 1, reso))
                               )
                           ).T
        test_X = np.zeros((temp_X.shape[0], 4))
        for i, d in enumerate(dims):
            test_X[:, d] = temp_X[:, i]
        test_Y = self.gp.predict(test_X)
        fig = plt.figure()
        ax = plt.subplot(111, projection="3d")
        for k in range(self.gp.y_train_.shape[1]):
            ax.plot(self.gp.X_train_[:, 0],
                    self.gp.X_train_[:, 1],
                    self.gp.y_train_[:, k], "o", color=cm.Set1(k))
            ax.plot_surface(test_X[:, 0].reshape(reso, reso),
                            test_X[:, 1].reshape(reso, reso),
                            test_Y[:, k].reshape(reso, reso),
                            color=cm.Set1(k), alpha=0.3)
        plt.show(block=True)
        return fig

    @staticmethod
    def destroy():
        pass

    def save(self, directory):
        """
        Save the environment on disk.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        anchors = {"X": self.gp.X_train_,
                   "Y": self.gp.y_train_}

        with open(os.path.join(directory, "environment_parameters.yml"), "w") as f:
            yaml.dump(anchors, f, indent=1)

        # save the object on disk
        with open(os.path.join(directory, "environment.pkl"), "wb") as f:
            cpickle.dump(self, f)

        # save an image of the environment
        fig = self.overview()
        fig.savefig(os.path.join(directory, "environment_image.png"))
        plt.close(fig)
