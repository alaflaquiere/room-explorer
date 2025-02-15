import os
import glob
import numpy as np
import pybullet
import colorsys

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
TEXTURES_PATH = os.path.join(ROOT_PATH, "..", "textures")
URDF_PATH = os.path.join(ROOT_PATH, "..", "urdf")

FLOOR_TEXTURES = glob.glob(os.path.join(TEXTURES_PATH, "*floor*.tga"))
WALL_TEXTURES = glob.glob(os.path.join(TEXTURES_PATH, "*wall*.tga"))

FLOOR = os.path.join(URDF_PATH, "floor.urdf")
WALLS = os.path.join(URDF_PATH, "walls.urdf")
OBJECTS = glob.glob(os.path.join(URDF_PATH, "*.urdf"))
OBJECTS.remove(FLOOR)
OBJECTS.remove(WALLS)


def get_colors(num_colors=6):
    colors = []
    for n in range(num_colors):
        hue = n / num_colors
        saturation = 0.9
        lightness = 0.9
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        colors.append([red, green, blue, 1])
    return colors


def build_scene(
        floor_textures=FLOOR_TEXTURES,
        wall_textures=WALL_TEXTURES,
        fix_light_position=False,
        gui=False):
    """
    Builds the scene
    """
    if gui:
        physics_client = pybullet.connect(pybullet.GUI)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
            0,
            physicsClientId=physics_client)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            0,
            physicsClientId=physics_client)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            0,
            physicsClientId=physics_client)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SHADOWS,
            1,
            physicsClientId=physics_client)
    else:
        physics_client = pybullet.connect(pybullet.DIRECT)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SHADOWS,
            1,
            physicsClientId=physics_client)

    # Add current folder path to the bullet research path
    pybullet.setAdditionalSearchPath(
        os.path.dirname(os.path.realpath(__file__)))

    # load the floor
    floorId = pybullet.loadURDF(FLOOR,
                                globalScaling=1,
                                basePosition=[0., 0., 0.],
                                baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=1)
    floor_texture = pybullet.loadTexture(np.random.choice(floor_textures))
    pybullet.changeVisualShape(floorId, -1, textureUniqueId=floor_texture)

    # load the walls
    wallsId = pybullet.loadURDF(WALLS,  # "../urdf/walls.urdf"
                                globalScaling=1,
                                basePosition=[0., 0., 0.],
                                baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=1)
    wall_texture = pybullet.loadTexture(np.random.choice(wall_textures))
    pybullet.changeVisualShape(wallsId, -1, textureUniqueId=wall_texture)

    # Light position
    if fix_light_position:
        translation = [1, -1, 1.6]
    else:
        xz = np.random.uniform(-1, 1, size=2)
        translation = np.array([xz[0], 1, xz[1]])

    pybullet.configureDebugVisualizer(
        lightPosition=translation,
        physicsClientId=physics_client)


def place_objects(
        colors,
        objects=OBJECTS,
        max_num_objects=3,
        min_num_objects=1,
        discrete_position=False,
        rotate_object=False):
    directions = np.linspace(-2.49, 2.49, 4)  # [-3., -1., 1., 3.]
    available_positions = []

    for y in directions:
        for x in directions:
            available_positions.append((x, y))

    available_positions = np.array(available_positions)
    num_objects = np.random.choice(range(min_num_objects, max_num_objects + 1))
    indices = np.random.choice(
        np.arange(len(available_positions)),
        replace=False,
        size=num_objects)

    for xy in available_positions[indices]:
        pos = xy + np.random.uniform(-0.3, 0.3, size=xy.shape) if not discrete_position else xy
        rot = pybullet.getQuaternionFromEuler((0, 0, 2 * np.pi * np.random.rand())) if rotate_object else [0, 0, 0, 1]
        color = colors[np.random.choice(len(colors))]
        obj_id = pybullet.loadURDF(np.random.choice(objects),
                                   globalScaling=0.85,
                                   basePosition=[pos[0], pos[1], 0.],
                                   baseOrientation=rot,
                                   useFixedBase=1)
        pybullet.changeVisualShape(obj_id, -1, rgbaColor=color)


def tear_down_scene():
    """
    Tears the scene down
    """
    pybullet.disconnect()
