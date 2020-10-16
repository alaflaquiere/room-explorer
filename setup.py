import setuptools


setuptools.setup(
    name="roomexplorer",
    version="0.0.0",
    packages=setuptools.find_packages(),
    package_data={"roomexplorer": [
        "renderer/urdf/*",
        "renderer/textures/*"]},
)
