from RoomEnvironment.RoomEnvironment import Room
import matplotlib.pyplot as plt
import numpy as np


def main():
    myroom = Room()

    image = myroom.overview()
    plt.imshow(image)
    plt.show()

    myroom.save("saved_environment")

    sensations = myroom.get_sensations(np.random.rand(1000, 2))
    print("> sensations: \n", sensations)


if __name__ == '__main__':
    main()
