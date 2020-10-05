import numpy as np
import matplotlib.pyplot as plt
from environment.RoomEnvironment import Room
from agent.Agents import MobileArm


def show_sensation(s, new_fig=True, block=False):
    assert type(s) is np.ndarray
    if s.ndim == 1:
        resolution = int(np.sqrt(s.shape[0] / 3))
        s = s.reshape(resolution, resolution, 3)
    if new_fig:
        plt.figure(figsize=(6, 6))
    else:
        plt.cla()
    plt.tight_layout()
    plt.imshow(s, interpolation="none")
    plt.axis("off")
    plt.show(block=block)


def interleave(a, b):
    assert a.shape[0] == b.shape[0]
    c = np.empty((2 * a.shape[0], b.shape[1]),
                 dtype=a.dtype)
    c[0::2, :] = a
    c[1::2, :] = b
    return c


def main():
    myroom = Room(resolution=16)
    image = myroom.overview()
    show_sensation(image)
    myroom.save("temp")

    myagent = MobileArm()
    myagent.save("temp")
    motors_t, shifts_t, states_t, motors_tp, shifts_tp, states_tp =\
        myagent.generate_random_transitions("hopping_base", 30)

    motors = interleave(motors_t, motors_tp)
    shifts = interleave(shifts_t, shifts_tp)
    states = interleave(states_t, states_tp)

    sensations = myroom.get_sensations(states)
    print("> sensations: \n", sensations)

    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.subplots(1, 2)
    for m, sh, s in zip(motors, shifts, sensations):
        plt.sca(ax1)
        show_sensation(s, new_fig=False)
        plt.sca(ax2)
        myagent.display(m, sh, new_fig=False)
        plt.pause(0.15)
    plt.show(block=True)


if __name__ == '__main__':
    main()
