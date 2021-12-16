"""Test using depth image hand fitting."""
import os

from deodr.examples.depth_image_hand_fitting import run

import tensorflow as tf


def test_depth_image_hand_fitting_pytorch():

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    if os.name == "nt":  # windows
        assert abs(energies[49] - 251.32711067513003) < 1e-10
    else:
        assert abs(energies[49] - 251.31652686512888) < 1e-5


def test_depth_image_hand_fitting_numpy():

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    if os.name == "nt":  # windows
        assert abs(energies[49] - 251.32711113732933) < 1e-10
    else:
        assert abs(energies[49] - 251.32711113730954) < 1e-5


def test_depth_image_hand_fitting_tensorflow():

    tf.config.set_visible_devices(
        [], "GPU"
    )  # Running on CPU to get determinisic results

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    if os.name == "nt":  # windows
        assert abs(energies[49] - 251.3164879047919) < 1e-10
    else:
        assert abs(energies[49] - 251.31648983466366) < 1e-5


if __name__ == "__main__":

    test_depth_image_hand_fitting_pytorch()
    test_depth_image_hand_fitting_numpy()
    test_depth_image_hand_fitting_tensorflow()
