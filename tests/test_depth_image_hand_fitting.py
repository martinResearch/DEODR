"""Test using depth image hand fitting."""

import numpy as np
import pytest

from deodr.examples.depth_image_hand_fitting import run


def test_depth_image_hand_fitting_pytorch() -> None:
    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )

    possible_results = [
        251.32711067513003,
        251.31652686512888,
        251.31652686495823,
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


def test_depth_image_hand_fitting_numpy() -> None:
    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )

    possible_results = [
        251.32711113732933,
        251.32711113730954,
        251.3271111242092,
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


@pytest.mark.skip(reason="Tensorflow does nto support numpy 2.0 yet")
def test_depth_image_hand_fitting_tensorflow() -> None:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # Running on CPU to get determinisic results

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )

    possible_results = [
        251.31648932312913,
        251.3164914350016,
        251.3164891265543,
        251.32711038915872,
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


if __name__ == "__main__":
    test_depth_image_hand_fitting_pytorch()
    test_depth_image_hand_fitting_numpy()

    test_depth_image_hand_fitting_tensorflow()
