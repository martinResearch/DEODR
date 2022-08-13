"""Test using rgb_image hand fitting."""

import numpy as np

from deodr.examples.rgb_image_hand_fitting import run

import tensorflow as tf


def test_rgb_image_hand_fitting_pytorch() -> None:

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )

    possible_results = [
        2103.665850588039,  # github action ubuntu python 3.7
        2104.9656991756697,  # github action ubuntu python 3.9
        2100.0239709048583,
        2132.9307950405196,
        2106.5436357944604,
        2117.9946156293213,  # google colab
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


def test_rgb_image_hand_fitting_numpy() -> None:

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    # getting different result on python 3.6 or 3.7 in the github action, not sure why
    possible_results = [
        2122.8322696714026,  # python 3.7 on github action
        2107.850380422819,
        2107.850380422819,  # google colab Intel(R) Xeon(R) CPU @ 2.20GHz
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


def test_rgb_image_hand_fitting_tensorflow() -> None:

    tf.config.set_visible_devices(
        [], "GPU"
    )  # Running on CPU to get deterministic results

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    possible_results = [
        2112.9566220857746,
        2115.9320061795634,
        2107.962374538259,
        2115.9974345976066,
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


if __name__ == "__main__":
    test_rgb_image_hand_fitting_pytorch()
    test_rgb_image_hand_fitting_tensorflow()
    test_rgb_image_hand_fitting_numpy()
