"""Test using rgb_image hand fitting."""

import os

from deodr.examples.rgb_image_hand_fitting import run

import tensorflow as tf


def test_rgb_image_hand_fitting_pytorch():

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    if os.name == "nt":  # windows
        assert (abs(energies[49] - 2100.0239709048583) < 1e-10) or (
            abs(energies[49] - 2132.9307950405196) < 1e-10
        )  # 2132 : result on lenovo laptop using Intel Core i5 4210-U
    else:
        assert abs(energies[49] - 2106.5436357944604) < 12


def test_rgb_image_hand_fitting_numpy():

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    # getting different result on python 3.6 or 3.7 in the github action, not sure why
    if os.name == "nt":  # windows
        assert abs(energies[49] - 2107.850380422819) < 1e-10
    else:
        assert abs(energies[49] - 2113.7013184079137) < 2


def test_rgb_image_hand_fitting_tensorflow():

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
        assert abs(energies[49] - 2112.9566220857746) < 1e-10
        # seems to change a lot from one run to the next when using GPU
        # could use os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # github action 2132.9307950405196
        # 2110.1568876644037
        # 2206.686787515083
    else:
        assert abs(energies[49] - 2115.9320061795634) < 1


if __name__ == "__main__":

    test_rgb_image_hand_fitting_pytorch()
    test_rgb_image_hand_fitting_numpy()
    test_rgb_image_hand_fitting_tensorflow()
