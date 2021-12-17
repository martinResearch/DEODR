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
        )
        # 2100.0239709048583 : result on Intel(R) Xeon(R) W-2155 CPU @ 3.30GHz   3.31GHz
        # 2132.9307950405196 : result on Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz   2.40GHz
        # Possible explanation https://github.com/pytorch/pytorch/issues/54684
    else:
        assert abs(energies[49] - 2106.5436357944604) < 12
        # google colab 2117.9946156293213


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
        # google colab 2107.850380422819


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
        # Running on CPU because it seems to change a lot
        # from one run to the next when using GPU
        # could use os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # github action 2132.9307950405196

    else:  # posix if linux
        assert (abs(energies[49] - 2115.9320061795634) < 1e-10) or (
            abs(energies[49] - 2107.962374538259) < 1e-10
        )
        # google colab 2107.962374538259


if __name__ == "__main__":

    test_rgb_image_hand_fitting_pytorch()
    test_rgb_image_hand_fitting_numpy()
    test_rgb_image_hand_fitting_tensorflow()
