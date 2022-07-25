"""Test using depth image hand fitting."""
import os

from deodr.examples.depth_image_hand_fitting import run

import tensorflow as tf


def test_depth_image_hand_fitting_pytorch() -> None:

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
        assert (abs(energies[49] - 251.31652686512888) < 1e-10) or (
            abs(energies[49] - 251.31652686495823) < 1e-10
        )
        # github worklofw 251.31652686495823, 251.31652686512888


def test_depth_image_hand_fitting_numpy() -> None:

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
        assert (abs(energies[49] - 251.32711113730954) < 1e-10) or (
            abs(energies[49] - 251.3271111242092) < 1e-10
        )
        # github workflow 251.32711113730954, 251.3271111242092


def test_depth_image_hand_fitting_tensorflow() -> None:

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
    elif os.name == "posix":  # linux
        assert (
            (abs(energies[49] - 251.31648932312913) < 1e-10)
            or (abs(energies[49] - 251.3164914350016) < 1e-10)
            or (abs(energies[49] - 251.3164891265543) < 1e-10)
        )
        # google colab Intel(R) Xeon(R) CPU @ 2.20GHz: 251.3164914350016
        # Gitbub workflow 251.31648932312913, 251.3164891265543
    else:
        raise BaseException(f"No results for os.name={os.name}")


if __name__ == "__main__":
    test_depth_image_hand_fitting_pytorch()
    test_depth_image_hand_fitting_numpy()

    test_depth_image_hand_fitting_tensorflow()
