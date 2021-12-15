"""Test using rgb_image hand fitting."""

import os
from deodr.examples.rgb_image_hand_fitting import run


def test_rgb_image_hand_fitting_pytorch():

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    if os.name == "nt":  # windows
        assert abs(energies[49] - 2100.0239709048583) < 1e-10
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
        assert (abs(energies[49] - 2107.850380422819) < 2) or (
            abs(energies[49] - 2113.7013184079137) < 2
        )


def test_rgb_image_hand_fitting_tensorflow():

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    if os.name == "nt":  # windows
        assert (
            abs(energies[49] - 2105.598355803427) < 1e-10
        )  # seems to change a lot from one run to the next
    else:
        assert abs(energies[49] - 2115.9320061795634) < 1


if __name__ == "__main__":

    test_rgb_image_hand_fitting_pytorch()
    test_rgb_image_hand_fitting_numpy()
    test_rgb_image_hand_fitting_tensorflow()
