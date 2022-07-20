"""Test using rgb_image hand fitting."""

from deodr.examples.rgb_image_hand_fitting import run


def test_rgb_image_hand_fitting_pytorch() -> None:

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 2106.5436357944604) < 12


def test_rgb_image_hand_fitting_numpy() -> None:

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    # getting different result on python 3.6 or 3.7 in the github action, not sure why
    assert (abs(energies[49] - 2107.850380422819) < 2) or (
        abs(energies[49] - 2113.7013184079137) < 2
    )


def test_rgb_image_hand_fitting_tensorflow() -> None:

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 2115.9320061795634) < 1


if __name__ == "__main__":

    test_rgb_image_hand_fitting_pytorch()
    test_rgb_image_hand_fitting_numpy()
    test_rgb_image_hand_fitting_tensorflow()
