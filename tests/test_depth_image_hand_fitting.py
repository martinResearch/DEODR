"""Test using depth image hand fitting."""

from deodr.examples.depth_image_hand_fitting import run


def test_depth_image_hand_fitting_pytorch():

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    expected = 253.03801096040416
    assert (
        abs(energies[49] - expected) < 1e-5
    ), f"Getting {energies[49]} instead of {expected}"


def test_depth_image_hand_fitting_numpy():

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )

    expected = 253.03801111458935
    assert (
        abs(energies[49] - expected) < 1e-5
    ), f"Getting {energies[49]} instead of {expected}"


def test_depth_image_hand_fitting_tensorflow():

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )

    expected = 253.03801088397518
    assert (
        abs(energies[49] - expected) < 1e-5
    ), f"Getting {energies[49]} instead of {expected}"


if __name__ == "__main__":

    test_depth_image_hand_fitting_pytorch()
    test_depth_image_hand_fitting_numpy()
    test_depth_image_hand_fitting_tensorflow()
