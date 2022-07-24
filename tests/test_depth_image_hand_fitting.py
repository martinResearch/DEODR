"""Test using depth image hand fitting."""

from deodr.examples.depth_image_hand_fitting import run


def test_depth_image_hand_fitting_pytorch() -> None:

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 251.31652686512888) < 2e-2


def test_depth_image_hand_fitting_numpy() -> None:

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 251.32711113730954) < 1e-2


def test_depth_image_hand_fitting_tensorflow() -> None:

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 251.31648983466366) < 2e-2


if __name__ == "__main__":
    test_depth_image_hand_fitting_numpy()
    test_depth_image_hand_fitting_pytorch()
    test_depth_image_hand_fitting_tensorflow()
