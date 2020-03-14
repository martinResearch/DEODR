from deodr.examples.depth_image_hand_fitting import run


def test_depth_image_hand_fitting_pytorch():

    energies = run(
        dl_library="pytorch",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 252.83065023526686) < 1e-5


def test_depth_image_hand_fitting_numpy():

    energies = run(
        dl_library="none",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 253.03801111458935) < 1e-5


def test_depth_image_hand_fitting_tensorflow():

    energies = run(
        dl_library="tensorflow",
        plot_curves=False,
        display=False,
        save_images=False,
        max_iter=50,
    )
    assert abs(energies[49] - 252.830650232813) < 1e-5


if __name__ == "__main__":

    test_depth_image_hand_fitting_pytorch()
    test_depth_image_hand_fitting_numpy()
    test_depth_image_hand_fitting_tensorflow()
