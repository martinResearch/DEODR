"""Test using rgb_image hand fitting."""

import numpy as np
import pytest

from deodr.examples.rgb_image_hand_fitting import run


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
        2112.3014481160876,  # github action windows python 3.8
        2108.0875835193865,  # github action ubuntu python 3.7
        2121.835226209652,  # github action windows python 3.10
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
        2113.7013184079137,  # python 3.7 on windows on github action
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


@pytest.mark.skip(reason="Tensorflow does nto support numpy 2.0 yet")
def test_rgb_image_hand_fitting_tensorflow() -> None:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # Running on CPU to get deterministic results

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
        2115.791145850562,
        2114.564762503404,
        2102.9991446481354,
        2167.4449854352574,
    ]

    assert np.any(np.abs(np.array(possible_results) - energies[49]) < 1e-5)


if __name__ == "__main__":
    test_rgb_image_hand_fitting_tensorflow()
    test_rgb_image_hand_fitting_pytorch()

    test_rgb_image_hand_fitting_numpy()
