"""Test using rgb mesh rendering."""

import hashlib
import os

import imageio.v3 as imageio
import numpy as np

import deodr
from deodr import differentiable_renderer_cython  # type: ignore
from deodr.examples.render_mesh import example_moderngl, example_rgb
from deodr.examples.triangle_soup_fitting import create_example_scene


def test_render_mesh_moderngl() -> None:
    if os.name != "nt":  # did not manage to install mesa on windows github action
        example_moderngl(display=False)


def test_render_mesh_duck(update: bool = False) -> None:
    image = example_rgb(display=False, save_image=False, width=320, height=240)
    image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
    image_uint8 = (image * 255).astype(np.uint8)
    if update:
        imageio.imwrite(image_file, image_uint8)
    image_prev = imageio.imread(image_file)
    assert np.max(np.abs(image_prev.astype(int) - image_uint8.astype(int))) == 0


def test_render_mesh_triangle_soup() -> None:
    np.random.seed(2)
    scene_gt = create_example_scene(clockwise=True)

    assert (
        hashlib.sha256(scene_gt.ij.tobytes()).hexdigest()
        == "56a498bf243bd514c9ab4a3bfd90f8105aa2c168023fa288dc39ad82e2d36a20"
    )
    assert (
        hashlib.sha256(scene_gt.depths.tobytes()).hexdigest()
        == "e25eed6310fef37e401aef594c4c95e1b3cccf962a3646976cf546c58ddfac0a"
    )
    assert (
        hashlib.sha256(scene_gt.uv.tobytes()).hexdigest()
        == "f436623445124ecff7139efa57cce21c2768e23727bac974e236ea33651cc7c9"
    )
    assert (
        hashlib.sha256(scene_gt.shade.tobytes()).hexdigest()
        == "4b796b925c4349245e52a3e6311e99d536dc71e8aa8dc43cbd67cbe35d48892f"
    )
    assert (
        hashlib.sha256(scene_gt.colors.tobytes()).hexdigest()
        == "76dbff728be3eb0860bd27adf493e935dbd81cd7232ec732ba30c4f73ea35c94"
    )

    sigma = 1
    image = np.zeros((scene_gt.height, scene_gt.width, scene_gt.nb_colors))
    z_buffer = np.zeros((scene_gt.height, scene_gt.width))
    differentiable_renderer_cython.renderSceneCpp(scene_gt, sigma, image, z_buffer)

    filename = os.path.join(os.path.dirname(__file__), "data", "triangle_soup.png")
    # imageio.imwrite(filename,image)

    image_lkg = imageio.imread(filename)
    assert np.max(np.abs(image_lkg - image * 255)) <= 1

    if os.name == "nt":  # windows
        assert (
            hashlib.sha256(image.tobytes()).hexdigest()
            == "4de52cc3e902f92ff64324b261ddc45cd6d148ec7e670cf2942532d515af62d8"
        )
        assert (
            hashlib.sha256(z_buffer.tobytes()).hexdigest()
            == "b6f87e03c60bd820efa09d0536495b25d5852f67ecbecd2622f8bf1910d6052a"
        )

    # elif os.name == "posix":  # linux
    #     # google colab  Intel(R) Xeon(R) CPU @ 2.20GHz
    #     assert (
    #         hashlib.sha256(image.tobytes()).hexdigest()
    #         == "ee530428ecac0a11880aa942e92e40515cdebf86a5e6dd7aadc99b8dcaaf11a6"
    #     )
    #     assert (
    #         hashlib.sha256(z_buffer.tobytes()).hexdigest()
    #         == "b6f87e03c60bd820efa09d0536495b25d5852f67ecbecd2622f8bf1910d6052a"
    #     )

    # else:
    #     raise BaseException(f"No results for os.name={os.name}")


if __name__ == "__main__":
    test_render_mesh_duck()
    test_render_mesh_triangle_soup()
    test_render_mesh_moderngl()
