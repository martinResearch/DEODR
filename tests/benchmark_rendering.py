import numpy as np


from deodr.examples.triangle_soup_fitting import create_example_scene
from deodr import differentiable_renderer_cython  # type: ignore
import matplotlib.pyplot as plt

import time


def benchmark_render_mesh_triangle_soup() -> None:
    np.random.seed(2)
    scene_gt = create_example_scene(
        n_tri=200, width=500, height=500, clockwise=True, textured_ratio=0
    )

    image = np.zeros((scene_gt.height, scene_gt.width, scene_gt.nb_colors))
    z_buffer = np.zeros((scene_gt.height, scene_gt.width))
    sigma = 0
    durations = []
    for i in range(1000):
        image.fill(0)
        z_buffer.fill(0)
        start = time.perf_counter_ns()
        differentiable_renderer_cython.renderScene(scene_gt, sigma, image, z_buffer)
        elapsed = time.perf_counter_ns() - start
        durations.append(elapsed)
    print(np.median(durations) / 1e9)
    # plt.imshow(image)
    # plt.show()


if __name__ == "__main__":
    benchmark_render_mesh_triangle_soup()
