"""Example with fitting a 32 triangles soup to an image."""


import copy
import os

import cv2

import deodr
from deodr import differentiable_renderer_cython
from deodr.differentiable_renderer import Scene2D

from imageio import imread

import matplotlib.pyplot as plt

import numpy as np


def create_example_scene(n_tri=30, width=200, height=200):

    material = np.double(imread(os.path.join(deodr.data_path, "trefle.jpg"))) / 255
    height_material = material.shape[0]
    width_material = material.shape[1]

    scale_matrix = np.array([[height, 0], [0, width]])
    scale_material = np.array([[height_material - 1, 0], [0, width_material - 1]])

    triangles = []
    for _ in range(n_tri):

        tmp = scale_matrix.dot(
            np.random.rand(2, 1).dot(np.ones((1, 3)))
            + 0.5 * (-0.5 + np.random.rand(2, 3))
        )
        while np.abs(np.linalg.det(np.vstack((tmp, np.ones((3)))))) < 1500:
            tmp = scale_matrix.dot(
                np.random.rand(2, 1).dot(np.ones((1, 3)))
                + 0.5 * (-0.5 + np.random.rand(2, 3))
            )

        if np.linalg.det(np.vstack((tmp, np.ones((3))))) > 0:
            tmp = np.fliplr(tmp)
        triangle = {}
        triangle["ij"] = tmp.T
        triangle["depths"] = np.random.rand(1) * np.ones(
            (3, 1)
        )  # constant depth triangles to avoid collisions
        triangle["textured"] = np.random.rand(1) > 0.5

        if triangle["textured"]:
            triangle["uv"] = (
                scale_material.dot(np.array([[0, 1, 0.2], [0, 0.2, 1]])).T + 1
            )  # texture coordinate of the vertices
            triangle["shade"] = np.random.rand(3, 1)  # shade  intensity at each vertex
            triangle["colors"] = np.zeros((3, 3))
            triangle["shaded"] = True
        else:
            triangle["uv"] = np.zeros((3, 2))
            triangle["shade"] = np.zeros((3, 1))
            triangle["colors"] = np.random.rand(3, 3)
            # colors of the vertices (can be gray, rgb color,or even other dimension
            # vectors) when using simple linear interpolation across triangles
            triangle["shaded"] = False

        triangle["edgeflags"] = np.array(
            [True, True, True]
        )  # all edges are discontinuity edges as no triangle pair share an edge
        triangles.append(triangle)

    scene = {}
    for key in triangles[0].keys():
        scene[key] = np.squeeze(
            np.vstack([np.array(triangle[key]) for triangle in triangles])
        )
    scene["faces"] = np.arange(3 * n_tri).reshape(-1, 3).astype(np.uint32)
    scene["faces_uv"] = np.arange(3 * n_tri).reshape(-1, 3).astype(np.uint32)

    scene["height"] = height
    scene["width"] = width
    scene["texture"] = material
    scene["nb_colors"] = 3
    scene["background"] = np.tile(
        np.array([0.3, 0.5, 0.7])[None, None, :], (height, width, 1)
    )
    return Scene2D(**scene)


def run(nb_max_iter=500, display=True):
    print("process id=%d" % os.getpid())

    np.random.seed(2)
    scene_gt = create_example_scene()
    antialiase_error = False
    sigma = 1

    image_target = np.zeros((scene_gt.height, scene_gt.width, scene_gt.nb_colors))

    z_buffer = np.zeros((scene_gt.height, scene_gt.width))
    differentiable_renderer_cython.renderScene(scene_gt, sigma, image_target, z_buffer)

    n_vertices = len(scene_gt.depths)

    displacement_magnitude_ij = 10
    displacement_magnitude_uv = 0
    displacement_magnitude_colors = 0

    alpha_ij = 0.01
    beta_ij = 0.80
    alpha_uv = 0.03
    beta_uv = 0.80
    alpha_color = 0.001
    beta_color = 0.70

    max_uv = np.array(scene_gt.texture.shape[:2]) - 1

    scene_init = copy.deepcopy(scene_gt)
    scene_init.ij = (
        scene_gt.ij + np.random.randn(n_vertices, 2) * displacement_magnitude_ij
    )
    scene_init.uv = (
        scene_gt.uv + np.random.randn(n_vertices, 2) * displacement_magnitude_uv
    )
    scene_init.uv = np.maximum(scene_init.uv, 0)
    scene_init.uv = np.minimum(scene_init.uv, max_uv)
    scene_init.colors = (
        scene_gt.colors + np.random.randn(n_vertices, 3) * displacement_magnitude_colors
    )

    final_loss = {}

    for antialiase_error in [True, False]:
        np.random.seed(2)
        scene_iter = copy.deepcopy(scene_init)

        speed_ij = np.zeros((n_vertices, 2))
        speed_uv = np.zeros((n_vertices, 2))
        speed_color = np.zeros((n_vertices, 3))

        losses = []
        for niter in range(nb_max_iter):
            image, depth, loss_image, loss = scene_iter.render_compare_and_backward(
                sigma, antialiase_error, image_target
            )
            print(f"iter {niter} loss = {loss}")
            # imsave(os.path.join(iterfolder,f'soup_{niter}.png'), combinedIMage)

            losses.append(loss)
            if loss_image.ndim == 2:
                loss_image = np.broadcast_to(loss_image[:, :, None], image.shape)
            if display:
                cv2.waitKey(1)
                cv2.imshow(
                    "animation",
                    np.column_stack((image_target, image, loss_image))[:, :, ::-1],
                )

            if displacement_magnitude_ij > 0:
                speed_ij = beta_ij * speed_ij - scene_iter.ij_b * alpha_ij
                scene_iter.ij = scene_iter.ij + speed_ij

            if displacement_magnitude_colors > 0:
                speed_color = (
                    beta_color * speed_color - scene_iter.colors_b * alpha_color
                )
                scene_iter.colors = scene_iter.colors + speed_color

            if displacement_magnitude_uv > 0:
                speed_uv = beta_uv * speed_uv - scene_iter.uv_b * alpha_uv
                scene_iter.uv = scene_iter.uv + speed_uv
                scene_iter.uv = max(scene_iter.uv, 0)
                scene_iter.uv = min(scene_iter.uv, max_uv)
        if display:
            plt.plot(losses, label="antialiaseError=%d" % antialiase_error)
        final_loss[antialiase_error] = loss
    if display:
        plt.legend()
        plt.show()
    return final_loss


if __name__ == "__main__":
    run()
