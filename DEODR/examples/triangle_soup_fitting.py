from DEODR import differentiable_renderer_cython
from DEODR.differentiable_renderer import Scene2D
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import os


def create_example_scene():

    Ntri = 30
    SizeW = 200
    SizeH = 200
    file_folder = os.path.dirname(os.path.abspath(__file__))
    material = np.double(imread(os.path.join(file_folder, "trefle.jpg"))) / 255
    Hmaterial = material.shape[0]
    Wmaterial = material.shape[1]

    scale_matrix = np.array([[SizeH, 0], [0, SizeW]])
    scale_material = np.array([[Hmaterial - 1, 0], [0, Wmaterial - 1]])

    triangles = []
    for k in range(Ntri):

        tmp = scale_matrix.dot(
            np.random.rand(2, 1).dot(np.ones((1, 3)))
            + 0.5 * (-0.5 + np.random.rand(2, 3))
        )
        while np.abs(np.linalg.det(np.vstack((tmp, np.ones((3)))))) < 1500:
            tmp = scale_matrix.dot(
                np.random.rand(2, 1).dot(np.ones((1, 3)))
                + 0.5 * (-0.5 + np.random.rand(2, 3))
            )

        if np.linalg.det(np.vstack((tmp, np.ones((3))))) < 0:
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
            triangle["colors"] = np.random.rand(
                3, 3
            )  # colors of the vertices (can be gray, rgb color,or even other dimension vectors) when using simple linear interpolation across triangles
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
    scene["faces"] = np.arange(3 * Ntri).reshape(-1, 3).astype(np.uint32)
    scene["faces_uv"] = np.arange(3 * Ntri).reshape(-1, 3).astype(np.uint32)

    scene["image_H"] = SizeH
    scene["image_W"] = SizeW
    scene["texture"] = material
    scene["nbColors"] = 3
    scene["background"] = np.tile(
        np.array([0.3, 0.5, 0.7])[None, None, :], (SizeH, SizeW, 1)
    )
    return Scene2D(**scene)


def main():
    print("process id=%d" % os.getpid())

    display = True
    np.random.seed(2)
    scene_gt = create_example_scene()
    display = True
    save_images = True
    antialiaseError = False
    sigma = 1

    AbufferTarget = np.zeros((scene_gt.image_H, scene_gt.image_W, scene_gt.nbColors))

    Zbuffer = np.zeros((scene_gt.image_H, scene_gt.image_W))
    differentiable_renderer_cython.renderScene(scene_gt, sigma, AbufferTarget, Zbuffer)

    Nvertices = len(scene_gt.depths)

    displacement_magnitude_ij = 10
    displacement_magnitude_uv = 0
    displacement_magnitude_colors = 0

    alpha_ij = 0.01
    beta_ij = 0.80
    alpha_uv = 0.03
    beta_uv = 0.80
    alpha_color = 0.001
    beta_color = 0.70
    scale_material = np.array(
        [[scene_gt.texture.shape[0] - 1, 0], [0, scene_gt.texture.shape[1] - 1]]
    )
    max_uv = np.array(scene_gt.texture.shape[:2]) - 1

    scene_init = copy.deepcopy(scene_gt)
    scene_init.ij = (
        scene_gt.ij + np.random.randn(Nvertices, 2) * displacement_magnitude_ij
    )
    scene_init.uv = (
        scene_gt.uv + np.random.randn(Nvertices, 2) * displacement_magnitude_uv
    )
    scene_init.uv = np.maximum(scene_init.uv, 0)
    scene_init.uv = np.minimum(scene_init.uv, max_uv)
    scene_init.colors = (
        scene_gt.colors + np.random.randn(Nvertices, 3) * displacement_magnitude_colors
    )

    for antialiaseError in [True, False]:
        np.random.seed(2)
        scene_iter = copy.deepcopy(scene_init)

        speed_ij = np.zeros((Nvertices, 2))
        speed_uv = np.zeros((Nvertices, 2))
        speed_color = np.zeros((Nvertices, 3))

        nbMaxIter = 500
        losses = []
        for iter in range(nbMaxIter):
            Image, depth, lossImage, loss = scene_iter.render_compare_and_backward(
                sigma, antialiaseError, AbufferTarget
            )

            # imsave(os.path.join(iterfolder,f'soup_{iter}.png'), combinedIMage)
            key = cv2.waitKey(1)
            losses.append(loss)
            if lossImage.ndim == 2:
                lossImage = np.broadcast_to(lossImage[:, :, None], Image.shape)
            cv2.imshow(
                "animation",
                np.column_stack((AbufferTarget, Image, lossImage))[:, :, ::-1],
            )

            if displacement_magnitude_ij > 0:
                speed_ij = beta_ij * speed_ij - scene_iter.ij_b * alpha_ij
                scene_iter.ij = scene_iter.ij + speed_ij

            if displacement_magnitude_colors > 0:
                speed_color = beta_color * speed_color - scene2b.colors_b * alpha_color
                scene_iter.colors = scene_iter.colors + speed_color

            if displacement_magnitude_uv > 0:
                speed_uv = beta_uv * speed_uv - scene2b.uv_b * alpha_uv
                scene_iter.uv = scene_iter.uv + speed_uv
                scene_iter.uv = max(scene_iter.uv, 0)
                scene_iter.uv = min(scene_iter.uv, max_uv)
        plt.plot(losses, label="antialiaseError=%d" % antialiaseError)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
