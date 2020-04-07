"""example fitting a texture mesh to an image using face images"""

import cv2

from deodr.differentiable_renderer import Scene2D

import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial import Delaunay

import sklearn.datasets
from sklearn import decomposition


faces = sklearn.datasets.fetch_olivetti_faces()

# plt.imshow(faces.images[0])

faces_pca = decomposition.PCA(n_components=150, whiten=True)
faces_pca.fit(faces.data)
plt.imshow(faces_pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.bone)

# coefs = faces_pca.transform(faces.data)
# print(faces)

# faces_pca.inverse_transform(coefs[[0]])


# create a regular grid and a triangulation  of the 2D image
N = 5
points = np.column_stack(
    [t.flatten() for t in np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N)]
)
tri = Delaunay(points)
triangles = tri.simplices.astype(np.uint32)
plt.triplot(points[:, 0], points[:, 1], triangles)
on_border = np.any((points == 0) | (points == 1), axis=1)

# deform the grid
max_displacement = 0.5
np.random.seed(0)
points_deformed_gt = (
    points + (np.random.rand(*points.shape) - 0.5) * max_displacement / N
)
points_deformed_gt[on_border] = points[on_border]
plt.triplot(points_deformed_gt[:, 0], points_deformed_gt[:, 1], triangles)
# plt.show()
nb_points = points.shape[0]
nb_triangles = triangles.shape[0]
ij = points_deformed_gt * 64 - 0.5
uv = points * 64 + 0.5
textured = np.ones((nb_triangles), dtype=np.bool)
shaded = np.ones((nb_triangles), dtype=np.bool)
depths = np.ones((nb_points))
shade = np.ones((nb_points))
colors = np.ones((nb_points, 1))
edgeflags = np.zeros((nb_triangles, 3), dtype=np.bool)
height = 64
width = 64
nb_colors = 1
texture = faces.images[10][:, :, None]
background = np.zeros((height, width, 1))

scene_gt = Scene2D(
    triangles,
    triangles,
    ij,
    depths,
    textured,
    uv,
    shade,
    colors,
    shaded,
    edgeflags,
    height,
    width,
    nb_colors,
    texture,
    background,
)

image_gt, _ = scene_gt.render(sigma=1)
plt.subplot(3, 1, 1)
plt.imshow(np.squeeze(texture, axis=2))
plt.subplot(3, 1, 2)
plt.imshow(np.squeeze(image_gt, axis=2))
plt.subplot(3, 1, 3)
plt.imshow(np.squeeze(np.abs(image_gt - texture), axis=2))

# plt.show()
np.max(texture - image_gt)


scene = Scene2D(
    triangles,
    triangles,
    ij,
    depths,
    textured,
    uv,
    shade,
    colors,
    shaded,
    edgeflags,
    height,
    width,
    nb_colors,
    texture,
    background,
)
# cv2.namedWindow('animation',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('animation', 600,600)

rescale_factor = 10


def fun(points_deformed, pca_coefs):
    ij = points_deformed * 64 - 0.5
    # face=faces_pca.inverse_transform(coefs[:,None])
    face = (faces_pca.mean_ + pca_coefs.dot(faces_pca.components_)).reshape((64, 64))
    scene.ij = ij
    scene.texture = face[:, :, None]
    print("render")
    image, z_buffer, diff_image, err = scene.render_compare_and_backward(
        obs=image_gt, sigma=1
    )
    print("np.max(np.abs(scene.ij_b))=%f" % np.max(np.abs(scene.ij_b)))
    print("E=%f" % err)
    print("done")
    image_gt_zoomed = cv2.resize(
        (image_gt.copy() * 255).astype(np.uint8),
        None,
        fx=rescale_factor,
        fy=rescale_factor,
        interpolation=cv2.INTER_NEAREST,
    )
    image_zoomed = cv2.resize(
        (image.copy() * 255).astype(np.uint8),
        None,
        fx=rescale_factor,
        fy=rescale_factor,
        interpolation=cv2.INTER_NEAREST,
    )
    diff_image_zoomed = cv2.resize(
        (np.abs(diff_image) * 255).astype(np.uint8),
        None,
        fx=rescale_factor,
        fy=rescale_factor,
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.polylines(
        image_gt_zoomed,
        (points_deformed_gt[tri.simplices] * 64 * rescale_factor).astype(np.int32),
        is_closed=True,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
    )

    cv2.polylines(
        image_zoomed,
        (points_deformed[tri.simplices] * 64 * rescale_factor).astype(np.int32),
        is_closed=True,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
    )
    cv2.imshow(
        "animation", np.column_stack((image_gt_zoomed, image_zoomed, diff_image_zoomed))
    )
    cv2.waitKey(1)

    # get gradient on pca coefs
    coefs_grad = faces_pca.components_.dot(scene.texture_b.flatten())
    points_deformed_grad = scene.ij_b * 64
    print(np.max(np.abs(points_deformed_grad)))
    grads = {"points_deformed": points_deformed_grad, "pca_coefs": coefs_grad}
    return err, grads


nb_iter = 100

pca_coefs = np.zeros((faces_pca.n_components))
points_deformed = points.copy()


variables = {"points_deformed": points_deformed, "pca_coefs": pca_coefs}
lambdas = {"points_deformed": 0.0001, "pca_coefs": 0.5}


for niter in range(nb_iter):

    E, grads = fun(**variables)
    print(f"iter{niter} E={E}")
    for name in variables.keys():
        variables[name] = variables[name] - lambdas[name] * grads[name]

    variables["points_deformed"][on_border] = points[on_border]

cv2.waitKey()
