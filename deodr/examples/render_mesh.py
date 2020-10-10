"""Examples with 3D mesh rendering using various backend and comparison with deodr."""

import os

import deodr
from deodr import differentiable_renderer
from deodr.triangulated_mesh import ColoredTriMesh

import imageio

import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

import trimesh


def run(obj_file, width=640, height=480, display=True):
    example_rgb(
        obj_file, width=width, height=height, display=display, display_moderngl=True
    )


def default_scene(obj_file, width=640, height=480, use_distortion=True):

    mesh_trimesh = trimesh.load(obj_file)

    mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)

    # rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
    rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()

    camera = differentiable_renderer.default_camera(
        width, height, 80, mesh.vertices, rot
    )
    if use_distortion:
        camera.distortion = np.array([-0.5, 0.5, 0, 0, 0])

    bg_color = np.array((0.8, 0.8, 0.8))
    scene = differentiable_renderer.Scene3D()
    light_ambient = 0
    light_directional = 0.3 * np.array([1, -1, 0])
    scene.set_light(light_directional=light_directional, light_ambient=light_ambient)
    scene.set_mesh(mesh)
    background_image = np.ones((height, width, 3)) * bg_color
    scene.set_background(background_image)
    return scene, camera


def example_rgb(display=True, save_image=False, width=640, height=480):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=width, height=height)
    image = scene.render(camera)
    if save_image:
        image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        image_uint8 = (image * 255).astype(np.uint8)
        imageio.imwrite(image_file, image_uint8)
    if display:
        plt.figure()
        plt.title("deodr rendering")
        plt.imshow(image)
    return image


def example_channels(display=True, save_image=False, width=640, height=480):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=width, height=height)

    def normalize(v):
        if v.ndim == 3 and v.shape[2] < 3:
            nv = np.zeros((v.shape[0], v.shape[1], 3))
            nv[:, :, : v.shape[2]] = v
        else:
            nv = v
        return (nv - nv.min()) / (nv.max() - nv.min())

    scene.sigma = 0

    channels = scene.render_deferred(camera)
    if display:
        plt.figure()
        for i, (name, v) in enumerate(channels.items()):
            ax = plt.subplot(2, 4, i + 1)
            ax.set_title(name)
            ax.imshow(normalize(v))

    if save_image:
        for name, v in channels.items():
            image_file = os.path.abspath(
                os.path.join(deodr.data_path, f"test/duck_{name}.png")
            )
            os.makedirs(os.path.dirname(image_file), exist_ok=True)
            image_uint8 = (normalize(v) * 255).astype(np.uint8)
            imageio.imwrite(image_file, image_uint8)


def example_pyrender(display=True, save_image=False, width=640, height=480):
    import deodr.opengl.pyrender

    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(
        obj_file, use_distortion=False, width=width, height=height
    )
    scene.sigma = 0  # removing edge overdraw antialiasing
    image_no_antialiasing = scene.render(camera)
    image_pyrender, depth = deodr.opengl.pyrender.render(scene, camera)
    if display:
        plt.figure()
        ax = plt.subplot(1, 3, 1)
        ax.set_title("deodr no antialiasing")
        ax.imshow(image_no_antialiasing)

        ax = plt.subplot(1, 3, 2)
        ax.set_title("pyrender")
        ax.imshow(image_pyrender)

        ax = plt.subplot(1, 3, 3)
        ax.set_title("difference")
        ax.imshow(np.abs(image_no_antialiasing - image_pyrender.astype(np.float) / 255))


def example_moderngl(display=True, width=640, height=480):
    import deodr.opengl.moderngl

    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=width, height=height)
    scene.sigma = 0  # removing edge overdraw antialiasing
    # adding some perturbation to get better test
    camera.extrinsic[1, 2] = camera.extrinsic[1, 2] + 0.1
    camera.extrinsic[1, 1] = camera.extrinsic[1, 1] * 0.9

    image_no_antialiasing = scene.render(camera)
    moderngl_renderer = deodr.opengl.moderngl.OffscreenRenderer()
    moderngl_renderer.set_scene(scene)
    image_moderngl = moderngl_renderer.render(camera)
    diff = np.abs(
        image_no_antialiasing.astype(np.float) * 255 - image_moderngl.astype(np.float)
    )
    if display:
        plt.figure()
        ax = plt.subplot(1, 3, 1)
        ax.set_title("deodr no antialiasing")
        ax.imshow(image_no_antialiasing)

        ax = plt.subplot(1, 3, 2)
        ax.set_title("moderngl")
        ax.imshow(image_moderngl)

        ax = plt.subplot(1, 3, 3)
        ax.set_title("difference")
        ax.imshow(10 * diff / 255)
        plt.show()

    max_diff = np.max(diff)

    print(f"max_diff between deodr and moderngl rendering = {max_diff}")
    assert max_diff < 18


if __name__ == "__main__":
    example_moderngl(display=True)
    example_rgb(save_image=False)
    example_channels(save_image=False)
    example_pyrender()
    plt.show()
