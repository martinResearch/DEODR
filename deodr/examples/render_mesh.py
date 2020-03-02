from deodr.triangulated_mesh import ColoredTriMesh
from deodr import differentiable_renderer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio
import trimesh
import deodr
import pyrender
from scipy.spatial.transform import Rotation


def run(obj_file, width=640, height=480, display=True):
    render_mesh(
        obj_file, width=width, height=height, display=display, display_moderngl=True
    )


def default_scene(obj_file, width=320, height=240, use_distortion=True):

    mesh_trimesh = trimesh.load(obj_file)
    pyrender.Mesh.from_trimesh(mesh_trimesh)

    mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)

    # rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_dcm()
    rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_dcm()

    camera = differentiable_renderer.default_camera(
        width, height, 80, mesh.vertices, rot
    )
    if use_distortion:
        camera.distortion = np.array([-0.5, 0.5, 0, 0, 0])

    bg_color = np.array((0.8, 0.8, 0.8))
    scene = differentiable_renderer.Scene3D()
    ambiant_light = 0
    ligth_directional = 0.3 * np.array([1, -1, 0])
    scene.set_light(ligth_directional=ligth_directional, ambiant_light=ambiant_light)
    scene.set_mesh(mesh)
    background_image = np.ones((height, width, 3)) * bg_color
    scene.set_background(background_image)
    return scene, camera


def example_rgb(display=True, save_image=False):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=640, height=480)
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


def example_channels(display=True, save_image=False):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=640, height=480)

    def normalize(v):
        if v.ndim == 3 and v.shape[2] < 3:
            nv = np.zeros((v.shape[0], v.shape[1], 3))
            nv[:, :, : v.shape[2]] = v
        else:
            nv = v
        return (nv - nv.min()) / (nv.max() - nv.min())

    channels = scene.render_deffered(camera)
    if display:
        plt.figure()
        for i, (name, v) in enumerate(channels.items()):
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title(name)
            ax.imshow(normalize(v))

    if save_image:
        for i, (name, v) in enumerate(channels.items()):
            image_file = os.path.abspath(
                os.path.join(deodr.data_path, f"test/duck_{name}.png")
            )
            os.makedirs(os.path.dirname(image_file), exist_ok=True)
            image_uint8 = (normalize(v) * 255).astype(np.uint8)
            imageio.imwrite(image_file, image_uint8)


def example_pyrender(display=True, save_image=False):
    import deodr.opengl.pyrender

    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, use_distortion=False)
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


def example_moderngl(display=True):
    import deodr.opengl.moderngl

    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file)
    scene.sigma = 0  # removing edge overdraw antialiasing
    image_no_antialiasing = scene.render(camera)
    renderer = deodr.opengl.moderngl.OffscreenRenderer()
    image_moderngl = renderer.render(scene, camera)
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
        ax.imshow(
            10 * np.abs(image_no_antialiasing - image_moderngl.astype(np.float) / 255)
        )
    assert (
        np.max(np.abs(image_no_antialiasing - image_moderngl.astype(np.float) / 255))
        < 1
    )


if __name__ == "__main__":
    example_rgb(save_image=True)
    example_channels(save_image=True)
    example_moderngl()
    example_pyrender()
    plt.show()
