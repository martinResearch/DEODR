from deodr.triangulated_mesh import ColoredTriMesh
from deodr import differentiable_renderer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio
import trimesh
import deodr


def loadmesh(file):

    mesh_trimesh = trimesh.load(file)
    return ColoredTriMesh.from_trimesh(mesh_trimesh)


def run(obj_file, width=640, height=480, display=True):
    render_mesh(obj_file, width=width, height=height, display=display)


def render_mesh(obj_file, width=640, height=480, display=True):

    mesh = loadmesh(obj_file)

    ax = plt.subplot(111)
    if mesh.textured:
        mesh.plot_uv_map(ax)

    object_center = 0.5 * (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0))
    object_radius = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    camera_center = object_center + np.array([0, 0, 3]) * object_radius
    focal = 2 * width

    rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    trans = -rot.T.dot(camera_center)
    extrinsic = np.column_stack((rot, trans))
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    dist = [-2, 0, 0, 0, 0]
    camera = differentiable_renderer.Camera(
        extrinsic=extrinsic, intrinsic=intrinsic, dist=dist, resolution=(width, height)
    )

    scene = differentiable_renderer.Scene3D()
    scene.set_light(ligth_directional=np.array([-0.5, 0, -0.5]), ambiant_light=0.3)
    scene.set_mesh(mesh)
    background_image = np.ones((height, width, 3))
    scene.set_background(background_image)

    image = scene.render(camera)
    if display:
        plt.figure()
        plt.imshow(image)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)
        mesh.plot(ax, plot_normals=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        u, v, w = scene.ligth_directional
        ax.quiver(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([u]),
            np.array([v]),
            np.array([w]),
            color=[1, 1, 0.5],
        )

    channels = scene.render_deffered(camera)
    if display:
        plt.figure()
        for i, (name, v) in enumerate(channels.items()):
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title(name)
            if v.ndim == 3 and v.shape[2] < 3:
                nv = np.zeros((v.shape[0], v.shape[1], 3))
                nv[:, :, : v.shape[2]] = v
                ax.imshow((nv - nv.min()) / (nv.max() - nv.min()))
            else:
                ax.imshow((v - v.min()) / (v.max() - v.min()))

        plt.show()
    return image, channels


def example(save_image=False):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    image, channels = render_mesh(obj_file, width=320, height=240)
    image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    image_uint8 = (image * 255).astype(np.uint8)
    imageio.imwrite(image_file, image_uint8)


if __name__ == "__main__":
    run(save_image=False)
