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


def render_mesh(obj_file, SizeW=640, SizeH=480, display=True):

    mesh = loadmesh(obj_file)

    ax = plt.subplot(111)
    if mesh.textured:
        mesh.plot_uv_map(ax)

    objectCenter = 0.5 * (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0))
    objectRadius = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    cameraCenter = objectCenter + np.array([0, 0, 3]) * objectRadius
    focal = 2 * SizeW

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T = -R.T.dot(cameraCenter)
    extrinsic = np.column_stack((R, T))
    intrinsic = np.array([[focal, 0, SizeW / 2], [0, focal, SizeH / 2], [0, 0, 1]])
    dist = [-2, 0, 0, 0, 0]
    camera = differentiable_renderer.Camera(
        extrinsic=extrinsic, intrinsic=intrinsic, dist=dist, resolution=(SizeW, SizeH)
    )

    scene = differentiable_renderer.Scene3D()
    scene.setLight(ligthDirectional=np.array([-0.5, 0, -0.5]), ambiantLight=0.3)
    scene.setMesh(mesh)
    backgroundImage = np.ones((SizeH, SizeW, 3))
    scene.setBackground(backgroundImage)

    Abuffer = scene.render(camera)
    if display:
        plt.figure()
        plt.imshow(Abuffer)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)
        mesh.plot(ax, plot_normals=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        u, v, w = scene.ligthDirectional
        ax.quiver(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([u]),
            np.array([v]),
            np.array([w]),
            color=[1, 1, 0.5],
        )

    channels = scene.renderDeffered(camera)
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
    return Abuffer, channels


def run(save_image=False):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    Abuffer, channels = render_mesh(obj_file, SizeW=320, SizeH=240)
    image_file = os.path.abspath(os.path.join(deodr.data_path, "/test/duck.png"))
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    Abuffer_uint8 = (Abuffer * 255).astype(np.uint8)
    imageio.imwrite(image_file, Abuffer_uint8)


if __name__ == "__main__":
    run(save_image=False)
