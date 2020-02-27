from deodr.triangulated_mesh import ColoredTriMesh
from deodr import differentiable_renderer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio
import trimesh
import deodr


def run(obj_file, width=640, height=480, display=True):
    render_mesh(obj_file, width=width, height=height, display=display)


def render_mesh(obj_file, width=640, height=480, display=True, display_pyrender=True):

    mesh_trimesh = trimesh.load(obj_file)

    mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)

    ax = plt.subplot(111)
    if mesh.textured:
        mesh.plot_uv_map(ax)

    rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    camera = differentiable_renderer.default_camera(
        width, height, 80, mesh.vertices, rot
    )
    bg_color=np.array((0,0.2,0))
    scene = differentiable_renderer.Scene3D()
    ambiant_light = 1
    directional_intensity=0
    ligth_directional = np.array([-0, 0, -1])*directional_intensity
    scene.set_light(ligth_directional=ligth_directional, ambiant_light=ambiant_light)
    scene.set_mesh(mesh)
    background_image = np.ones((height, width, 3))*bg_color
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

       

    if display_pyrender:
        import pyrender

        keep = (channels["faceid"] > 0).flatten()
        pts = channels["xyz"].reshape(-1, 3)[keep, :]
        colors = image.reshape(-1, 3)[keep, :]
        pyrender_scene = pyrender.Scene(
            ambient_light=ambiant_light*np.ones((3)), bg_color=bg_color
        )

        # cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        cam = pyrender.IntrinsicsCamera(
            fx=camera.intrinsic[0, 0],
            fy=camera.intrinsic[1, 1],
            cx=camera.intrinsic[0, 2],
            cy=camera.intrinsic[1, 2],
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
        point_cloud = pyrender.Mesh.from_points(pts, colors=colors)
        directional_light = pyrender.light.DirectionalLight(intensity=np.linalg.norm(ligth_directional))
        
        pyrender_scene.add(pyrender_mesh, pose=np.eye(4))
        pyrender_scene.add(point_cloud, pose=np.eye(4))
        m = camera.camera_to_world_mtx_4x4()
        pose_camera = np.column_stack((np.diag([1, -1, -1, 1]).dot(m[:, :3]), m[:, 3]))#not sure why this is so comlex
        pyrender_scene.add(directional_light, pose=np.diag([1,1,1,1]))
        pyrender_scene.add(cam, pose=pose_camera)

        r=pyrender.offscreen.OffscreenRenderer(width, height, point_size=0.0)
        image_pyrender, depth = r.render(pyrender_scene)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.subplot(1,3,2)
        plt.imshow(image_pyrender)
        plt.subplot(1,3,3)        
        plt.imshow(np.abs(image-image_pyrender.astype(np.float)/255))
        #pyrender.Viewer(
        #   pyrender_scene, use_raymond_lighting=True, viewport_size=(width, height)
        #)
    if display or display_pyrender:
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
    example(save_image=False)
