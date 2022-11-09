"""Module to render deodr scenes using OpenGL through pyrender.

Note that pyrender does not support camera distortion.
"""
from typing import Optional, Tuple
import numpy as np

import pyrender

import trimesh
from deodr.differentiable_renderer import Camera, Scene3D

from deodr.triangulated_mesh import ColoredTriMesh


def arcsinc(x: float) -> float:
    return np.arcsin(x) / x if abs(x) > 1e-6 else 1


def min_rotation(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1,
                 aligns it with vec2.
    """
    assert vec1.shape == (3,)
    assert vec2.shape == (3,)
    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),
        (vec2 / np.linalg.norm(vec2)).reshape(3),
    )
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    d = (1 - c) / (s**2) if s > 1e-6 else 0.5
    return np.eye(3) + kmat + kmat.dot(kmat) * d


def deodr_directional_light_to_pyrender(
    deodr_directional_light: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    assert deodr_directional_light.shape == (3,)
    directional_light_intensity = np.linalg.norm(deodr_directional_light)
    if directional_light_intensity > 0:
        directional_light_direction = (
            deodr_directional_light / directional_light_intensity
        )
        directional_light_rotation = min_rotation(
            np.array([0, 0, -1]), directional_light_direction
        )
        pose = np.zeros((4, 4))
        pose[:3, :3] = directional_light_rotation
        pose[3, 3] = 1
        directional_light = pyrender.light.DirectionalLight(
            intensity=directional_light_intensity
        )
    else:
        directional_light = None
        pose = None

    return directional_light, pose


def deodr_mesh_to_pyrender(deodr_mesh: ColoredTriMesh) -> pyrender.Mesh:
    assert deodr_mesh.uv is not None
    assert deodr_mesh.texture is not None

    # trimesh and pyrender do to handle faces indices for texture
    # that are different from face indices for the 3d vertices
    # we need to duplicate vertices
    faces, mask_v, mask_vt = trimesh.visual.texture.unmerge_faces(
        deodr_mesh.faces, deodr_mesh.faces_uv
    )
    vertices = deodr_mesh.vertices[mask_v]
    deodr_mesh.compute_vertex_normals()
    vertex_normals = deodr_mesh.vertex_normals[mask_v]
    uv = deodr_mesh.uv[mask_vt]
    pyrender_uv = np.column_stack(
        (
            (uv[:, 0]) / deodr_mesh.texture.shape[0],
            (1 - uv[:, 1] / deodr_mesh.texture.shape[1]),
        )
    )

    material = None
    poses = None
    color_0 = None
    if material is None:
        base_color_texture = pyrender.texture.Texture(
            source=deodr_mesh.texture, source_channels="RGB"
        )
        material = pyrender.MetallicRoughnessMaterial(
            alphaMode="BLEND",
            baseColorFactor=[1, 1, 1, 1.0],
            metallicFactor=0,
            roughnessFactor=1,
            baseColorTexture=base_color_texture,
        )
        material.wireframe = False

    primitive = pyrender.Primitive(
        positions=vertices,
        normals=vertex_normals,
        texcoord_0=pyrender_uv,
        color_0=color_0,
        indices=faces,
        material=material,
        mode=pyrender.constants.GLTF.TRIANGLES,
        poses=poses,
    )

    return pyrender.Mesh(primitives=[primitive])


def render(deodr_scene: Scene3D, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    assert (
        deodr_scene.mesh is not None
    ), "You need a mesh in the scene you want to render"
    """Render a deodr scene using pyrender"""
    if deodr_scene.background_image is not None:
        bg_color = deodr_scene.background_image[0, 0]
        if not (np.all(deodr_scene.background_image == bg_color[None, None, :])):
            raise (
                BaseException(
                    "pyrender does not support background image, please provide a"
                    " background image that correspond to a uniform color"
                )
            )
    else:
        bg_color = deodr_scene.background_color
    pyrender_scene = pyrender.Scene(
        ambient_light=deodr_scene.light_ambient * np.ones((3)), bg_color=bg_color
    )

    if camera.distortion is not None:
        raise (BaseException("cameras with distortion not handled yet with pyrender"))

    # cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    cam = pyrender.IntrinsicsCamera(
        fx=camera.intrinsic[0, 0],
        fy=camera.intrinsic[1, 1],
        cx=camera.intrinsic[0, 2],
        cy=camera.intrinsic[1, 2],
    )

    # convert to pyrender directional light parameterization
    assert deodr_scene.light_directional is not None
    directional_light, directional_light_pose = deodr_directional_light_to_pyrender(
        deodr_scene.light_directional
    )
    if directional_light is not None:
        pyrender_scene.add(directional_light, pose=directional_light_pose)

    # convert the mesh
    pyrender_mesh = deodr_mesh_to_pyrender(deodr_scene.mesh)
    pyrender_scene.add(pyrender_mesh, pose=np.eye(4))

    # convert the camera
    m = camera.camera_to_world_mtx_4x4()
    pose_camera = m.dot(np.diag([1, -1, -1, 1]))

    pyrender_scene.add(cam, pose=pose_camera)

    # render
    width = camera.width
    height = camera.height
    r = pyrender.offscreen.OffscreenRenderer(width, height, point_size=1.0)
    image_pyrender, depth = r.render(pyrender_scene)
    return image_pyrender, depth
