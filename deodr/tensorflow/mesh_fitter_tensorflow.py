"""Modules containing Tensorflow classes to fit 3D meshes to images using differentiable rendering."""
from typing import Optional, Tuple, Union

import copy

import numpy as np

import scipy.sparse.linalg
import scipy.spatial.transform.rotation

import tensorflow as tf

from . import (
    CameraTensorflow,
    LaplacianRigidEnergyTensorflow,
    Scene3DTensorflow,
)
from .triangulated_mesh_tensorflow import ColoredTriMeshTensorflow as ColoredTriMesh
from .. import LaplacianRigidEnergy


def qrot(q: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    qr = tf.tile(q[None, :], (v.shape[0], 1))
    qvec = qr[:, :-1]
    uv = tf.linalg.cross(qvec, v)
    uuv = tf.linalg.cross(qvec, uv)
    return v + 2 * (qr[:, 3][None, 0] * uv + uuv)


def mult_and_clamp(x: Union[np.ndarray, tf.Tensor], a: float, t: float) -> np.ndarray:
    return np.minimum(np.maximum(x * a, -t), t)


class MeshDepthFitter:
    """Class to fit a deformable mesh to a depth image using Tensorflow."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        euler_init: np.ndarray,
        translation_init: np.ndarray,
        cregu: float = 2000,
        inertia: float = 0.96,
        damping: float = 0.05,
    ):
        self.cregu = cregu
        self.inertia = inertia
        self.damping = damping
        self.step_factor_vertices = 0.0005
        self.step_max_vertices = 0.5
        self.step_factor_quaternion = 0.00006
        self.step_max_quaternion = 0.1
        self.step_factor_translation = 0.00005
        self.step_max_translation = 0.1

        self.mesh = ColoredTriMesh(
            faces, vertices, colors=np.zeros((vertices.shape[0], 0))
        )  # we do a copy to avoid negative stride not support by Tensorflow
        object_center = vertices.mean(axis=0)
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([-0.5, 0, 5]) * object_radius

        self.scene = Scene3DTensorflow()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = tf.constant(copy.copy(vertices))
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.set_mesh_transform_init(euler=euler_init, translation=translation_init)
        self.reset()

    def set_mesh_transform_init(
        self, euler: np.ndarray, translation: np.ndarray
    ) -> None:
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler(
            "zyx", euler
        ).as_quat()
        self.transform_translation_init = translation

    def reset(self) -> None:
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices_init.shape)
        self.transform_quaternion = copy.copy(self.transform_quaternion_init)
        self.transform_translation = copy.copy(self.transform_translation_init)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

    def set_max_depth(self, max_depth: float) -> None:
        self.max_depth = max_depth
        self.scene.set_background_color([max_depth])

    def set_depth_scale(self, depth_scale: float) -> None:
        self.depthScale = depth_scale

    def set_image(
        self,
        mesh_image: np.ndarray,
        focal: Optional[float] = None,
        distortion: Optional[np.ndarray] = None,
    ) -> None:
        self.width = mesh_image.shape[1]
        self.height = mesh_image.shape[0]
        assert mesh_image.ndim == 2
        self.mesh_image = mesh_image
        if focal is None:
            focal = 2 * self.width

        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        trans = -rot.T.dot(self.camera_center)
        intrinsic = np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((rot, trans))
        self.camera = CameraTensorflow(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
            distortion=distortion,
        )
        self.iter = 0

    def step(self) -> Tuple[float, np.ndarray, np.ndarray]:
        self.vertices = (
            self.vertices - tf.reduce_mean(self.vertices, axis=0)[None, :]
        )  # centervertices because we have another paramter to control translations

        with tf.GradientTape() as tape:

            vertices_with_grad = tf.constant(self.vertices)
            quaternion_with_grad = tf.constant(self.transform_quaternion)
            translation_with_grad = tf.constant(self.transform_translation)

            tape.watch(vertices_with_grad)
            tape.watch(quaternion_with_grad)
            tape.watch(translation_with_grad)

            vertices_with_grad_centered = (
                vertices_with_grad - tf.reduce_mean(vertices_with_grad, axis=0)[None, :]
            )

            q_normalized = quaternion_with_grad / tf.norm(
                quaternion_with_grad
            )  # that will lead to a gradient that is in the tangeant space
            vertices_with_grad_transformed = (
                qrot(q_normalized, vertices_with_grad_centered) + translation_with_grad
            )

            self.mesh.set_vertices(vertices_with_grad_transformed)

            depth_scale = 1 * self.depthScale
            depth = self.scene.render_depth(
                self.camera,
                depth_scale=depth_scale,
            )
            depth = tf.clip_by_value(depth, 0, self.max_depth)

            diff_image = tf.reduce_sum(
                (depth - tf.constant(self.mesh_image[:, :, None])) ** 2, axis=2
            )
            loss = tf.reduce_sum(diff_image)

            trainable_variables = [
                vertices_with_grad,
                quaternion_with_grad,
                translation_with_grad,
            ]
            vertices_grad, quaternion_grad, translation_grad = tape.gradient(
                loss, trainable_variables
            )

        energy_data = loss.numpy()

        grad_data = vertices_grad

        (
            energy_rigid,
            grad_rigidity,
            _,
        ) = self.rigid_energy.evaluate(self.vertices.numpy())
        energy = energy_data + energy_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))

        # update v
        grad = grad_data + grad_rigidity

        # update vertices
        step_vertices = mult_and_clamp(
            -grad.numpy(), self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * self.inertia + (1 - self.inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -quaternion_grad.numpy(),
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * self.inertia + (1 - self.inertia) * step_quaternion
        )
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        # update translation
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(
            self.transform_quaternion
        )
        step_translation = mult_and_clamp(
            -translation_grad.numpy(),
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * self.inertia
            + (1 - self.inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation

        self.iter += 1
        return energy, depth[:, :, 0].numpy(), diff_image.numpy()


class MeshRGBFitterWithPose:
    """Class to fit a deformable mesh to a color image using Tensorflow."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        euler_init: np.ndarray,
        translation_init: np.ndarray,
        default_color: np.ndarray,
        default_light: np.ndarray,
        cregu: float = 2000,
        inertia: float = 0.96,
        damping: float = 0.05,
        update_lights: bool = True,
        update_color: bool = True,
    ):
        self.cregu = cregu

        self.inertia = inertia
        self.damping = damping
        self.step_factor_vertices = 0.0005
        self.step_max_vertices = 0.5
        self.step_factor_quaternion = 0.00006
        self.step_max_quaternion = 0.05
        self.step_factor_translation = 0.00005
        self.step_max_translation = 0.1

        self.default_color = default_color
        self.default_light = default_light
        self.update_lights = update_lights
        self.update_color = update_color
        self.mesh = ColoredTriMesh(
            faces=faces.copy(),
            vertices=vertices,
            colors=np.zeros((vertices.shape[0], 0)),
        )  # we do a copy to avoid negative stride not support by Tensorflow
        object_center = vertices.mean(axis=0) + translation_init
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([0, 0, 9]) * object_radius

        self.scene = Scene3DTensorflow()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergyTensorflow(self.mesh, vertices, cregu)
        self.vertices_init = tf.constant(copy.copy(vertices))
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.set_mesh_transform_init(euler=euler_init, translation=translation_init)
        self.reset()

    def set_background_color(self, background_color: np.ndarray) -> None:
        self.scene.set_background_color(background_color)

    def set_mesh_transform_init(
        self, euler: np.ndarray, translation: np.ndarray
    ) -> None:
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler(
            "zyx", euler
        ).as_quat()
        self.transform_translation_init = translation

    def reset(self) -> None:
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices.shape)
        self.transform_quaternion = copy.copy(self.transform_quaternion_init)
        self.transform_translation = copy.copy(self.transform_translation_init)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

        self.mesh_color = copy.copy(self.default_color)
        self.light_directional = copy.copy(self.default_light["directional"])
        self.light_ambient = copy.copy(self.default_light["ambient"])

        self.speed_light_directional = np.zeros(self.light_directional.shape)
        self.speed_light_ambient = np.zeros(self.light_ambient.shape)
        self.speed_mesh_color = np.zeros(self.mesh_color.shape)

    def set_image(
        self,
        mesh_image: np.ndarray,
        focal: Optional[float] = None,
        distortion: Optional[np.ndarray] = None,
    ) -> None:
        self.width = mesh_image.shape[1]
        self.height = mesh_image.shape[0]
        assert mesh_image.ndim == 3
        self.mesh_image = mesh_image
        if focal is None:
            focal = 2 * self.width

        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        trans = -rot.T.dot(self.camera_center)
        intrinsic = np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((rot, trans))
        self.camera = CameraTensorflow(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            distortion=distortion,
            width=self.width,
            height=self.height,
        )
        self.iter = 0

    def step(self) -> Tuple[float, np.ndarray, np.ndarray]:

        with tf.GradientTape() as tape:

            vertices_with_grad = tf.constant(self.vertices)
            quaternion_with_grad = tf.constant(self.transform_quaternion)
            translation_with_grad = tf.constant(self.transform_translation)

            light_directional_with_grad = tf.constant(self.light_directional)
            light_ambient_with_grad = tf.constant(self.light_ambient)
            mesh_color_with_grad = tf.constant(self.mesh_color)

            tape.watch(vertices_with_grad)
            tape.watch(quaternion_with_grad)
            tape.watch(translation_with_grad)

            tape.watch(light_directional_with_grad)
            tape.watch(light_ambient_with_grad)
            tape.watch(mesh_color_with_grad)

            vertices_with_grad_centered = (
                vertices_with_grad - tf.reduce_mean(vertices_with_grad, axis=0)[None, :]
            )

            q_normalized = quaternion_with_grad / tf.norm(
                quaternion_with_grad
            )  # that will lead to a gradient that is in the tangeant space
            vertices_with_grad_transformed = (
                qrot(q_normalized, vertices_with_grad_centered) + translation_with_grad
            )
            self.mesh.set_vertices(vertices_with_grad_transformed)

            self.scene.set_light(
                light_directional=light_directional_with_grad,
                light_ambient=light_ambient_with_grad,
            )
            self.mesh.set_vertices_colors(
                tf.tile(mesh_color_with_grad[None, :], [self.mesh.nb_vertices, 1])
            )

            image = self.scene.render(self.camera)

            diff_image = tf.reduce_sum(
                (image - tf.constant(self.mesh_image)) ** 2, axis=2
            )
            loss = tf.reduce_sum(diff_image)

            trainable_variables = [
                vertices_with_grad,
                quaternion_with_grad,
                translation_with_grad,
                light_directional_with_grad,
                light_ambient_with_grad,
                mesh_color_with_grad,
            ]
            (
                vertices_grad,
                quaternion_grad,
                translation_grad,
                light_directional_grad,
                light_ambient_grad,
                mesh_color_grad,
            ) = tape.gradient(loss, trainable_variables)

        energy_data = loss.numpy()

        grad_data = vertices_grad

        (
            energy_rigid,
            grad_rigidity,
            approx_hessian_rigidity,
        ) = self.rigid_energy.evaluate(self.vertices)
        energy = energy_data + energy_rigid.numpy()
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))

        # update v
        grad = grad_data + grad_rigidity

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(
            -grad.numpy(), self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * inertia + (1 - inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -quaternion_grad, self.step_factor_quaternion, self.step_max_quaternion
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        # update translation
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(
            self.transform_quaternion
        )
        step_translation = mult_and_clamp(
            -translation_grad, self.step_factor_translation, self.step_max_translation
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation
        # update directional light
        step = -light_directional_grad * 0.0001
        self.speed_light_directional = (1 - self.damping) * (
            self.speed_light_directional * inertia + (1 - inertia) * step
        )
        self.light_directional = self.light_directional + self.speed_light_directional
        # update ambient light
        step = -light_ambient_grad * 0.0001
        self.speed_light_ambient = (1 - self.damping) * (
            self.speed_light_ambient * inertia + (1 - inertia) * step
        )
        self.light_ambient = self.light_ambient + self.speed_light_ambient
        # update mesh color
        step = -mesh_color_grad * 0.00001
        self.speed_mesh_color = (1 - self.damping) * (
            self.speed_mesh_color * inertia + (1 - inertia) * step
        )
        self.mesh_color = self.mesh_color + self.speed_mesh_color

        self.iter += 1
        return energy, image.numpy(), diff_image.numpy()  # type: ignore
