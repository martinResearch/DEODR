"""Modules containing Tensorflow classes to fit 3D meshes to images using differentiable rendering."""


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
from .triangulated_mesh_tensorflow import TriMeshTensorflow as TriMesh
from .. import LaplacianRigidEnergy


def qrot(q, v):
    qr = tf.tile(q[None, :], (v.shape[0], 1))
    qvec = qr[:, :-1]
    uv = tf.linalg.cross(qvec, v)
    uuv = tf.linalg.cross(qvec, uv)
    return v + 2 * (qr[:, 3][None, 0] * uv + uuv)


class MeshDepthFitter:
    """Class to fit a deformable mesh to a depth image using Tensorflow."""

    def __init__(
        self,
        vertices,
        faces,
        euler_init,
        translation_init,
        cregu=2000,
        inertia=0.96,
        damping=0.05,
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

        self.mesh = TriMesh(
            faces, vertices
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

    def set_mesh_transform_init(self, euler, translation):
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler(
            "zyx", euler
        ).as_quat()
        self.transform_translation_init = translation

    def reset(self):
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices_init.shape)
        self.transform_quaternion = copy.copy(self.transform_quaternion_init)
        self.transform_translation = copy.copy(self.transform_translation_init)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

    def set_max_depth(self, max_depth):
        self.scene.max_depth = max_depth
        self.scene.set_background(
            np.full((self.height, self.width, 1), max_depth, dtype=np.float)
        )

    def set_depth_scale(self, depth_scale):
        self.depthScale = depth_scale

    def set_image(self, hand_image, focal=None, distortion=None):
        self.width = hand_image.shape[1]
        self.height = hand_image.shape[0]
        assert hand_image.ndim == 2
        self.hand_image = hand_image
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

    def step(self):
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
            depth = tf.clip_by_value(depth, 0, self.scene.max_depth)

            diff_image = tf.reduce_sum(
                (depth - tf.constant(self.hand_image[:, :, None])) ** 2, axis=2
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
            approx_hessian_rigidity,
        ) = self.rigid_energy.evaluate(self.vertices.numpy())
        energy = energy_data + energy_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))

        # update v
        grad = grad_data + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

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
        vertices,
        faces,
        euler_init,
        translation_init,
        default_color,
        default_light,
        cregu=2000,
        inertia=0.96,
        damping=0.05,
        update_lights=True,
        update_color=True,
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
            faces.copy()
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

    def set_background_color(self, background_color):
        self.scene.set_background(
            np.tile(background_color[None, None, :], (self.height, self.width, 1))
        )

    def set_mesh_transform_init(self, euler, translation):
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler(
            "zyx", euler
        ).as_quat()
        self.transform_translation_init = translation

    def reset(self):
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices.shape)
        self.transform_quaternion = copy.copy(self.transform_quaternion_init)
        self.transform_translation = copy.copy(self.transform_translation_init)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

        self.hand_color = copy.copy(self.default_color)
        self.light_directional = copy.copy(self.default_light["directional"])
        self.light_ambient = copy.copy(self.default_light["ambient"])

        self.speed_light_directional = np.zeros(self.light_directional.shape)
        self.speed_light_ambient = np.zeros(self.light_ambient.shape)
        self.speed_hand_color = np.zeros(self.hand_color.shape)

    def set_image(self, hand_image, focal=None, distortion=None):
        self.width = hand_image.shape[1]
        self.height = hand_image.shape[0]
        assert hand_image.ndim == 3
        self.hand_image = hand_image
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

    def step(self):

        with tf.GradientTape() as tape:

            vertices_with_grad = tf.constant(self.vertices)
            quaternion_with_grad = tf.constant(self.transform_quaternion)
            translation_with_grad = tf.constant(self.transform_translation)

            light_directional_with_grad = tf.constant(self.light_directional)
            light_ambient_with_grad = tf.constant(self.light_ambient)
            hand_color_with_grad = tf.constant(self.hand_color)

            tape.watch(vertices_with_grad)
            tape.watch(quaternion_with_grad)
            tape.watch(translation_with_grad)

            tape.watch(light_directional_with_grad)
            tape.watch(light_ambient_with_grad)
            tape.watch(hand_color_with_grad)

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
                tf.tile(hand_color_with_grad[None, :], [self.mesh.nb_vertices, 1])
            )

            image = self.scene.render(self.camera)

            diff_image = tf.reduce_sum(
                (image - tf.constant(self.hand_image)) ** 2, axis=2
            )
            loss = tf.reduce_sum(diff_image)

            trainable_variables = [
                vertices_with_grad,
                quaternion_with_grad,
                translation_with_grad,
                light_directional_with_grad,
                light_ambient_with_grad,
                hand_color_with_grad,
            ]
            (
                vertices_grad,
                quaternion_grad,
                translation_grad,
                light_directional_grad,
                light_ambient_grad,
                hand_color_grad,
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

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

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
        # update hand color
        step = -hand_color_grad * 0.00001
        self.speed_hand_color = (1 - self.damping) * (
            self.speed_hand_color * inertia + (1 - inertia) * step
        )
        self.hand_color = self.hand_color + self.speed_hand_color

        self.iter += 1
        return energy, image.numpy(), diff_image.numpy()
