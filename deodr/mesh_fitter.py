"""Modules containing classes to fit 3D meshes to images using differentiable rendering."""

import copy

import numpy as np

import scipy.sparse.linalg
import scipy.spatial.transform.rotation

from . import Camera, ColoredTriMesh, LaplacianRigidEnergy, Scene3D, TriMesh
from .tools import (
    normalize,
    normalize_backward,
    qrot,
    qrot_backward,
    check_jacobian_finite_differences,
)


class MeshDepthFitter:
    """Class to fit a deformable mesh to a depth image."""

    def __init__(
        self,
        vertices,
        faces,
        euler_init,
        translation_init,
        cregu=2000,
        inertia=0.96,
        damping=0.05
    ):
        self.cregu = cregu
        self.inertia = inertia
        self.damping = damping
        self.step_factor_vertices = 0.0005
        self.step_max_vertices = 1
        self.step_factor_quaternion = 0.00006
        self.step_max_quaternion = 0.1
        self.step_factor_translation = 0.00005
        self.step_max_translation = 0.1

        self.mesh = TriMesh(
            faces, vertices=vertices
        )  # we do a copy to avoid negative stride not support by pytorch
        object_center = vertices.mean(axis=0)
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([-0.5, 0, 5]) * object_radius

        self.scene = Scene3D()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = copy.copy(vertices)
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
        self.scene.set_background_color(np.array([max_depth], dtype=np.float))

    def set_depth_scale(self, depth_scale):
        self.depthScale = depth_scale

    def set_image(self, mesh_image, focal=None, distortion=None):
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
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            distortion=distortion,
            height=self.height,
            width=self.width,
        )
        self.iter = 0

    def render(self):
        q_normalized = normalize(
            self.transform_quaternion
        )  # that will lead to a gradient that is in the tangeant space
        vertices_transformed = (
            qrot(q_normalized, self.vertices) + self.transform_translation
        )
        self.mesh.set_vertices(vertices_transformed)
        self.depth_not_clipped = self.scene.render_depth(
            self.camera, depth_scale=self.depthScale,
        )
        depth = np.clip(self.depth_not_clipped, 0, self.scene.max_depth)
        return depth

    def render_backward(self, depth_b):
        self.scene.clear_gradients()
        depth_b[self.depth_not_clipped < 0] = 0
        depth_b[self.depth_not_clipped > self.scene.max_depth] = 0
        self.scene.render_depth_backward(depth_b)
        vertices_transformed_b = self.scene.mesh.vertices_b
        self.transform_translation_b = np.sum(vertices_transformed_b, axis=0)
        q_normalized = normalize(self.transform_quaternion)
        q_normalized_b, self.vertices_b = qrot_backward(
            q_normalized, self.vertices, vertices_transformed_b
        )
        self.transform_quaternion_b = normalize_backward(
            self.transform_quaternion, q_normalized_b
        )  # that will lead to a gradient that is in the tangeant space
        return

    def step(self):

        self.vertices = self.vertices - np.mean(self.vertices, axis=0)[None, :]
        depth = self.render()

        diff_image = np.sum((depth - self.mesh_image[:, :, None]) ** 2, axis=2)
        energy_data = np.sum(diff_image)
        depth_b = 2 * (depth - self.mesh_image[:, :, None])
        self.render_backward(depth_b)

        self.vertices_b = self.vertices_b - np.mean(self.vertices_b, axis=0)[None, :]
        grad_data = self.vertices_b
        # update v

        (
            energy_rigid,
            grad_rigidity,
            _,
        ) = self.rigid_energy.evaluate(self.vertices)
        energy = energy_data + energy_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))

        # update v
        grad = grad_data + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia
        # update vertices
        step_vertices = mult_and_clamp(
            -grad, self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * self.inertia + (1 - self.inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -self.transform_quaternion_b,
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(
            self.transform_quaternion
        )
        # update translation
        step_translation = mult_and_clamp(
            -self.transform_translation_b,
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation

        self.iter += 1
        return energy, depth[:, :, 0], diff_image


class MeshRGBFitterWithPose:
    """Class to fit a deformable mesh to a color image."""

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
        self.mesh = ColoredTriMesh(faces.copy(), vertices=vertices, nb_colors=3)
        object_center = vertices.mean(axis=0) + translation_init
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([0, 0, 9]) * object_radius

        self.scene = Scene3D()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = copy.copy(vertices)
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.set_mesh_transform_init(euler=euler_init, translation=translation_init)
        self.reset()

    def set_background_color(self, background_color):
        self.scene.set_background_color(background_color)

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

        self.mesh_color = copy.copy(self.default_color)
        self.light_directional = copy.copy(self.default_light["directional"])
        self.light_ambient = copy.copy(self.default_light["ambient"])

        self.speed_light_directional = np.zeros(self.light_directional.shape)
        self.speed_light_ambient = np.zeros(self.light_ambient.shape)
        self.speed_mesh_color = np.zeros(self.mesh_color.shape)

    def set_image(self, mesh_image, focal=None, distortion=None):
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
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            distortion=distortion,
            width=self.width,
            height=self.height,
        )
        self.iter = 0

    def render(self):
        q_normalized = normalize(
            self.transform_quaternion
        )  # that will lead to a gradient that is in the tangeant space
        vertices_transformed = (
            qrot(q_normalized, self.vertices) + self.transform_translation
        )
        self.mesh.set_vertices(vertices_transformed)
        self.scene.set_light(
            light_directional=self.light_directional, light_ambient=self.light_ambient
        )
        self.mesh.set_vertices_colors(
            np.tile(self.mesh_color, (self.mesh.nb_vertices, 1))
        )
        image = self.scene.render(self.camera)
        return image

    def render_backward(self, image_b):
        self.scene.clear_gradients()
        self.scene.render_backward(image_b)
        self.mesh_color_b = np.sum(self.mesh.vertices_colors_b, axis=0)
        self.light_directional_b = self.scene.light_directional_b
        self.light_ambient_b = self.scene.light_ambient_b
        vertices_transformed_b = self.scene.mesh.vertices_b
        self.transform_translation_b = np.sum(vertices_transformed_b, axis=0)
        q_normalized = normalize(self.transform_quaternion)
        q_normalized_b, self.vertices_b = qrot_backward(
            q_normalized, self.vertices, vertices_transformed_b
        )
        self.transform_quaternion_b = normalize_backward(
            self.transform_quaternion, q_normalized_b
        )  # that will lead to a gradient that is in the tangeant space
        return

    def step(self):
        self.vertices = self.vertices - np.mean(self.vertices, axis=0)[None, :]

        image = self.render()

        diff_image = np.sum((image - self.mesh_image) ** 2, axis=2)
        image_b = 2 * (image - self.mesh_image)
        energy_data = np.sum(diff_image)

        (
            energy_rigid,
            grad_rigidity,
            _,
        ) = self.rigid_energy.evaluate(self.vertices)
        energy = energy_data + energy_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))

        self.render_backward(image_b)

        self.vertices_b = self.vertices_b - np.mean(self.vertices_b, axis=0)[None, :]
        # update v
        grad = self.vertices_b + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(
            -grad, self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * inertia + (1 - inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -self.transform_quaternion_b,
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(
            self.transform_quaternion
        )
        # update translation
        step_translation = mult_and_clamp(
            -self.transform_translation_b,
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation
        # update directional light
        step = -self.light_directional_b * 0.0001
        self.speed_light_directional = (1 - self.damping) * (
            self.speed_light_directional * inertia + (1 - inertia) * step
        )
        self.light_directional = self.light_directional + self.speed_light_directional
        # update ambient light
        step = -self.light_ambient_b * 0.0001
        self.speed_light_ambient = (1 - self.damping) * (
            self.speed_light_ambient * inertia + (1 - inertia) * step
        )
        self.light_ambient = self.light_ambient + self.speed_light_ambient
        # update mesh color
        step = -self.mesh_color_b * 0.00001
        self.speed_mesh_color = (1 - self.damping) * (
            self.speed_mesh_color * inertia + (1 - inertia) * step
        )
        self.mesh_color = self.mesh_color + self.speed_mesh_color

        self.iter += 1
        return energy, image, diff_image


class MeshRGBFitterWithPoseMultiFrame:
    """Class to fit a deformable mesh to multiple color images."""

    def __init__(
        self,
        vertices,
        faces,
        euler_init,
        translation_init,
        default_color,
        default_light,
        cregu=2000,
        cdata=1,
        inertia=0.97,
        damping=0.15,
        update_lights=True,
        update_color=True,
    ):
        self.cregu = cregu
        self.cdata = cdata
        self.inertia = inertia
        self.damping = damping
        self.step_factor_vertices = 0.0005
        self.step_max_vertices = 0.5
        self.step_factor_quaternion = 0.00005
        self.step_max_quaternion = 0.05
        self.step_factor_translation = 0.00004
        self.step_max_translation = 0.1

        self.default_color = default_color
        self.default_light = default_light
        self.update_lights = update_lights
        self.update_color = update_color
        self.mesh = ColoredTriMesh(faces, vertices, nb_colors=3)
        object_center = vertices.mean(axis=0)
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([0, 0, 6]) * object_radius

        self.scene = Scene3D()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = copy.copy(vertices)
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.set_mesh_transform_init(euler=euler_init, translation=translation_init)
        self.reset()

    def set_background_color(self, background_color):
        self.scene.set_background_color(background_color)

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

        self.mesh_color = copy.copy(self.default_color)
        self.light_directional = copy.copy(self.default_light["directional"])
        self.light_ambient = copy.copy(self.default_light["ambient"])

        self.speed_light_directional = np.zeros(self.light_directional.shape)
        self.speed_light_ambient = np.zeros(self.light_ambient.shape)
        self.speed_mesh_color = np.zeros(self.mesh_color.shape)

    def set_images(self, mesh_images, focal=None):
        self.width = mesh_images[0].shape[1]
        self.height = mesh_images[0].shape[0]
        assert mesh_images[0].ndim == 3
        self.mesh_images = mesh_images
        if focal is None:
            focal = 2 * self.width

        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        trans = -rot.T.dot(self.camera_center)
        intrinsic = np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((rot, trans))
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
        )
        self.iter = 0

    def set_image(self, mesh_image, focal=None):
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
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
        )
        self.iter = 0

    def render(self, idframe=None):
        unormalized_quaternion = self.transform_quaternion[idframe]
        q_normalized = normalize(
            unormalized_quaternion
        )  # that will lead to a gradient that is in the tangeant space
        vertices_transformed = (
            qrot(q_normalized, self.vertices) + self.transform_translation[idframe]
        )
        self.mesh.set_vertices(vertices_transformed)
        self.scene.set_light(
            light_directional=self.light_directional, light_ambient=self.light_ambient
        )
        self.mesh.set_vertices_colors(
            np.tile(self.mesh_color, (self.mesh.nb_vertices, 1))
        )
        image = self.scene.render(self.camera)
        self.store_backward["render"] = (idframe, unormalized_quaternion, q_normalized)
        return image

    def clear_gradients(self):
        self.light_directional_b = np.zeros(self.light_directional.shape)
        self.light_ambient_b = np.zeros(self.light_ambient.shape)
        self.vertices_b = np.zeros(self.vertices.shape)
        self.transform_quaternion_b = np.zeros(self.transform_quaternion.shape)
        self.transform_translation_b = np.zeros(self.transform_translation.shape)
        self.mesh_color_b = np.zeros(self.mesh_color.shape)
        self.store_backward = {}

    def render_backward(self, image_b):
        idframe, unormalized_quaternion, q_normalized = self.store_backward["render"]
        self.scene.clear_gradients()
        self.scene.render_backward(image_b)
        self.mesh_color_b += np.sum(self.mesh.vertices_colors_b, axis=0)
        self.light_directional_b += self.scene.light_directional_b
        self.light_ambient_b += self.scene.light_ambient_b
        vertices_transformed_b = self.scene.mesh.vertices_b
        self.transform_translation_b[idframe] += np.sum(vertices_transformed_b, axis=0)
        q_normalized_b, vertices_b = qrot_backward(
            q_normalized, self.vertices, vertices_transformed_b
        )
        self.vertices_b += vertices_b
        self.transform_quaternion_b[idframe] += normalize_backward(
            unormalized_quaternion, q_normalized_b
        )  # that will lead to a gradient that is in the tangeant space
        return

    def energy_data(self, vertices, return_images=True):
        self.vertices = vertices
        image = [None] * self.nb_facesrames
        diff_image = [None] * self.nb_facesrames
        image_b = [None] * self.nb_facesrames
        energy_datas = [None] * self.nb_facesrames
        self.clear_gradients()
        coef_data = self.cdata / self.nb_facesrames
        for idframe in range(self.nb_facesrames):
            image[idframe] = self.render(idframe=idframe)
            diff_image[idframe] = np.sum(
                (image[idframe] - self.mesh_images[idframe]) ** 2, axis=2
            )
            image_b = coef_data * 2 * (image[idframe] - self.mesh_images[idframe])
            energy_datas[idframe] = coef_data * np.sum(diff_image[idframe])
            self.render_backward(image_b)
        energy_data = np.sum(energy_datas)
        if return_images:
            return energy_data, image, diff_image
        else:
            return energy_data

    def step(self, check_gradient=False):

        self.vertices = self.vertices - np.mean(self.vertices, axis=0)[None, :]

        self.nb_facesrames = len(self.mesh_images)

        energy_data, image, diff_image = self.energy_data(self.vertices)
        (
            energy_rigid,
            grad_rigidity,
            _,
        ) = self.rigid_energy.evaluate(self.vertices)

        if check_gradient:

            def func(x):
                return self.rigid_energy.evaluate(
                    x, return_grad=False, return_hessian=False
                )

            check_jacobian_finite_differences(
                grad_rigidity.flatten(), func, self.vertices
            )

            def func(x):
                return self.energy_data(x, return_images=False)

            grad_data = self.vertices_b.copy()
            check_jacobian_finite_differences(grad_data.flatten(), func, self.vertices)

        energy = energy_data + energy_rigid
        print(
            f"iter {self.iter} Energy={energy} : EData={energy_data} E_rigid={energy_rigid}"
        )

        if self.iter < 500:
            self.vertices_b = (
                self.vertices_b - np.mean(self.vertices_b, axis=0)[None, :]
            )
        # update v
        grad = self.vertices_b + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(
            -grad, self.step_factor_vertices, self.step_max_vertices
        )

        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * inertia + (1 - inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation

        step_quaternion = mult_and_clamp(
            -self.transform_quaternion_b,
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(
            self.transform_quaternion
        )
        # update translation
        step_translation = mult_and_clamp(
            -self.transform_translation_b,
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation
        # update directional light
        step = -self.light_directional_b * 0.0001
        self.speed_light_directional = (1 - self.damping) * (
            self.speed_light_directional * inertia + (1 - inertia) * step
        )
        self.light_directional = self.light_directional + self.speed_light_directional
        # update ambient light
        step = -self.light_ambient_b * 0.0001
        self.speed_light_ambient = (1 - self.damping) * (
            self.speed_light_ambient * inertia + (1 - inertia) * step
        )
        self.light_ambient = self.light_ambient + self.speed_light_ambient
        # update mesh color
        step = -self.mesh_color_b * 0.00001
        self.speed_mesh_color = (1 - self.damping) * (
            self.speed_mesh_color * inertia + (1 - inertia) * step
        )
        self.mesh_color = self.mesh_color + self.speed_mesh_color

        self.iter += 1
        return energy, image, diff_image
