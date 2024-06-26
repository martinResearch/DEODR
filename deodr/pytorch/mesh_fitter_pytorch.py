# type: ignore
"""Modules containing pytorch classes to fit 3D meshes to images using differentiable rendering."""

import copy
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse.linalg
import scipy.spatial.transform.rotation
import torch

from .. import LaplacianRigidEnergy
from . import CameraPytorch, LaplacianRigidEnergyPytorch, Scene3DPytorch
from .triangulated_mesh_pytorch import ColoredTriMeshPytorch as ColoredTriMesh


def print_grad(name: str) -> Callable[[torch.Tensor], None]:
    # to visualize the gradient of a variable use
    # variable_name.register_hook(print_grad('variable_name'))
    def hook(grad: torch.Tensor) -> None:
        print(f"grad {name} = {grad}")

    return hook


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qr = q[None, :].repeat(v.shape[0], 1)
    qvec = qr[:, :-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return v + 2 * (qr[:, [3]] * uv + uuv)


class MeshDepthFitterEnergy(torch.nn.Module):
    """Pytorch module to fit a deformable mesh to a depth image."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        euler_init: np.ndarray,
        translation_init: np.ndarray,
        cregu: float = 2000,
    ):
        super(MeshDepthFitterEnergy, self).__init__()
        self.mesh = ColoredTriMesh(
            faces=faces[:, ::-1].copy(),
            vertices=vertices,
            colors=np.zeros((vertices.shape[0], 0)),
        )  # we do a copy to avoid negative stride not supported by pytorch
        object_center = vertices.mean(axis=0)
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([-0.5, 0, 5]) * object_radius
        self.scene = Scene3DPytorch()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergyPytorch(self.mesh, vertices, cregu)
        self.Vinit = copy.copy(self.mesh.vertices)
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler("zyx", euler_init).as_quat()
        self.transform_translation_init = translation_init
        self._vertices = torch.nn.Parameter(torch.tensor(self.Vinit, dtype=torch.float64))
        self.quaternion = torch.nn.Parameter(torch.tensor(self.transform_quaternion_init, dtype=torch.float64))
        self.translation = torch.nn.Parameter(torch.tensor(self.transform_translation_init, dtype=torch.float64))

    def set_max_depth(self, max_depth: float) -> None:
        self.max_depth = max_depth
        self.scene.set_background_color(np.array([max_depth], dtype=np.float64))

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
        t = -rot.T.dot(self.camera_center)
        intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
        extrinsic = np.column_stack((rot, t))
        self.camera = CameraPytorch(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            distortion=distortion,
            height=mesh_image.shape[0],
            width=mesh_image.shape[1],
        )
        self.iter = 0

    def forward(self) -> torch.Tensor:
        q_normalized = self.quaternion / self.quaternion.norm()
        print(self.quaternion.norm())
        vertices_centered = self._vertices - torch.mean(self._vertices, dim=0)[None, :]
        v_transformed = qrot(q_normalized, vertices_centered) + self.translation
        self.mesh.set_vertices(v_transformed)
        depth_scale = 1 * self.depthScale
        depth = self.scene.render_depth(
            self.camera,
            depth_scale=depth_scale,
        )
        depth = torch.clamp(depth, 0, self.max_depth)
        diff_image = torch.sum((depth - torch.tensor(self.mesh_image[:, :, None])) ** 2, dim=2)
        self.depth = depth
        self.diff_image = diff_image
        energy_data = torch.sum(diff_image)
        energy_rigid = self.rigid_energy.evaluate(
            self._vertices,
        )[0]
        energy = energy_data + energy_rigid
        self.loss = energy_data + energy_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))
        return self.loss


class MeshDepthFitterPytorchOptim:
    """Pytorch optimizer to fit a deformable mesh to an image."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        euler_init: np.ndarray,
        translation_init: np.ndarray,
        cregu: float = 2000,
        lr: float = 0.8,
    ):
        self.energy = MeshDepthFitterEnergy(vertices, faces, euler_init, translation_init, cregu)
        params = self.energy.parameters()
        self.optimizer = torch.optim.LBFGS(params, lr=0.8, max_iter=1)
        # self.optimizer = torch.optim.SGD(params, lr=0.000005, momentum=0.1,
        # dampening=0.1        )
        # self.optimizer =torch.optim.RMSprop(params, lr=1e-3, alpha=0.99,  eps=1e-8,
        # weight_decay=0,  momentum=0.001)
        # self.optimizer = torch.optim.Adadelta(params, lr=0.1, rho=0.95,
        #   eps=1e-6,  weight_decay=0)
        # self.optimizer = torch.optim.Adagrad(self.energy.parameters(), lr=0.02)

    def set_image(self, depth_image: np.ndarray, focal: float) -> None:
        self.energy.set_image(depth_image, focal=focal)

    def set_max_depth(self, max_depth: float) -> None:
        self.energy.set_max_depth(max_depth)

    def set_depth_scale(self, depth_scale: float) -> None:
        self.energy.set_depth_scale(depth_scale)

    def step(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        def closure() -> float:
            self.optimizer.zero_grad()
            loss = self.energy()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        # self.iter += 1

        return (
            self.energy.loss,
            self.energy.depth[:, :, 0].detach().numpy(),
            self.energy.diff_image.detach().numpy(),
        )


def mult_and_clamp(x: np.ndarray, a: float, t: float) -> np.ndarray:
    return np.minimum(np.maximum(x * a, -t), t)


class MeshDepthFitter:
    """Class to fit a deformable mesh to a depth image."""

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
            faces=faces.copy(),
            vertices=vertices,
            colors=np.zeros((vertices.shape[0], 0)),
        )  # we do a copy to avoid negative stride not support by pytorch
        object_center = vertices.mean(axis=0) + translation_init
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([-0.5, 0, 5]) * object_radius

        self.scene = Scene3DPytorch()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = torch.tensor(copy.copy(vertices))
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.set_mesh_transform_init(euler=euler_init, translation=translation_init)
        self.reset()

    def set_mesh_transform_init(self, euler: np.ndarray, translation: np.ndarray) -> None:
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler("zyx", euler).as_quat()
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
        intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
        extrinsic = np.column_stack((rot, trans))
        self.camera = CameraPytorch(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
            distortion=distortion,
        )
        self.iter = 0

    def step(self) -> Tuple[float, np.ndarray, np.ndarray]:
        self.vertices = self.vertices - torch.mean(self.vertices, dim=0)[None, :]
        # vertices_with_grad = self.vertices.clone().requires_grad(True)
        vertices_with_grad = self.vertices.clone().detach().requires_grad_(True)
        vertices_with_grad_centered = vertices_with_grad - torch.mean(vertices_with_grad, dim=0)[None, :]
        quaternion_with_grad = torch.tensor(self.transform_quaternion, dtype=torch.float64, requires_grad=True)
        translation_with_grad = torch.tensor(self.transform_translation, dtype=torch.float64, requires_grad=True)

        q_normalized = (
            quaternion_with_grad / quaternion_with_grad.norm()
        )  # that will lead to a gradient that is in the tangent space
        vertices_with_grad_transformed = qrot(q_normalized, vertices_with_grad_centered) + translation_with_grad

        self.mesh.set_vertices(vertices_with_grad_transformed)

        depth_scale = 1 * self.depthScale
        depth = self.scene.render_depth(self.camera, depth_scale=depth_scale)
        depth = torch.clamp(depth, 0, self.max_depth)

        diff_image = torch.sum((depth - torch.tensor(self.mesh_image[:, :, None])) ** 2, dim=2)
        loss = torch.sum(diff_image)

        loss.backward()
        energy_data = loss.detach().numpy()

        grad_data = vertices_with_grad.grad.numpy()

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
        step_vertices = mult_and_clamp(-grad, self.step_factor_vertices, self.step_max_vertices)
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * self.inertia + (1 - self.inertia) * step_vertices
        )
        self.vertices = self.vertices + torch.tensor(self.speed_vertices)
        # update rotation
        step_quaternion = mult_and_clamp(
            -quaternion_with_grad.grad.numpy(),
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * self.inertia + (1 - self.inertia) * step_quaternion
        )
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(self.transform_quaternion)
        # update translation

        step_translation = mult_and_clamp(
            -translation_with_grad.grad.numpy(),
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * self.inertia + (1 - self.inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation

        self.iter += 1
        return energy, depth[:, :, 0].detach().numpy(), diff_image.detach().numpy()


class MeshRGBFitterWithPose:
    """Class to fit a deformable mesh to a color image."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        euler_init: np.ndarray,
        translation_init: np.ndarray,
        default_color: np.ndarray,
        default_light_directional: np.ndarray,
        default_light_ambient: float,
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
        self.default_light_directional = default_light_directional
        self.default_light_ambient = default_light_ambient
        self.update_lights = update_lights
        self.update_color = update_color
        self.mesh = ColoredTriMesh(
            faces=faces.copy(),
            vertices=vertices,
            colors=np.zeros((vertices.shape[0], 0)),
        )  # we do a copy to avoid negative stride not support by pytorch
        object_center = vertices.mean(axis=0) + translation_init
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([0, 0, 9]) * object_radius

        self.scene = Scene3DPytorch()
        self.scene.set_mesh(self.mesh)
        self.rigid_energy = LaplacianRigidEnergyPytorch(self.mesh, vertices, cregu)
        self.vertices_init = torch.tensor(copy.copy(vertices))
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.set_mesh_transform_init(euler=euler_init, translation=translation_init)
        self.reset()

    def set_background_color(self, background_color: np.ndarray) -> None:
        self.scene.set_background_color(background_color)

    def set_mesh_transform_init(self, euler: np.ndarray, translation: np.ndarray) -> None:
        self.transform_quaternion_init = scipy.spatial.transform.Rotation.from_euler("zyx", euler).as_quat()
        self.transform_translation_init = translation

    def reset(self) -> None:
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices.shape)
        self.transform_quaternion = copy.copy(self.transform_quaternion_init)
        self.transform_translation = copy.copy(self.transform_translation_init)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

        self.mesh_color = copy.copy(self.default_color)
        self.light_directional = copy.copy(self.default_light_directional)
        self.light_ambient = copy.copy(self.default_light_ambient)

        self.speed_light_directional = np.zeros(self.light_directional.shape)
        self.speed_light_ambient = 0.0
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
        intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
        extrinsic = np.column_stack((rot, trans))
        self.camera = CameraPytorch(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
            distortion=distortion,
        )
        self.iter = 0

    def step(self) -> Tuple[float, np.ndarray, np.ndarray]:
        self.vertices = self.vertices - torch.mean(self.vertices, dim=0)[None, :]
        vertices_with_grad = self.vertices.clone().detach().requires_grad_(True)
        vertices_with_grad_centered = vertices_with_grad - torch.mean(vertices_with_grad, dim=0)[None, :]
        quaternion_with_grad = torch.tensor(self.transform_quaternion, dtype=torch.float64, requires_grad=True)
        translation_with_grad = torch.tensor(self.transform_translation, dtype=torch.float64, requires_grad=True)

        light_directional_with_grad = torch.tensor(self.light_directional, dtype=torch.float64, requires_grad=True)
        light_ambient_with_grad = torch.tensor(self.light_ambient, dtype=torch.float64, requires_grad=True)
        mesh_color_with_grad = torch.tensor(self.mesh_color, dtype=torch.float64, requires_grad=True)

        q_normalized = (
            quaternion_with_grad / quaternion_with_grad.norm()
        )  # that will lead to a gradient that is in the tangent space
        vertices_with_grad_transformed = qrot(q_normalized, vertices_with_grad_centered) + translation_with_grad
        self.mesh.set_vertices(vertices_with_grad_transformed)

        self.scene.set_light(
            light_directional=light_directional_with_grad,
            light_ambient=light_ambient_with_grad,
        )
        self.mesh.set_vertices_colors(mesh_color_with_grad.repeat([self.mesh.nb_vertices, 1]))

        image = self.scene.render(self.camera)

        diff_image = torch.sum((image - torch.tensor(self.mesh_image)) ** 2, dim=2)
        loss = torch.sum(diff_image)

        loss.backward()
        energy_data = loss.detach().numpy()

        grad_data = vertices_with_grad.grad

        (
            energy_rigid,
            grad_rigidity,
            _,
        ) = self.rigid_energy.evaluate(self.vertices)
        energy = energy_data + energy_rigid.numpy()
        print("Energy=%f : EData=%f E_rigid=%f" % (energy, energy_data, energy_rigid))

        # update v
        grad = grad_data + grad_rigidity

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(-grad.numpy(), self.step_factor_vertices, self.step_max_vertices)
        self.speed_vertices = (1 - self.damping) * (self.speed_vertices * inertia + (1 - inertia) * step_vertices)
        self.vertices = self.vertices + torch.tensor(self.speed_vertices)
        # update rotation
        step_quaternion = mult_and_clamp(
            -quaternion_with_grad.grad.numpy(),
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (self.speed_quaternion * inertia + (1 - inertia) * step_quaternion)
        self.transform_quaternion = self.transform_quaternion + self.speed_quaternion
        self.transform_quaternion = self.transform_quaternion / np.linalg.norm(self.transform_quaternion)

        # update translation
        step_translation = mult_and_clamp(
            -translation_with_grad.grad.numpy(),
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transform_translation = self.transform_translation + self.speed_translation
        # update directional light
        step = -light_directional_with_grad.grad.numpy() * 0.0001
        self.speed_light_directional = (1 - self.damping) * (
            self.speed_light_directional * inertia + (1 - inertia) * step
        )
        self.light_directional = self.light_directional + self.speed_light_directional
        # update ambient light
        step = -light_ambient_with_grad.grad.numpy() * 0.0001
        self.speed_light_ambient = (1 - self.damping) * (self.speed_light_ambient * inertia + (1 - inertia) * step)
        self.light_ambient = self.light_ambient + self.speed_light_ambient
        # update mesh color
        step = -mesh_color_with_grad.grad.numpy() * 0.00001
        self.speed_mesh_color = (1 - self.damping) * (self.speed_mesh_color * inertia + (1 - inertia) * step)
        self.mesh_color = self.mesh_color + self.speed_mesh_color

        self.iter += 1
        return energy, image.detach().numpy(), diff_image.detach().numpy()
