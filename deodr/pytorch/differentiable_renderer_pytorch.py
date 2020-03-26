"""Pytorch interface to deodr."""

import numpy as np

import torch

from .. import differentiable_renderer_cython
from ..differentiable_renderer import Camera, Scene3D


class CameraPytorch(Camera):
    """Pytorch implementation of the camera class."""

    def __init__(self, extrinsic, intrinsic, height, width, distortion=None):
        super().__init__(
            extrinsic, intrinsic, height, width, distortion=distortion, checks=False
        )

    def world_to_camera(self, points_3d):
        assert isinstance(points_3d, torch.Tensor)
        return torch.cat(
            (points_3d, torch.ones((points_3d.shape[0], 1), dtype=torch.double)), dim=1
        ).mm(torch.tensor(self.extrinsic.T))

    def left_mul_intrinsic(self, projected):
        return torch.cat(
            (projected, torch.ones((projected.shape[0], 1), dtype=torch.double)), dim=1
        ).mm(torch.tensor(self.intrinsic[:2, :].T))

    def column_stack(self, values):
        return torch.stack(values, dim=1)


class TorchDifferentiableRenderer2DFunc(torch.autograd.Function):
    """Pytorch implementation of the 2D rendering function."""

    @staticmethod
    def forward(ctx, ij, colors, scene):
        nb_color_chanels = colors.shape[1]
        image = np.empty((scene.height, scene.width, nb_color_chanels))
        z_buffer = np.empty((scene.height, scene.width))
        ctx.scene = scene
        scene.ij = ij.detach().numpy()  # should automatically detached according to
        # https://pytorch.org/docs/master/notes/extending.html
        scene.colors = colors.detach().numpy()
        differentiable_renderer_cython.renderScene(scene, 1, image, z_buffer)
        ctx.save_for_backward(ij, colors)
        ctx.image = (
            image.copy()
        )  # making a copy to keep the antializaed image for visualization ,
        # could be optional
        ctx.z_buffer = z_buffer
        return torch.as_tensor(image)

    @staticmethod
    def backward(ctx, image_b):
        scene = ctx.scene
        scene.uv_b = np.zeros(scene.uv.shape)
        scene.ij_b = np.zeros(scene.ij.shape)
        scene.shade_b = np.zeros(scene.shade.shape)
        scene.colors_b = np.zeros(scene.colors.shape)
        scene.texture_b = np.zeros(scene.texture.shape)
        differentiable_renderer_cython.renderSceneB(
            scene, 1, ctx.image, ctx.z_buffer, image_b.numpy()
        )
        return torch.as_tensor(scene.ij_b), torch.as_tensor(scene.colors_b), None


TorchDifferentiableRender2D = TorchDifferentiableRenderer2DFunc.apply


class Scene3DPytorch(Scene3D):
    """Pytorch implementation of deodr 3D scenes."""

    def __init__(self):
        super().__init__()

    def set_light(self, light_directional, light_ambient):
        if not (isinstance(light_directional, torch.Tensor)):
            light_directional = torch.tensor(light_directional)
        self.light_directional = light_directional
        self.light_ambient = light_ambient

    def _compute_vertices_colors_with_illumination(self):
        vertices_luminosity = (
            torch.relu(
                -torch.sum(self.mesh.vertex_normals * self.light_directional, dim=1)
            )
            + self.light_ambient
        )
        return self.mesh.vertices_colors * vertices_luminosity[:, None]

    def _render_2d(self, ij, colors):
        self.depths = self.depths.detach()
        return TorchDifferentiableRender2D(ij, colors, self), self.depths
