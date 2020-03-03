import numpy as np
from .. import differentiable_renderer_cython
import torch
from ..differentiable_renderer import Scene3D, Camera


class CameraPytorch(Camera):
    def __init__(self, extrinsic, intrinsic, resolution, distortion=None):
        super().__init__(
            extrinsic, intrinsic, resolution, distortion=distortion, checks=False
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
    def __init__(self):
        super().__init__()

    def set_light(self, ligth_directional, ambient_light):
        if not (isinstance(ligth_directional, torch.Tensor)):
            ligth_directional = torch.tensor(ligth_directional)
        self.ligth_directional = ligth_directional
        self.ambient_light = ambient_light

    def _compute_vertices_colors_with_illumination(self):
        vertices_luminosity = (
            torch.relu(
                -torch.sum(self.mesh.vertex_normals * self.ligth_directional, dim=1)
            )
            + self.ambient_light
        )
        return self.mesh.vertices_colors * vertices_luminosity[:, None]

    def _render_2d(self, ij, colors):
        self.depths = self.depths.detach()
        return TorchDifferentiableRender2D(ij, colors, self), self.depths
