# type: ignore
"""Pytorch interface to deodr."""

from typing import Any, List, Optional, Union, Tuple
import numpy as np

import torch

from .. import differentiable_renderer_cython  # type: ignore
from ..differentiable_renderer import Camera, Scene3D


class CameraPytorch(Camera):
    """Pytorch implementation of the camera class."""

    def __init__(
        self,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        height: int,
        width: int,
        distortion: Optional[np.ndarray] = None,
    ):
        super().__init__(
            extrinsic, intrinsic, height, width, distortion=distortion, checks=False
        )

    def world_to_camera(self, points_3d: torch.Tensor) -> torch.Tensor:
        assert isinstance(points_3d, torch.Tensor)
        return torch.cat(
            (points_3d, torch.ones((points_3d.shape[0], 1), dtype=torch.double)), dim=1
        ).mm(torch.tensor(self.extrinsic.T))

    def left_mul_intrinsic(self, projected: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (projected, torch.ones((projected.shape[0], 1), dtype=torch.double)), dim=1
        ).mm(torch.tensor(self.intrinsic[:2, :].T))

    def column_stack(
        self, values: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        return torch.stack(values, dim=1)


class TorchDifferentiableRenderer2DFunc(torch.autograd.Function):
    """Pytorch implementation of the 2D rendering function."""

    @staticmethod
    def forward(  # type: ignore
        ctx: Any, ij: torch.Tensor, colors: torch.Tensor, scene: "Scene3DPytorch"
    ) -> torch.Tensor:
        nb_color_channels = colors.shape[1]
        image = np.empty(
            (scene.scene_2d.height, scene.scene_2d.width, nb_color_channels)
        )
        z_buffer = np.empty((scene.scene_2d.height, scene.scene_2d.width))
        ctx.scene = scene
        scene.scene_2d.ij = (
            ij.detach().numpy()
        )  # should automatically detached according to
        # https://pytorch.org/docs/master/notes/extending.html
        scene.colors = colors.detach().numpy()
        differentiable_renderer_cython.renderScene(scene.scene_2d, 1, image, z_buffer)
        ctx.save_for_backward(ij, colors)
        ctx.image = image.copy()
        # making a copy to keep the antializaed image for visualization ,
        # could be optional
        ctx.z_buffer = z_buffer
        return torch.as_tensor(image)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        assert len(grad_outputs) == 1
        image_b = grad_outputs[0]
        scene = ctx.scene
        scene.scene_2d.uv_b = np.zeros(scene.scene_2d.uv.shape)
        scene.scene_2d.ij_b = np.zeros(scene.scene_2d.ij.shape)
        scene.scene_2d.shade_b = np.zeros(scene.scene_2d.shade.shape)
        scene.scene_2d.colors_b = np.zeros(scene.scene_2d.colors.shape)
        scene.scene_2d.texture_b = np.zeros(scene.scene_2d.texture.shape)
        differentiable_renderer_cython.renderSceneB(
            scene.scene_2d, 1, ctx.image, ctx.z_buffer, image_b.numpy()
        )
        return (
            torch.as_tensor(scene.scene_2d.ij_b),
            torch.as_tensor(scene.scene_2d.colors_b),
            None,
        )


TorchDifferentiableRender2D = TorchDifferentiableRenderer2DFunc.apply


class Scene3DPytorch(Scene3D):
    """Pytorch implementation of deodr 3D scenes."""

    def __init__(self) -> None:
        super().__init__()

    def set_light(
        self, light_directional: torch.Tensor, light_ambient: torch.Tensor
    ) -> None:
        if not (isinstance(light_directional, torch.Tensor)):
            light_directional = torch.tensor(light_directional)
        self.light_directional = light_directional
        self.light_ambient_pytorch = light_ambient

    def _compute_vertices_colors_with_illumination(self) -> torch.Tensor:
        assert self.mesh is not None
        vertices_luminosity = (
            torch.relu(
                -torch.sum(self.mesh.vertex_normals * self.light_directional, dim=1)
            )
            + self.light_ambient_pytorch
        )
        return self.mesh.vertices_colors * vertices_luminosity[:, None]

    def _render_2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.depths = self.scene_2d.depths.detach()
        return (
            TorchDifferentiableRender2D(self.scene_2d.ij, self.scene_2d.colors, self),
            self.depths,
        )
