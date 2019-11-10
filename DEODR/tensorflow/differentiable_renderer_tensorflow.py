import numpy as np
from .. import differentiable_renderer_cython
import tensorflow as tf
import copy
from ..differentiable_renderer import Scene3D


def TensorflowDifferentiableRender2D(ij, colors, scene):
    @tf.custom_gradient
    def forward(
        ij, colors
    ):  # using inner function as we don't differentate w.r.t scene
        nbColorChanels = colors.shape[1]
        Abuffer = np.empty((scene.image_H, scene.image_W, nbColorChanels))
        Zbuffer = np.empty((scene.image_H, scene.image_W))
        scene.ij = np.array(
            ij
        )  # should automatically detached according to https://pytorch.org/docs/master/notes/extending.html
        scene.colors = np.array(colors)
        scene.depths = np.array(scene.depths)
        differentiable_renderer_cython.renderScene(scene, 1, Abuffer, Zbuffer)

        def backward(Abuffer_b):
            scene.uv_b = np.zeros(scene.uv.shape)
            scene.ij_b = np.zeros(scene.ij.shape)
            scene.shade_b = np.zeros(scene.shade.shape)
            scene.colors_b = np.zeros(scene.colors.shape)
            scene.texture_b = np.zeros(scene.texture.shape)
            Abuffer_copy = (
                Abuffer.copy()
            )  # making a copy to avoid removing antializaing on the image returned by the forward pass (the c++ backpropagation undo antializating), could be optional if we don't care about getting aliased images
            differentiable_renderer_cython.renderSceneB(
                scene, 1, Abuffer_copy, Zbuffer, Abuffer_b.numpy()
            )
            return tf.constant(scene.ij_b), tf.constant(scene.colors_b)

        return tf.convert_to_tensor(Abuffer), backward

    return forward(ij, colors)


class Scene3DTensorflow(Scene3D):
    def __init__(self):
        super().__init__()

    def setLight(self, ligthDirectional, ambiantLight):
        if not (isinstance(ligthDirectional, tf.Tensor)):
            ligthDirectional = tf.constant(ligthDirectional)
        self.ligthDirectional = ligthDirectional
        self.ambiantLight = ambiantLight

    def _cameraProject(self, cameraMatrix, P3D):
        assert isinstance(P3D, tf.Tensor)
        r = tf.linalg.matmul(
            tf.concat((P3D, tf.ones((P3D.shape[0], 1), dtype=P3D.dtype)), axis=1),
            tf.constant(cameraMatrix.T),
        )
        depths = r[:, 2]
        P2D = r[:, :2] / depths[:, None]
        return P2D, depths

    def _computeVerticesColorsWithIllumination(self):
        verticesLuminosity = (
            tf.nn.relu(
                -tf.reduce_sum(self.mesh.vertexNormals * self.ligthDirectional, axis=1)
            )
            + self.ambiantLight
        )
        return self.mesh.verticesColors * verticesLuminosity[:, None]

    def _render2D(self, ij, colors):

        return TensorflowDifferentiableRender2D(ij, colors, self)
