from DEODR.tensorflow import Scene3DTensorflow, LaplacianRigidEnergyTensorflow
from DEODR.tensorflow import TriMeshTensorflow as TriMesh
from DEODR import LaplacianRigidEnergy
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy.spatial.transform.rotation
import copy
import cv2
import tensorflow as tf


def qrot(q, v):
    qr = tf.tile(q[None, :], (v.shape[0], 1))
    qvec = qr[:, :-1]
    uv = tf.linalg.cross(qvec, v)
    uuv = tf.linalg.cross(qvec, uv)
    return v + 2 * (qr[:, 3][None, 0] * uv + uuv)


class MeshDepthFitter:
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
            faces[:, ::-1].copy()
        )  # we do a copy to avoid negative stride not support by Tensorflow
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices, axis=0))
        self.cameraCenter = objectCenter + np.array([-0.5, 0, 5]) * objectRadius

        self.scene = Scene3DTensorflow()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = tf.constant(copy.copy(vertices))
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.setMeshTransformInit(euler=euler_init, translation=translation_init)
        self.reset()

    def setMeshTransformInit(self, euler, translation):
        self.transformQuaternionInit = scipy.spatial.transform.Rotation.from_euler(
            "zyx", euler
        ).as_quat()
        self.transformTranslationInit = translation

    def reset(self):
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices_init.shape)
        self.transformQuaternion = copy.copy(self.transformQuaternionInit)
        self.transformTranslation = copy.copy(self.transformTranslationInit)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

    def setMaxDepth(self, maxDepth):
        self.scene.maxDepth = maxDepth
        self.scene.setBackground(
            np.full((self.SizeH, self.SizeW, 1), maxDepth, dtype=np.float)
        )

    def setDepthScale(self, depthScale):
        self.depthScale = depthScale

    def setImage(self, handImage, focal=None):
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]
        assert handImage.ndim == 2
        self.handImage = handImage
        if focal is None:
            focal = 2 * self.SizeW

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = -R.T.dot(self.cameraCenter)
        self.CameraMatrix = np.array(
            [[focal, 0, self.SizeW / 2], [0, focal, self.SizeH / 2], [0, 0, 1]]
        ).dot(np.column_stack((R, T)))
        self.iter = 0

    def step(self):
        self.vertices = (
            self.vertices - tf.reduce_mean(self.vertices, axis=0)[None, :]
        )  # centervertices because we have another paramter to control translations
        x = tf.ones((2, 2))

        with tf.GradientTape() as tape:

            vertices_with_grad = tf.constant(self.vertices)
            quaternion_with_grad = tf.constant(self.transformQuaternion)
            translation_with_grad = tf.constant(self.transformTranslation)

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

            self.mesh.setVertices(vertices_with_grad_transformed)

            depth_scale = 1 * self.depthScale
            Depth = self.scene.renderDepth(
                self.CameraMatrix,
                resolution=(self.SizeW, self.SizeH),
                depth_scale=depth_scale,
            )
            Depth = tf.clip_by_value(Depth, 0, self.scene.maxDepth)

            diffImage = tf.reduce_sum(
                (Depth - tf.constant(self.handImage[:, :, None])) ** 2, axis=2
            )
            loss = tf.reduce_sum(diffImage)

            trainable_variables = [
                vertices_with_grad,
                quaternion_with_grad,
                translation_with_grad,
            ]
            vertices_grad, quaternion_grad, translation_grad = tape.gradient(
                loss, trainable_variables
            )

        EData = loss.numpy()

        GradData = vertices_grad

        E_rigid, grad_rigidity, approx_hessian_rigidity = self.rigidEnergy.eval(
            self.vertices.numpy()
        )
        Energy = EData + E_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (Energy, EData, E_rigid))

        # update v
        G = GradData + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        # update vertices
        step_vertices = mult_and_clamp(
            -G.numpy(), self.step_factor_vertices, self.step_max_vertices
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
        self.transformQuaternion = self.transformQuaternion + self.speed_quaternion
        # update translation
        self.transformQuaternion = self.transformQuaternion / np.linalg.norm(
            self.transformQuaternion
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
        self.transformTranslation = self.transformTranslation + self.speed_translation

        self.iter += 1
        return Energy, Depth[:, :, 0].numpy(), diffImage.numpy()


class MeshRGBFitterWithPose:
    def __init__(
        self,
        vertices,
        faces,
        euler_init,
        translation_init,
        defaultColor,
        defaultLight,
        cregu=2000,
        inertia=0.96,
        damping=0.05,
        updateLights=True,
        updateColor=True,
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

        self.defaultColor = defaultColor
        self.defaultLight = defaultLight
        self.updateLights = updateLights
        self.updateColor = updateColor
        self.mesh = TriMesh(faces.copy())  # we do a copy to avoid negative stride not support by Tensorflow
        objectCenter = vertices.mean(axis=0)+translation_init
        objectRadius = np.max(np.std(vertices, axis=0))
        self.cameraCenter = objectCenter + np.array([0, 0, 9]) * objectRadius

        self.scene = Scene3DTensorflow()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergyTensorflow(self.mesh, vertices, cregu)
        self.vertices_init = tf.constant(copy.copy(vertices))
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.setMeshTransformInit(euler=euler_init, translation=translation_init)
        self.reset()

    def setBackgroundColor(self, backgroundColor):
        self.scene.setBackground(
            np.tile(backgroundColor[None, None, :], (self.SizeH, self.SizeW, 1))
        )

    def setMeshTransformInit(self, euler, translation):
        self.transformQuaternionInit = scipy.spatial.transform.Rotation.from_euler(
            "zyx", euler
        ).as_quat()
        self.transformTranslationInit = translation

    def reset(self):
        self.vertices = copy.copy(self.vertices_init)
        self.speed_vertices = np.zeros(self.vertices.shape)
        self.transformQuaternion = copy.copy(self.transformQuaternionInit)
        self.transformTranslation = copy.copy(self.transformTranslationInit)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)

        self.handColor = copy.copy(self.defaultColor)
        self.ligthDirectional = copy.copy(self.defaultLight["directional"])
        self.ambiantLight = copy.copy(self.defaultLight["ambiant"])

        self.speed_ligthDirectional = np.zeros(self.ligthDirectional.shape)
        self.speed_ambiantLight = np.zeros(self.ambiantLight.shape)
        self.speed_handColor = np.zeros(self.handColor.shape)

    def setImage(self, handImage, focal=None):
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]
        assert handImage.ndim == 3
        self.handImage = handImage
        if focal is None:
            focal = 2 * self.SizeW

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = -R.T.dot(self.cameraCenter)
        self.CameraMatrix = np.array(
            [[focal, 0, self.SizeW / 2], [0, focal, self.SizeH / 2], [0, 0, 1]]
        ).dot(np.column_stack((R, T)))
        self.iter = 0

    def step(self):

        with tf.GradientTape() as tape:

            vertices_with_grad = tf.constant(self.vertices)
            quaternion_with_grad = tf.constant(self.transformQuaternion)
            translation_with_grad = tf.constant(self.transformTranslation)

            ligthDirectional_with_grad = tf.constant(self.ligthDirectional)
            ambiantLight_with_grad = tf.constant(self.ambiantLight)
            handColor_with_grad = tf.constant(self.handColor)

            tape.watch(vertices_with_grad)
            tape.watch(quaternion_with_grad)
            tape.watch(translation_with_grad)

            tape.watch(ligthDirectional_with_grad)
            tape.watch(ambiantLight_with_grad)
            tape.watch(handColor_with_grad)

            vertices_with_grad_centered = (
                vertices_with_grad - tf.reduce_mean(vertices_with_grad, axis=0)[None, :]
            )

            q_normalized = quaternion_with_grad / tf.norm(
                quaternion_with_grad
            )  # that will lead to a gradient that is in the tangeant space
            vertices_with_grad_transformed = (
                qrot(q_normalized, vertices_with_grad_centered) + translation_with_grad
            )
            self.mesh.setVertices(vertices_with_grad_transformed)

            self.scene.setLight(
                ligthDirectional=ligthDirectional_with_grad,
                ambiantLight=ambiantLight_with_grad,
            )
            self.mesh.setVerticesColors(
                tf.tile(handColor_with_grad[None, :], [self.mesh.nbV, 1])
            )

            Abuffer = self.scene.render(
                self.CameraMatrix, resolution=(self.SizeW, self.SizeH)
            )

            diffImage = tf.reduce_sum(
                (Abuffer - tf.constant(self.handImage)) ** 2, axis=2
            )
            loss = tf.reduce_sum(diffImage)

            trainable_variables = [
                vertices_with_grad,
                quaternion_with_grad,
                translation_with_grad,
                ligthDirectional_with_grad,
                ambiantLight_with_grad,
                handColor_with_grad,
            ]
            vertices_grad, quaternion_grad, translation_grad, ligthDirectional_grad, ambiantLight_grad, handColor_grad = tape.gradient(
                loss, trainable_variables
            )

        EData = loss.numpy()

        GradData = vertices_grad

        E_rigid, grad_rigidity, approx_hessian_rigidity = self.rigidEnergy.eval(
            self.vertices
        )
        Energy = EData + E_rigid.numpy()
        print("Energy=%f : EData=%f E_rigid=%f" % (Energy, EData, E_rigid))

        # update v
        G = GradData + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(
            -G.numpy(), self.step_factor_vertices, self.step_max_vertices
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
        self.transformQuaternion = self.transformQuaternion + self.speed_quaternion
        # update translation
        self.transformQuaternion = self.transformQuaternion / np.linalg.norm(
            self.transformQuaternion
        )
        step_translation = mult_and_clamp(
            -translation_grad, self.step_factor_translation, self.step_max_translation
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transformTranslation = self.transformTranslation + self.speed_translation
        # update directional light
        step = -ligthDirectional_grad * 0.0001
        self.speed_ligthDirectional = (1 - self.damping) * (
            self.speed_ligthDirectional * inertia + (1 - inertia) * step
        )
        self.ligthDirectional = self.ligthDirectional + self.speed_ligthDirectional
        # update ambiant light
        step = -ambiantLight_grad * 0.0001
        self.speed_ambiantLight = (1 - self.damping) * (
            self.speed_ambiantLight * inertia + (1 - inertia) * step
        )
        self.ambiantLight = self.ambiantLight + self.speed_ambiantLight
        # update hand color
        step = -handColor_grad * 0.00001
        self.speed_handColor = (1 - self.damping) * (
            self.speed_handColor * inertia + (1 - inertia) * step
        )
        self.handColor = self.handColor + self.speed_handColor

        self.iter += 1
        return Energy, Abuffer.numpy(), diffImage.numpy()
