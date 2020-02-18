from DEODR import Scene3D, Camera, LaplacianRigidEnergy
from DEODR import LaplacianRigidEnergy
from DEODR import TriMesh, ColoredTriMesh
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy.spatial.transform.rotation
import copy
import cv2
from .tools import *


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
            faces, vertices=vertices
        )  # we do a copy to avoid negative stride not support by pytorch
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices, axis=0))
        self.cameraCenter = objectCenter + np.array([-0.5, 0, 5]) * objectRadius

        self.scene = Scene3D()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = copy.copy(vertices)
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

    def setImage(self, handImage, focal=None, dist=None):
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]
        assert handImage.ndim == 2
        self.handImage = handImage
        if focal is None:
            focal = 2 * self.SizeW
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = -R.T.dot(self.cameraCenter)
        intrinsic = np.array(
            [[focal, 0, self.SizeW / 2], [0, focal, self.SizeH / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((R, T))
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            dist=dist,
            resolution=(self.SizeW, self.SizeH),
        )
        self.iter = 0

    def render(self):
        q_normalized = normalize(
            self.transformQuaternion
        )  # that will lead to a gradient that is in the tangeant space
        vertices_transformed = (
            qrot(q_normalized, self.vertices) + self.transformTranslation
        )
        self.mesh.setVertices(vertices_transformed)
        self.DepthNotCliped = self.scene.renderDepth(
            self.camera,
            resolution=(self.SizeW, self.SizeH),
            depth_scale=self.depthScale,
        )
        Depth = np.clip(self.DepthNotCliped, 0, self.scene.maxDepth)
        return Depth

    def render_backward(self, Depth_b):
        self.scene.clear_gradients()
        Depth_b[self.DepthNotCliped < 0] = 0
        Depth_b[self.DepthNotCliped > self.scene.maxDepth] = 0
        self.scene.renderDepth_backward(Depth_b)
        vertices_transformed_b = self.scene.mesh.vertices_b
        self.transformTranslation_b = np.sum(vertices_transformed_b, axis=0)
        q_normalized = normalize(self.transformQuaternion)
        q_normalized_b, self.vertices_b = qrot_backward(
            q_normalized, self.vertices, vertices_transformed_b
        )
        self.transformQuaternion_b = normalize_backward(
            self.transformQuaternion, q_normalized_b
        )  # that will lead to a gradient that is in the tangeant space
        return

    def step(self):

        self.vertices = self.vertices - np.mean(self.vertices, axis=0)[None, :]
        Depth = self.render()

        diffImage = np.sum((Depth - self.handImage[:, :, None]) ** 2, axis=2)
        EData = np.sum(diffImage)
        Depth_b = 2 * (Depth - self.handImage[:, :, None])
        self.render_backward(Depth_b)

        self.vertices_b = self.vertices_b - np.mean(self.vertices_b, axis=0)[None, :]
        GradData = self.vertices_b
        # update v

        E_rigid, grad_rigidity, approx_hessian_rigidity = self.rigidEnergy.eval(
            self.vertices
        )
        Energy = EData + E_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (Energy, EData, E_rigid))

        # update v
        G = GradData + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia
        # update vertices
        step_vertices = mult_and_clamp(
            -G, self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * self.inertia + (1 - self.inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -self.transformQuaternion_b,
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transformQuaternion = self.transformQuaternion + self.speed_quaternion
        self.transformQuaternion = self.transformQuaternion / np.linalg.norm(
            self.transformQuaternion
        )
        # update translation
        step_translation = mult_and_clamp(
            -self.transformTranslation_b,
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transformTranslation = self.transformTranslation + self.speed_translation

        self.iter += 1
        return Energy, Depth[:, :, 0], diffImage


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
        self.mesh = ColoredTriMesh(faces.copy(), vertices=vertices, nbColors=3)
        objectCenter = vertices.mean(axis=0) + translation_init
        objectRadius = np.max(np.std(vertices, axis=0))
        self.cameraCenter = objectCenter + np.array([0, 0, 9]) * objectRadius

        self.scene = Scene3D()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = copy.copy(vertices)
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

    def setImage(self, handImage, focal=None, dist=None):
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]
        assert handImage.ndim == 3
        self.handImage = handImage
        if focal is None:
            focal = 2 * self.SizeW

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = -R.T.dot(self.cameraCenter)
        intrinsic = np.array(
            [[focal, 0, self.SizeW / 2], [0, focal, self.SizeH / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((R, T))
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            dist=dist,
            resolution=(self.SizeW, self.SizeH),
        )
        self.iter = 0

    def render(self):
        q_normalized = normalize(
            self.transformQuaternion
        )  # that will lead to a gradient that is in the tangeant space
        vertices_transformed = (
            qrot(q_normalized, self.vertices) + self.transformTranslation
        )
        self.mesh.setVertices(vertices_transformed)
        self.scene.setLight(
            ligthDirectional=self.ligthDirectional, ambiantLight=self.ambiantLight
        )
        self.mesh.setVerticesColors(np.tile(self.handColor, (self.mesh.nbV, 1)))
        Abuffer = self.scene.render(self.camera)
        return Abuffer

    def render_backward(self, Abuffer_b):
        self.scene.clear_gradients()
        self.scene.render_backward(Abuffer_b)
        self.handColor_b = np.sum(self.mesh.verticesColors_b, axis=0)
        self.ligthDirectional_b = self.scene.lightDirectional_b
        self.ambiantLight_b = self.scene.ambiantLight_b
        vertices_transformed_b = self.scene.mesh.vertices_b
        self.transformTranslation_b = np.sum(vertices_transformed_b, axis=0)
        q_normalized = normalize(self.transformQuaternion)
        q_normalized_b, self.vertices_b = qrot_backward(
            q_normalized, self.vertices, vertices_transformed_b
        )
        self.transformQuaternion_b = normalize_backward(
            self.transformQuaternion, q_normalized_b
        )  # that will lead to a gradient that is in the tangeant space
        return

    def step(self):
        self.vertices = self.vertices - np.mean(self.vertices, axis=0)[None, :]

        Abuffer = self.render()

        diffImage = np.sum((Abuffer - self.handImage) ** 2, axis=2)
        Abuffer_b = 2 * (Abuffer - self.handImage)
        EData = np.sum(diffImage)

        E_rigid, grad_rigidity, approx_hessian_rigidity = self.rigidEnergy.eval(
            self.vertices
        )
        Energy = EData + E_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (Energy, EData, E_rigid))

        self.render_backward(Abuffer_b)

        self.vertices_b = self.vertices_b - np.mean(self.vertices_b, axis=0)[None, :]
        # update v
        G = self.vertices_b + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(
            -G, self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * inertia + (1 - inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -self.transformQuaternion_b,
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transformQuaternion = self.transformQuaternion + self.speed_quaternion
        self.transformQuaternion = self.transformQuaternion / np.linalg.norm(
            self.transformQuaternion
        )
        # update translation
        step_translation = mult_and_clamp(
            -self.transformTranslation_b,
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transformTranslation = self.transformTranslation + self.speed_translation
        # update directional light
        step = -self.ligthDirectional_b * 0.0001
        self.speed_ligthDirectional = (1 - self.damping) * (
            self.speed_ligthDirectional * inertia + (1 - inertia) * step
        )
        self.ligthDirectional = self.ligthDirectional + self.speed_ligthDirectional
        # update ambiant light
        step = -self.ambiantLight_b * 0.0001
        self.speed_ambiantLight = (1 - self.damping) * (
            self.speed_ambiantLight * inertia + (1 - inertia) * step
        )
        self.ambiantLight = self.ambiantLight + self.speed_ambiantLight
        # update hand color
        step = -self.handColor_b * 0.00001
        self.speed_handColor = (1 - self.damping) * (
            self.speed_handColor * inertia + (1 - inertia) * step
        )
        self.handColor = self.handColor + self.speed_handColor

        self.iter += 1
        return Energy, Abuffer, diffImage


class MeshRGBFitterWithPoseMultiFrame:
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
        self.mesh = ColoredTriMesh(faces, vertices, nbColors=3)
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices, axis=0))
        self.cameraCenter = objectCenter + np.array([0, 0, 6]) * objectRadius

        self.scene = Scene3D()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = copy.copy(vertices)
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

    def setImages(self, handImages, focal=None):
        self.SizeW = handImages[0].shape[1]
        self.SizeH = handImages[0].shape[0]
        assert handImages[0].ndim == 3
        self.handImages = handImages
        if focal is None:
            focal = 2 * self.SizeW

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = -R.T.dot(self.cameraCenter)
        intrinsic = np.array(
            [[focal, 0, self.SizeW / 2], [0, focal, self.SizeH / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((R, T))
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=(self.SizeW, self.SizeH),
        )
        self.iter = 0

    def setImage(self, handImage, focal=None):
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]
        assert handImage.ndim == 3
        self.handImage = handImage
        if focal is None:
            focal = 2 * self.SizeW

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = -R.T.dot(self.cameraCenter)
        intrinsic = np.array(
            [[focal, 0, self.SizeW / 2], [0, focal, self.SizeH / 2], [0, 0, 1]]
        )
        extrinsic = np.column_stack((R, T))
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=(self.SizeW, self.SizeH),
        )
        self.iter = 0

    def render(self, idframe=None):
        unormalized_quaternion = self.transformQuaternion[idframe]
        translation = self.transformTranslation[idframe]
        q_normalized = normalize(
            unormalized_quaternion
        )  # that will lead to a gradient that is in the tangeant space
        vertices_transformed = (
            qrot(q_normalized, self.vertices) + self.transformTranslation[idframe]
        )
        self.mesh.setVertices(vertices_transformed)
        self.scene.setLight(
            ligthDirectional=self.ligthDirectional, ambiantLight=self.ambiantLight
        )
        self.mesh.setVerticesColors(np.tile(self.handColor, (self.mesh.nbV, 1)))
        Abuffer = self.scene.render(self.camera)
        self.store_backward["render"] = (idframe, unormalized_quaternion, q_normalized)
        return Abuffer

    def clear_gradients(self):
        self.ligthDirectional_b = np.zeros(self.ligthDirectional.shape)
        self.ambiantLight_b = np.zeros(self.ambiantLight.shape)
        self.vertices_b = np.zeros(self.vertices.shape)
        self.transformQuaternion_b = np.zeros(self.transformQuaternion.shape)
        self.transformTranslation_b = np.zeros(self.transformTranslation.shape)
        self.handColor_b = np.zeros(self.handColor.shape)
        self.store_backward = {}

    def render_backward(self, Abuffer_b):
        idframe, unormalized_quaternion, q_normalized = self.store_backward["render"]
        self.scene.clear_gradients()
        self.scene.render_backward(Abuffer_b)
        self.handColor_b += np.sum(self.mesh.verticesColors_b, axis=0)
        self.ligthDirectional_b += self.scene.lightDirectional_b
        self.ambiantLight_b += self.scene.ambiantLight_b
        vertices_transformed_b = self.scene.mesh.vertices_b
        self.transformTranslation_b[idframe] += np.sum(vertices_transformed_b, axis=0)
        q_normalized_b, vertices_b = qrot_backward(
            q_normalized, self.vertices, vertices_transformed_b
        )
        self.vertices_b += vertices_b
        self.transformQuaternion_b[idframe] += normalize_backward(
            unormalized_quaternion, q_normalized_b
        )  # that will lead to a gradient that is in the tangeant space
        return

    def step(self):
        self.vertices = self.vertices - np.mean(self.vertices, axis=0)[None, :]

        self.nbFrames = len(self.handImages)

        Abuffer = [None] * self.nbFrames
        diffImage = [None] * self.nbFrames
        Abuffer_b = [None] * self.nbFrames
        EDatas = [None] * self.nbFrames
        self.clear_gradients()
        coefData = 1 / self.nbFrames
        for idframe in range(self.nbFrames):
            Abuffer[idframe] = self.render(idframe=idframe)
            diffImage[idframe] = np.sum(
                (Abuffer[idframe] - self.handImages[idframe]) ** 2, axis=2
            )
            Abuffer_b = coefData * 2 * (Abuffer[idframe] - self.handImages[idframe])
            EDatas[idframe] = coefData * np.sum(diffImage[idframe])
            self.render_backward(Abuffer_b)
        EData = np.sum(EDatas)
        E_rigid, grad_rigidity, approx_hessian_rigidity = self.rigidEnergy.eval(
            self.vertices
        )
        Energy = EData + E_rigid
        print("Energy=%f : EData=%f E_rigid=%f" % (Energy, EData, E_rigid))

        self.vertices_b = self.vertices_b - np.mean(self.vertices_b, axis=0)[None, :]
        # update v
        G = self.vertices_b + grad_rigidity

        def mult_and_clamp(x, a, t):
            return np.minimum(np.maximum(x * a, -t), t)

        inertia = self.inertia

        # update vertices
        step_vertices = mult_and_clamp(
            -G, self.step_factor_vertices, self.step_max_vertices
        )
        self.speed_vertices = (1 - self.damping) * (
            self.speed_vertices * inertia + (1 - inertia) * step_vertices
        )
        self.vertices = self.vertices + self.speed_vertices
        # update rotation
        step_quaternion = mult_and_clamp(
            -self.transformQuaternion_b,
            self.step_factor_quaternion,
            self.step_max_quaternion,
        )
        self.speed_quaternion = (1 - self.damping) * (
            self.speed_quaternion * inertia + (1 - inertia) * step_quaternion
        )
        self.transformQuaternion = self.transformQuaternion + self.speed_quaternion
        self.transformQuaternion = self.transformQuaternion / np.linalg.norm(
            self.transformQuaternion
        )
        # update translation
        step_translation = mult_and_clamp(
            -self.transformTranslation_b,
            self.step_factor_translation,
            self.step_max_translation,
        )
        self.speed_translation = (1 - self.damping) * (
            self.speed_translation * inertia + (1 - inertia) * step_translation
        )
        self.transformTranslation = self.transformTranslation + self.speed_translation
        # update directional light
        step = -self.ligthDirectional_b * 0.0001
        self.speed_ligthDirectional = (1 - self.damping) * (
            self.speed_ligthDirectional * inertia + (1 - inertia) * step
        )
        self.ligthDirectional = self.ligthDirectional + self.speed_ligthDirectional
        # update ambiant light
        step = -self.ambiantLight_b * 0.0001
        self.speed_ambiantLight = (1 - self.damping) * (
            self.speed_ambiantLight * inertia + (1 - inertia) * step
        )
        self.ambiantLight = self.ambiantLight + self.speed_ambiantLight
        # update hand color
        step = -self.handColor_b * 0.00001
        self.speed_handColor = (1 - self.damping) * (
            self.speed_handColor * inertia + (1 - inertia) * step
        )
        self.handColor = self.handColor + self.speed_handColor

        self.iter += 1
        return Energy, Abuffer, diffImage
