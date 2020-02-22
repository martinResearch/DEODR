import numpy as np
from . import differentiable_renderer_cython
import copy


class Camera:
    def __init__(
        self, extrinsic, intrinsic, resolution, dist=None, checks=True, tol=1e-6
    ):
        """camera with same distortion paramterization as opencv"""
        if checks:
            assert extrinsic.shape == (3, 4)
            assert intrinsic.shape == (3, 3)
            assert np.all(intrinsic[2, :] == [0, 0, 1])
            assert (
                np.linalg.norm(extrinsic[:3, :3].T.dot(extrinsic[:3, :3]) - np.eye(3))
                < tol
            )
            if dist is not None:
                assert len(dist) == 5

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.dist = dist
        self.resolution = resolution

    def worldToCamera(self, points3D):
        return points3D.dot(self.extrinsic[:3, :3].T) + self.extrinsic[:3, 3]

    def leftMulIntrinsic(self, projected):
        return projected.dot(self.intrinsic[:2, :2].T) + self.intrinsic[:2, 2]

    def column_stack(self, values):
        return np.column_stack(values)

    def projectPoints(
        self, points3D, get_jacobians=False, store_backward=None, return_depths=True
    ):  # similar to cv2.projectPoints
        pCam = self.worldToCamera(points3D)
        depths = pCam[:, 2]
        projected = pCam[:, :2] / depths[:, None]

        if self.dist is None:
            projectedImageCoordinates = self.leftMulIntrinsic(projected)
            if store_backward is not None:
                store_backward["projectPoints"] = (pCam, depths, projected)
        else:
            k1, k2, p1, p2, k3, = self.dist
            x = projected[:, 0]
            y = projected[:, 1]
            r2 = x ** 2 + y ** 2
            radialDistortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            tangentialDistortionx = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
            tangentialDistortiony = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
            distortedx = x * radialDistortion + tangentialDistortionx
            distortedy = y * radialDistortion + tangentialDistortiony
            distorted = self.column_stack((distortedx, distortedy))
            projectedImageCoordinates = self.leftMulIntrinsic(distorted)
            if store_backward is not None:
                store_backward["projectPoints"] = (
                    pCam,
                    depths,
                    projected,
                    r2,
                    radialDistortion,
                )

        if return_depths:
            return projectedImageCoordinates, depths
        else:
            return projectedImageCoordinates

    def projectPoints_backward(
        self, projectedImageCoordinates_b, store_backward, depths_b=None
    ):

        if self.dist is None:
            pCam, depths, projected = store_backward["projectPoints"]
            projected_b = projectedImageCoordinates_b.dot(
                self.intrinsic[:2, :2].T
            )  # not sure about transpose

        else:
            pCam, depths, projected, r2, radialDistortion = store_backward[
                "projectPoints"
            ]
            k1, k2, p1, p2, k3, = self.dist
            x = projected[:, 0]
            y = projected[:, 1]
            distorted_b = projectedImageCoordinates_b.dot(
                self.intrinsic[:2, :2].T
            )  # not sure about transpose
            distortedx_b = distorted_b[:, 0]
            distortedy_b = distorted_b[:, 1]
            x_b = distortedx_b * radialDistortion
            y_b = distortedy_b * radialDistortion
            radialDistortion_b = distortedx_b * x + distortedy_b * y
            tangentialDistortionx_b = distortedx_b
            tangentialDistortiony_b = distortedy_b
            x_b += tangentialDistortionx_b * (2 * p1 * y + p2 * 4 * x)
            y_b += tangentialDistortionx_b * 2 * p1 * x
            x_b += tangentialDistortiony_b * 2 * p2 * y
            y_b += tangentialDistortiony_b * (2 * p2 * x + p1 * 4 * y)
            r2_b = tangentialDistortionx_b * p2 + tangentialDistortiony_b * p1
            r2_b += radialDistortion_b * (k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2)
            x_b += r2_b * 2 * x
            y_b += r2_b * 2 * y
            projected_b = np.column_stack((x_b, y_b))

        pCam_b = np.column_stack(
            (
                projected_b / depths[:, None],
                -np.sum(projected_b * pCam[:, :2], axis=1) / (depths ** 2),
            )
        )
        if depths_b is not None:
            pCam_b[:, 2] += depths_b
        points3D_b = pCam_b.dot(self.extrinsic[:3, :3].T)

        return points3D_b

    def getCenter(self):
        return -self.extrinsic[:3, :3].T.dot(self.extrinsic[:, 3])


class Scene2DBase:
    """this class represents the structure representing the 2.5
    scene expect by the C++ code"""

    def __init__(
        self,
        faces,
        faces_uv,
        ij,
        depths,
        textured,
        uv,
        shade,
        colors,
        shaded,
        edgeflags,
        image_H,
        image_W,
        nbColors,
        texture,
        background,
        clockwise=False,
    ):
        self.faces = faces
        self.faces_uv = faces_uv
        self.ij = ij
        self.depths = depths
        self.textured = textured
        self.uv = uv
        self.shade = shade
        self.colors = colors
        self.shaded = shaded
        self.edgeflags = edgeflags
        self.image_H = image_H
        self.image_W = image_W
        self.nbColors = nbColors
        self.texture = texture
        self.background = background
        self.clockwise = clockwise


class Scene2D(Scene2DBase):
    """this class represents a 2.5D scene. It contains a set of 2D vertices with
    associated depths and a list of faces that are triplets of vertices indexes"""

    def __init__(
        self,
        faces,
        faces_uv,
        ij,
        depths,
        textured,
        uv,
        shade,
        colors,
        shaded,
        edgeflags,
        image_H,
        image_W,
        nbColors,
        texture,
        background,
        clockwise=False,
    ):
        self.faces = faces
        self.faces_uv = faces_uv
        self.ij = ij
        self.depths = depths
        self.textured = textured
        self.uv = uv
        self.shade = shade
        self.colors = colors
        self.shaded = shaded
        self.edgeflags = edgeflags
        self.image_H = image_H
        self.image_W = image_W
        self.nbColors = nbColors
        self.texture = texture
        self.background = background
        self.clockwise = clockwise

        # fields to store gradients
        self.uv_b = np.zeros(self.uv.shape)
        self.ij_b = np.zeros(self.ij.shape)
        self.shade_b = np.zeros(self.shade.shape)
        self.colors_b = np.zeros(self.colors.shape)
        self.texture_b = np.zeros(self.texture.shape)

    def clear_gradients(self):
        self.uv_b.fill(0)
        self.ij_b.fill(0)
        self.shade_b.fill(0)
        self.colors_b.fill(0)
        self.texture_b.fill(0)

    def render_error(self, Aobs, sigma=1):
        Abuffer = np.zeros((self.image_H, self.image_W, self.nbColors))
        Zbuffer = np.zeros((self.image_H, self.image_W))
        ErrBuffer = np.empty((self.image_H, self.image_W))
        antialiaseError = True
        differentiable_renderer_cython.renderScene(
            self, sigma, Abuffer, Zbuffer, antialiaseError, Aobs, ErrBuffer
        )
        self.store_backward = (sigma, Aobs, Abuffer, Zbuffer, ErrBuffer)
        return Abuffer, Zbuffer, ErrBuffer

    def render(self, sigma=1):
        Abuffer = np.zeros((self.image_H, self.image_W, self.nbColors))
        Zbuffer = np.zeros((self.image_H, self.image_W))
        antialiaseError = False
        differentiable_renderer_cython.renderScene(
            self, sigma, Abuffer, Zbuffer, antialiaseError, None, None
        )
        self.store_backward = (sigma, Abuffer, Zbuffer)
        return Abuffer, Zbuffer

    def render_error_backward(self, ErrBuffer_b, make_copies=True):
        sigma, Aobs, Abuffer, Zbuffer, ErrBuffer = self.store_backward
        antialiaseError = True
        if make_copies:
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                Abuffer,
                Zbuffer,
                None,
                antialiaseError,
                Aobs,
                ErrBuffer.copy(),
                ErrBuffer_b,
            )
        else:
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                Abuffer,
                Zbuffer,
                None,
                antialiaseError,
                Aobs,
                ErrBuffer,
                ErrBuffer_b,
            )

    def render_backward(self, Abuffer_b, make_copies=True):
        sigma, Abuffer, Zbuffer = self.store_backward
        antialiaseError = False
        if (
            make_copies
        ):  # if we make copies we keep the antialized image unchanged Abuffer
            # along the occlusion boundaries
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                Abuffer.copy(),
                Zbuffer,
                Abuffer_b,
                antialiaseError,
                None,
                None,
                None,
            )
        else:
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                Abuffer,
                Zbuffer,
                Abuffer_b,
                antialiaseError,
                None,
                None,
                None,
            )

    def render_compare_and_backward(
        self,
        sigma=1,
        antialiaseError=False,
        Aobs=None,
        mask=None,
        clear_gradients=True,
        make_copies=True,
    ):
        if mask is None:
            mask = np.ones((Aobs.shape[0], Aobs.shape[1]))
        if antialiaseError:
            Abuffer, Zbuffer, ErrBuffer = self.render_error(Aobs, sigma)
        else:
            Abuffer, Zbuffer = self.render(sigma)

        if clear_gradients:
            self.clear_gradients()

        if antialiaseError:
            ErrBuffer = ErrBuffer * mask
            Err = np.sum(ErrBuffer)
            ErrBuffer_b = copy.copy(mask)
            self.render_error_backward(ErrBuffer_b, make_copies=make_copies)
        else:
            diffImage = (Abuffer - Aobs) * mask[:, :, None]
            ErrBuffer = (diffImage) ** 2
            Err = np.sum(ErrBuffer)
            Abuffer_b = 2 * diffImage
            self.render_backward(Abuffer_b, make_copies=make_copies)

        return Abuffer, Zbuffer, ErrBuffer, Err


class Scene3D:
    """this class represents a 3D scene containing a single mesh, a directional light
    and an ambiant light. The parameter sigma control the width of
    antialiasing edge overdraw"""

    def __init__(self, sigma=1):
        self.mesh = None
        self.ligthDirectional = None
        self.ambiantLight = None
        self.sigma = sigma

    def clear_gradients(self):
        # fields to store gradients
        self.uv_b = np.zeros((self.mesh.nbV, 2))
        self.ij_b = np.zeros((self.mesh.nbV, 2))
        self.shade_b = np.zeros((self.mesh.nbV))
        self.colors_b = np.zeros(self.colors.shape)
        self.texture_b = np.zeros((0, 0))

    def setLight(self, ligthDirectional, ambiantLight):
        self.ligthDirectional = np.array(ligthDirectional)
        self.ambiantLight = ambiantLight

    def setMesh(self, mesh):
        self.mesh = mesh

    def setBackground(self, backgroundImage):
        assert backgroundImage.dtype == np.double
        self.background = backgroundImage

    def computeVerticesLuminosity(self):
        directional = np.maximum(
            0, -np.sum(self.mesh.vertexNormals * self.ligthDirectional, axis=1)
        )
        self.store_backward_current["computeVerticesLuminosity"] = directional
        return directional + self.ambiantLight

    def _computeVerticesColorsWithIllumination(self):

        verticesLuminosity = self.computeVerticesLuminosity()
        colors = self.mesh.verticesColors * verticesLuminosity[:, None]
        if self.store_backward_current is not None:
            self.store_backward_current[
                "computeVerticesColorsWithIllumination"
            ] = verticesLuminosity
        return colors

    def _computeVerticescolorsWithIllumination_backward(self, colors_b):
        verticesLuminosity = self.store_backward_current[
            "computeVerticesColorsWithIllumination"
        ]
        verticesLuminosity_b = np.sum(self.mesh.verticesColors * colors_b, axis=1)
        self.mesh.verticesColors_b = colors_b * verticesLuminosity[:, None]
        self.ambiantLight_b = np.sum(verticesLuminosity_b)
        directional_b = verticesLuminosity_b
        self.computeVerticesLuminosity_backward(directional_b)

    def computeVerticesLuminosity_backward(self, directional_b):
        directional = self.store_backward_current["computeVerticesLuminosity"]
        self.lightDirectional_b = -np.sum(
            ((directional_b * (directional > 0))[:, None]) * self.mesh.vertexNormals,
            axis=0,
        )
        self.vertexNormals_b = (
            -((directional_b * (directional > 0))[:, None]) * self.ligthDirectional
        )

    def _render2D(self, ij, colors):
        nbColorChanels = colors.shape[1]
        Abuffer = np.empty((self.image_H, self.image_W, nbColorChanels))
        Zbuffer = np.empty((self.image_H, self.image_W))
        self.ij = np.array(ij)
        self.colors = np.array(colors)
        differentiable_renderer_cython.renderScene(self, self.sigma, Abuffer, Zbuffer)

        if self.store_backward_current is not None:
            self.store_backward_current["render2D"] = (ij, colors, Abuffer, Zbuffer)

        return Abuffer, Zbuffer

    def _render2D_backward(self, Abuffer_b):
        ij, colors, Abuffer, Zbuffer = self.store_backward_current["render2D"]
        self.ij = np.array(ij)
        self.colors = np.array(colors)
        differentiable_renderer_cython.renderSceneB(
            self, self.sigma, Abuffer.copy(), Zbuffer, Abuffer_b
        )
        return self.ij_b, self.colors_b

    def render(self, camera, return_zbuffer=False):
        self.store_backward_current = {}
        self.mesh.computeVertexNormals()

        ij, depths = camera.projectPoints(
            self.mesh.vertices, store_backward=self.store_backward_current
        )

        # compute silhouette edges
        self.edgeflags = self.mesh.edgeOnSilhouette(ij)
        # construct 2D scene
        self.faces = self.mesh.faces.astype(np.uint32)

        self.depths = depths
        if self.mesh.uv is not None:
            self.uv = self.mesh.uv
            self.faces_uv = self.mesh.faces_uv
            self.textured = np.ones((self.mesh.nbF), dtype=np.bool)
            self.shade = self.computeVerticesLuminosity()
            self.shaded = np.ones(
                (self.mesh.nbF), dtype=np.bool
            )  # could eventually be non zero if we were using texture
            self.texture = self.mesh.texture
            colors = np.zeros((self.mesh.nbV, self.texture.shape[2]))
        else:
            colors = self._computeVerticesColorsWithIllumination()
            self.faces_uv = self.faces
            self.uv = np.zeros((self.mesh.nbV, 2))
            self.textured = np.zeros((self.mesh.nbF), dtype=np.bool)
            self.shade = np.zeros(
                (self.mesh.nbV), dtype=np.float
            )  # could eventually be non zero if we were using texture
            self.shaded = np.zeros(
                (self.mesh.nbF), dtype=np.bool
            )  # could eventually be non zero if we were using texture
            self.texture = np.zeros((0, 0))

        self.image_H = camera.resolution[1]
        self.image_W = camera.resolution[0]

        self.clockwise = self.mesh.clockwise
        Abuffer, Zbuffer = self._render2D(ij, colors)
        if self.store_backward_current is not None:
            self.store_backward_current["render"] = (
                camera,
                self.edgeflags,
            )  # store this field as it could be overwritten when
            # rendering several views
        if return_zbuffer:
            return Abuffer, Zbuffer
        else:
            return Abuffer

    def render_backward(self, Abuffer_b):

        camera, self.edgeflags = self.store_backward_current["render"]
        ij_b, colors_b = self._render2D_backward(Abuffer_b)
        self._computeVerticescolorsWithIllumination_backward(colors_b)
        self.mesh.vertices_b = camera.projectPoints_backward(
            ij_b, store_backward=self.store_backward_current
        )
        self.mesh.computeVertexNormals_backward(self.vertexNormals_b)

    def renderDepth(self, camera, resolution, depth_scale=1):
        self.store_backward_current = {}
        P2D, depths = camera.projectPoints(
            self.mesh.vertices, store_backward=self.store_backward_current
        )

        # compute silhouette edges
        self.mesh.computeFaceNormals()
        edge_bool = self.mesh.edgeOnSilhouette(P2D)

        self.faces = self.mesh.faces.astype(np.uint32)
        self.faces_uv = self.faces
        ij = P2D
        colors = depths[:, None] * depth_scale
        self.depths = depths
        self.edgeflags = edge_bool
        self.uv = np.zeros((self.mesh.nbV, 2))
        self.textured = np.zeros((self.mesh.nbF), dtype=np.bool)
        self.shade = np.zeros(
            (self.mesh.nbV), dtype=np.bool
        )  # eventually used when using texture
        self.image_H = resolution[1]
        self.image_W = resolution[0]
        self.shaded = np.zeros(
            (self.mesh.nbF), dtype=np.bool
        )  # eventually used when using texture
        self.texture = np.zeros((0, 0))
        self.clockwise = self.mesh.clockwise
        Abuffer, _ = self._render2D(ij, colors)
        if self.store_backward_current is not None:
            self.store_backward_current["renderDepth"] = (camera, depth_scale)
        return Abuffer

    def renderDepth_backward(self, Depth_b):
        camera, depth_scale = self.store_backward_current["renderDepth"]
        ij_b, colors_b = self._render2D_backward(Depth_b)
        depths_b = np.squeeze(colors_b * depth_scale, axis=1)
        self.mesh.vertices_b = camera.projectPoints_backward(
            ij_b, depths_b=depths_b, store_backward=self.store_backward_current
        )

    def renderDeffered(self, camera, depth_scale=1):

        P2D, depths = camera.projectPoints(self.mesh.vertices)

        # compute silhouette edges
        self.mesh.computeFaceNormals()
        edgeflags = self.mesh.edgeOnSilhouette(P2D)

        verticesLuminosity = self.computeVerticesLuminosity()

        # construct triangle soup (loosing connectivity), needed to render
        # discontinuous uv maps and face ids
        soup_nbF = self.mesh.nbF
        soup_nbV = 3 * self.mesh.nbF
        soup_faces = np.arange(0, soup_nbV, dtype=np.uint32).reshape(self.mesh.nbF, 3)
        soup_faces_uv = soup_faces
        soup_ij = P2D[self.mesh.faces].reshape(soup_nbV, 2)
        soup_xyz = self.mesh.vertices[self.mesh.faces].reshape(soup_nbV, 3)
        soup_faceids = np.tile(np.arange(0, self.mesh.nbF)[:, None], (1, 3)).reshape(
            soup_nbV, 1
        )
        soup_depths = depths[self.mesh.faces].reshape(soup_nbV, 1)
        soup_normals = self.mesh.vertexNormals[self.mesh.faces].reshape(soup_nbV, 3)
        soup_luminosity = verticesLuminosity[self.mesh.faces].reshape(soup_nbV, 1)

        if self.mesh.uv is None:
            soup_vcolors = self.mesh.verticesColor[self.mesh.faces]
            colors = np.column_stack(
                (
                    soup_depths[:, None] * depth_scale,
                    soup_faceids[:, :, None],
                    soup_normals,
                    soup_luminosity[:, :, None],
                    soup_vcolors,
                    soup_xyz,
                )
            )
        else:
            soup_uv = self.mesh.uv[self.mesh.faces_uv].reshape(soup_nbV, 2)
            colors = np.column_stack(
                (
                    soup_depths * depth_scale,
                    soup_faceids,
                    soup_normals,
                    soup_luminosity,
                    soup_uv,
                    soup_xyz,
                )
            )

        nbColors = colors.shape[1]
        uv = np.zeros((soup_nbV, 2))
        textured = np.zeros((soup_nbF), dtype=np.bool)
        shade = np.zeros((soup_nbV), dtype=np.bool)

        image_H = camera.resolution[1]
        image_W = camera.resolution[0]
        shaded = np.zeros(
            (soup_nbF), dtype=np.bool
        )  # eventually used when using texture
        texture = np.zeros((0, 0))

        background = np.zeros((image_H, image_W, nbColors))
        background[:, :, 0] = depths.max()
        scene2D = Scene2DBase(
            faces=soup_faces,
            faces_uv=soup_faces_uv,
            ij=soup_ij,
            depths=soup_depths,
            textured=textured,
            uv=uv,
            shade=shade,
            colors=colors,
            shaded=shaded,
            edgeflags=edgeflags,
            image_H=image_H,
            image_W=image_W,
            nbColors=nbColors,
            texture=texture,
            background=background,
        )
        Abuffer = np.empty((self.image_H, self.image_W, nbColors))
        Zbuffer = np.empty((self.image_H, self.image_W))
        differentiable_renderer_cython.renderScene(scene2D, 0, Abuffer, Zbuffer)

        if self.mesh.uv is not None:
            return {
                "depth": Abuffer[:, :, 0],
                "faceid": Abuffer[:, :, 1],
                "normal": Abuffer[:, :, 2:4],
                "luminosity": Abuffer[:, :, 5],
                "uv": Abuffer[:, :, 6:8],
                "xyz": Abuffer[:, :, 8:],
            }

        else:
            pass
