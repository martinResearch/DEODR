"""Implementation of triangulated meshes."""
from typing import Any, Optional, Tuple

import numpy as np
from typing import Dict


from scipy import sparse

from .tools import cross_backward, normalize, normalize_backward

try:
    from trimesh.base import Trimesh
except ImportError:
    Trimesh = None


class TriMeshAdjacencies:
    """Class that stores sparse adjacency matrices and methods that use these matrices.
    Unlike the TriMesh class there are no vertices stored in this class.
    """

    def __init__(
        self,
        faces: np.ndarray,
        clockwise: bool = False,
        nb_vertices: Optional[int] = None,
    ):
        faces = np.array(faces)
        assert faces.ndim == 2
        assert faces.shape[1] == 3

        self.faces = faces

        self.nb_faces = int(faces.shape[0])
        if nb_vertices is None:
            nb_vertices = int(np.max(faces.flat)) + 1
        self.nb_vertices = nb_vertices
        i = self.faces.flatten()
        j = np.tile(np.arange(self.nb_faces)[:, None], [1, 3]).flatten()
        v = np.ones((self.nb_faces, 3)).flatten()
        self._vertices_faces = sparse.coo_matrix(
            (v, (i, j)), shape=(self.nb_vertices, self.nb_faces)
        )
        id_faces = np.hstack(
            (
                np.arange(self.nb_faces),
                np.arange(self.nb_faces),
                np.arange(self.nb_faces),
            )
        )
        self.clockwise = clockwise
        edges = np.vstack(
            (self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]])
        )

        id_edge_tmp, edge_increase = self.id_edge(edges)

        _, id_edge, unique_counts = np.unique(
            id_edge_tmp, return_inverse=True, return_counts=True
        )

        self.nb_edges = np.max(id_edge) + 1
        self.edges = np.zeros((self.nb_edges, 2), dtype=np.uint32)
        self.edges[id_edge] = edges

        nb_inc = np.zeros((self.nb_edges))
        np.add.at(nb_inc, id_edge, edge_increase)
        nb_dec = np.zeros((self.nb_edges))
        np.add.at(nb_dec, id_edge, ~edge_increase)
        self.is_manifold = (
            np.all(unique_counts <= 2) and np.all(nb_inc <= 1) and np.all(nb_dec <= 1)
        )
        self.is_closed = self.is_manifold and np.all(unique_counts == 2)

        self.edges_vertices_ones = sparse.coo_matrix(
            (
                np.ones((2 * len(id_edge))),
                (np.tile(id_edge[:, None], (1, 2)).flatten(), edges.flatten()),
            ),
            shape=(self.nb_edges, self.nb_vertices),
        )

        self.edges_faces_ones = sparse.coo_matrix(
            (np.ones((len(id_edge))), (id_edge, id_faces)),
            shape=(self.nb_edges, self.nb_faces),
        )
        v = np.hstack(
            (
                np.full((self.nb_faces), 0),
                np.full((self.nb_faces), 1),
                np.full((self.nb_faces), 2),
            )
        )
        self.faces_edges = sparse.coo_matrix(
            (id_edge, (id_faces, v)), shape=(self.nb_faces, 3)
        ).todense()
        self.adjacency_vertices = (
            (self._vertices_faces * self._vertices_faces.T) > 0
        ) - sparse.eye(self.nb_vertices)

        self.degree_v_f = self._vertices_faces.dot(np.ones((self.nb_faces)))

        self.degree_v_e = self.adjacency_vertices.dot(
            np.ones((self.nb_vertices))
        )  # degree_v_e(i)=j means that the vertex i appears in j edges
        self.laplacian = (
            sparse.diags([self.degree_v_e], [0], (self.nb_vertices, self.nb_vertices))
            - self.adjacency_vertices
        )
        self.hasBoundaries = np.any(np.sum(self.edges_faces_ones, axis=1) == 1)
        assert np.all(self.laplacian * np.ones((self.nb_vertices)) == 0)
        self.store_backward: Dict[str, Any] = {}

    def boundary_edges(self) -> np.ndarray:
        is_boundary_edge = np.array(np.sum(self.edges_faces_ones, axis=1) == 1).squeeze(
            axis=1
        )
        return self.edges[is_boundary_edge, :]

    def id_edge(self, idv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert idv.ndim == 2
        assert idv.shape[1] == 2
        return (
            np.maximum(idv[:, 0], idv[:, 1]).astype(np.uint64)
            + np.minimum(idv[:, 0], idv[:, 1]).astype(np.uint64) * self.nb_vertices,
            idv[:, 0] < idv[:, 1],
        )

    def compute_face_normals(self, vertices: np.ndarray) -> np.ndarray:
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        triangles = vertices[self.faces, :]
        u = triangles[:, 1, :] - triangles[:, 0, :]
        v = triangles[:, 2, :] - triangles[:, 0, :]
        n = -np.cross(u, v) if self.clockwise else np.cross(u, v)
        normals = normalize(n, axis=1)
        self.store_backward["compute_face_normals"] = (u, v, n)
        return normals

    def compute_face_normals_backward(self, normals_b: np.ndarray) -> np.ndarray:
        assert normals_b.ndim == 2
        assert normals_b.shape[1] == 3
        u, v, n = self.store_backward["compute_face_normals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        if self.clockwise:
            u_b, v_b = cross_backward(u, v, -n_b)
        else:
            u_b, v_b = cross_backward(u, v, n_b)
        triangles_b = np.stack((-u_b - v_b, u_b, v_b), axis=1)
        vertices_b = np.zeros((self.nb_vertices, 3))
        np.add.at(vertices_b, self.faces, triangles_b)
        return vertices_b

    def compute_vertex_normals(self, face_normals: np.ndarray) -> np.ndarray:
        assert face_normals.ndim == 2
        assert face_normals.shape[1] == 3
        n = self._vertices_faces * face_normals
        normals = normalize(n, axis=1)
        self.store_backward["compute_vertex_normals"] = n
        return normals

    def compute_vertex_normals_backward(self, normals_b: np.ndarray) -> np.ndarray:
        assert normals_b.ndim == 2
        assert normals_b.shape[1] == 3
        n = self.store_backward["compute_vertex_normals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        return self._vertices_faces.T * n_b

    def edge_on_silhouette(self, vertices_2d: np.ndarray) -> np.ndarray:
        """Compute the a boolean for each of edges of each face that is true if
        and only if the edge is one the silhouette of the mesh given a view point
        """
        assert vertices_2d.ndim == 2
        assert vertices_2d.shape[1] == 2
        triangles = vertices_2d[self.faces, :]
        u = triangles[:, 1, :] - triangles[:, 0, :]
        v = triangles[:, 2, :] - triangles[:, 0, :]
        face_visible = np.cross(u, v) > 0 if self.clockwise else np.cross(u, v) < 0
        edge_bool = (self.edges_faces_ones * face_visible) == 1
        return edge_bool[self.faces_edges]


class TriMesh:
    """Class that implements a triangulated mesh."""

    def __init__(
        self,
        faces: np.ndarray,
        vertices: np.ndarray,
        clockwise: bool = False,
        compute_adjacencies: bool = True,
    ):
        faces = np.array(faces)
        assert np.issubdtype(faces.dtype, np.integer)
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        assert np.all(faces >= 0)

        self._faces = faces
        self.nb_vertices = int(np.max(faces)) + 1
        self.nb_faces = int(faces.shape[0])

        self._face_normals: Optional[np.ndarray] = None
        self._vertex_normals: Optional[np.ndarray] = None
        self.clockwise = clockwise

        self._vertices_b = np.zeros((self.nb_vertices, 3))

        self.set_vertices(vertices)
        if compute_adjacencies:
            self.compute_adjacencies()

    def compute_adjacencies(self) -> None:
        self._adjacencies = TriMeshAdjacencies(
            self.faces, self.clockwise, nb_vertices=self.nb_vertices
        )

        if self._adjacencies.is_closed:
            self.check_orientation()

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @property
    def adjacencies(self) -> TriMeshAdjacencies:
        if self._adjacencies is None:
            self.compute_adjacencies()
        return self._adjacencies

    def set_vertices(self, vertices: np.ndarray) -> None:
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        self._vertices = vertices
        self._face_normals = None
        self._vertex_normals = None
        self._vertices_b = np.zeros((self.nb_vertices, 3))

    def compute_volume(self) -> float:
        """Compute the volume enclosed by the triangulated surface. It assumes the
        surfaces is a closed manifold. This is done by summing the volumes of the
        simplices formed by joining the origin and the vertices of each triangle.
        """
        return (
            (1 if self.clockwise else -1)
            * np.sum(
                np.linalg.det(
                    np.dstack(
                        (
                            self.vertices[self._faces[:, 0]],
                            self.vertices[self._faces[:, 1]],
                            self.vertices[self._faces[:, 2]],
                        )
                    )
                )
            )
            / 6
        )

    def check_orientation(self) -> None:
        """Check the mesh faces are properly oriented for the normals to point
        outward.
        """
        if self.compute_volume() > 0:
            raise (
                BaseException(
                    "The volume within the surface is negative. It seems that you faces"
                    "are not oriented correctly according to the clockwise flag"
                )
            )

    def compute_face_normals(self) -> None:
        self._face_normals = self.adjacencies.compute_face_normals(self.vertices)

    @property
    def face_normals(self) -> np.ndarray:
        """Return the face normals.

        face normals evaluation is done in a lazy manner.
        """
        if self._face_normals is None:
            self.compute_face_normals()
        assert self._face_normals is not None
        return self._face_normals

    def compute_vertex_normals(self) -> None:
        self._vertex_normals = self.adjacencies.compute_vertex_normals(
            self.face_normals
        )

    @property
    def vertex_normals(self) -> np.ndarray:
        """Return the vertices normals.

        face normals evaluation is done in a lazy manner.
        """

        if self._vertex_normals is None:
            self.compute_vertex_normals()
        assert self._vertex_normals is not None
        return self._vertex_normals

    def compute_vertex_normals_backward(self, vertex_normals_b: np.ndarray) -> None:
        self._face_normals_b = self.adjacencies.compute_vertex_normals_backward(
            vertex_normals_b
        )
        self._vertices_b += self.adjacencies.compute_face_normals_backward(
            self._face_normals_b
        )

    def edge_on_silhouette(self, points_2d: np.ndarray) -> np.ndarray:
        """Compute the a boolean for each of edges that is true if and only if
        the edge is one the silhouette of the mesh.
        """
        assert self.adjacencies.is_manifold
        return self.adjacencies.edge_on_silhouette(points_2d)


class ColoredTriMesh(TriMesh):
    """Class that implements a colored triangulated mesh."""

    def __init__(
        self,
        faces: np.ndarray,
        vertices: np.ndarray,
        clockwise: bool = False,
        faces_uv: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        texture: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        nb_colors: Optional[int] = None,
        compute_adjacencies: bool = True,
    ):
        super(ColoredTriMesh, self).__init__(
            faces,
            vertices=vertices,
            clockwise=clockwise,
            compute_adjacencies=compute_adjacencies,
        )
        self.faces_uv = faces_uv
        self.uv = uv

        self.texture = texture
        self.vertices_colors = colors
        self.textured = self.texture is not None
        self.nb_colors = nb_colors
        if nb_colors is None:
            if texture is None:
                assert (
                    colors is not None
                ), "You need to provide at least on among nb_colors, texture or colors"
                self.nb_colors = colors.shape[1]
            else:
                assert (
                    texture is not None
                ), "You need to provide at least on among nb_colors, texture or colors"
                self.nb_colors = texture.shape[2]

        self.vertices_colors_b: Optional[np.ndarray] = None

    def set_vertices_colors(self, colors: np.ndarray) -> None:
        self.vertices_colors = colors

    def plot_uv_map(self, ax: Any) -> None:
        assert self.uv is not None, "You need to provide a uv to display the uv map"
        if self.texture is not None:
            ax.imshow(self.texture)
        ax.triplot(self.uv[:, 0], self.uv[:, 1], self.faces_uv)

    def plot(self, ax: Any) -> None:
        assert self.vertices is not None, "You need to provide vertices first"
        x, y, z = self.vertices.T
        u, v, w = self.vertex_normals.T
        ax.plot_trisurf(
            self.vertices[:, 0],
            self.vertices[:, 1],
            Z=self.vertices[:, 2],
            triangles=self.faces,
        )
        ax.quiver(x, y, z, u, v, w, length=0.03, normalize=True, color=[0, 1, 0])

    def subdivise(self, n_iter: int) -> "ColoredTriMesh":
        """loop subdivision.

        https://graphics.stanford.edu/~mdfisher/subdivision.html"""
        return loop_subdivision(self, n_iter)

    @staticmethod
    def from_trimesh(
        mesh: Trimesh, compute_adjacencies: bool = True
    ) -> "ColoredTriMesh":  # inspired from pyrender
        """Get the vertex colors, texture coordinates, and material properties
        from a :class:`~trimesh.base.Trimesh`.
        """
        colors = None
        uv = None
        texture: Optional[np.ndarray] = None

        # If the trimesh visual is undefined, return none for both

        # Process vertex colors
        if mesh.visual.kind == "vertex":
            colors = mesh.visual.vertex_colors.copy()
            if colors.ndim == 2 and colors.shape[1] == 4:
                colors = colors[:, :3]
            colors = colors.astype(np.float64) / 255

        # Process face colors
        elif mesh.visual.kind == "face":
            raise BaseException(
                "not supported yet, will need antialiasing at the seams"
            )

        # Process texture colors
        elif mesh.visual.kind == "texture":
            # Configure UV coordinates
            if mesh.visual.uv is not None:

                texture = np.array(mesh.visual.material.image) / 255
                texture.setflags(write=False)

                if texture.shape[2] == 4:
                    texture = texture[:, :, :3]  # removing alpha channel
                assert texture is not None  # helping mypy

                uv = (
                    np.column_stack(
                        (
                            (mesh.visual.uv[:, 0]) * texture.shape[1],
                            (1 - mesh.visual.uv[:, 1]) * texture.shape[0],
                        )
                    )
                    - 0.5
                )

        # merge identical 3D vertices even if their uv are different to keep surface
        # manifold. Trimesh seems to split vertices that have different uvs (using
        # unmerge_faces texture.py), making the surface not watertight, while there
        # were only seems in the texture.

        vertices, return_index, inv_ids = np.unique(
            mesh.vertices, axis=0, return_index=True, return_inverse=True
        )
        faces = inv_ids[mesh.faces].astype(np.uint32)
        if colors is not None:
            colors2 = colors[return_index, :]
            if np.any(colors != colors2[inv_ids, :]):
                raise (
                    BaseException(
                        "vertices at the same 3D location should have the same color"
                        "for the rendering to be differentiable"
                    )
                )
        else:
            colors2 = None

        return ColoredTriMesh(
            faces,
            vertices,
            clockwise=False,
            faces_uv=np.array(mesh.faces),
            uv=uv,
            texture=texture,
            colors=colors2,
            compute_adjacencies=compute_adjacencies,
        )

    def to_trimesh(self) -> Trimesh:
        # lazy modules loading
        import PIL
        import trimesh

        # largely inspired from trimesh's load_obj function

        if self.vertices_colors is not None:
            raise BaseException(
                "Conversion to trimesh with per vertex color not support yet"
            )

        v = self.vertices
        faces = self.faces
        faces_tex = self.faces_uv

        assert self.uv is not None, "Only mesh with texture supported."
        assert self.texture is not None, "Only mesh with texture supported."

        vt = np.column_stack(
            (
                (self.uv[:, 0] + 0.5) / self.texture.shape[1],
                1 - ((self.uv[:, 1] + 0.5) / self.texture.shape[0]),
            )
        )
        new_faces, mask_v, mask_vt = trimesh.visual.texture.unmerge_faces(
            faces, faces_tex
        )
        assert np.allclose(v[faces], v[mask_v][new_faces])
        assert new_faces.max() < len(v[mask_v])

        new_vertices = v[mask_v].copy()
        uv = vt[mask_vt].copy()

        texture_uint8 = np.clip(self.texture * 255, 0, 255).astype(np.uint8)
        if texture_uint8.shape[2] == 1:
            texture_uint8 = texture_uint8.squeeze(axis=2)
        texture_pil = PIL.Image.fromarray(texture_uint8)
        material = trimesh.visual.material.SimpleMaterial(image=texture_pil)
        visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)

        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, visual=visual)

    @staticmethod
    def load(filename: str) -> "ColoredTriMesh":
        import trimesh

        mesh_trimesh = trimesh.load(filename)
        return ColoredTriMesh.from_trimesh(mesh_trimesh)


def loop_subdivision(mesh: ColoredTriMesh, n_iter: int = 1) -> ColoredTriMesh:
    """Loop subdivision.

    https://graphics.stanford.edu/~mdfisher/subdivision.html"""

    if n_iter == 0:
        return mesh

    if n_iter > 1:
        mesh = loop_subdivision(mesh, n_iter - 1)

    edge_mid_points = (
        mesh.adjacencies.edges_faces_ones
        * (mesh.adjacencies._vertices_faces.T * mesh.vertices)
        / 8
    ) + (1 / 8) * np.sum(mesh.vertices[mesh.adjacencies.edges, :], axis=1)

    # edge_mid_points = 0.5 * np.sum(self.vertices[self.adjacencies.edges, :], axis=1)

    beta = (3 / 8) * (1 / mesh.adjacencies.degree_v_e)
    moved_points = (
        beta[:, None] * (mesh.adjacencies.adjacency_vertices * mesh.vertices)
        + (5 / 8) * mesh.vertices
    )
    # moved_points = self.vertices

    new_vertices = np.vstack((moved_points, edge_mid_points))
    faces1 = np.column_stack(
        (
            mesh.faces[:, 0],
            mesh.adjacencies.faces_edges[:, 0] + mesh.nb_vertices,
            mesh.adjacencies.faces_edges[:, 2] + mesh.nb_vertices,
        )
    )
    faces2 = np.column_stack(
        (
            mesh.faces[:, 1],
            mesh.adjacencies.faces_edges[:, 1] + mesh.nb_vertices,
            mesh.adjacencies.faces_edges[:, 0] + mesh.nb_vertices,
        )
    )
    faces3 = np.column_stack(
        (
            mesh.faces[:, 2],
            mesh.adjacencies.faces_edges[:, 2] + mesh.nb_vertices,
            mesh.adjacencies.faces_edges[:, 1] + mesh.nb_vertices,
        )
    )
    faces4 = np.column_stack(
        (
            mesh.adjacencies.faces_edges[:, 0] + mesh.nb_vertices,
            mesh.adjacencies.faces_edges[:, 1] + mesh.nb_vertices,
            mesh.adjacencies.faces_edges[:, 2] + mesh.nb_vertices,
        )
    )
    new_faces = np.row_stack((faces1, faces2, faces3, faces4))
    if mesh.uv is not None:
        raise BaseException("Textured mesh not supported yet in subdivision.")
    if mesh.vertices_colors is not None:
        edge_mid_points_colors = np.mean(
            mesh.vertices_colors[mesh.adjacencies.edges, :], axis=1
        )
        new_colors = np.vstack((mesh.vertices_colors, edge_mid_points_colors))
    else:
        new_colors = None
    return ColoredTriMesh(
        vertices=new_vertices,
        faces=new_faces,
        colors=new_colors,
        nb_colors=mesh.nb_colors,
    )
