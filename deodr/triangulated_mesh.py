from scipy import sparse
import numpy as np
from .tools import normalize, normalize_backward, cross_backward


class TriMeshAdjacencies:
    """this class stores adjacency matrices and methods that use this adjacencies.
    Unlike the TriMesh class there are no vertices stored in this class"""

    def __init__(self, faces, clockwise=False):
        self.faces = faces
        self.nb_faces = faces.shape[0]
        self.nb_vertices = np.max(faces.flat) + 1
        i = self.faces.flatten()
        j = np.tile(np.arange(self.nb_faces)[:, None], [1, 3]).flatten()
        v = np.ones((self.nb_faces, 3)).flatten()
        self._vertices_Faces = sparse.coo_matrix(
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

        nb_inc = np.zeros((self.nb_edges))
        np.add.at(nb_inc, id_edge, edge_increase)
        nb_dec = np.zeros((self.nb_edges))
        np.add.at(nb_dec, id_edge, ~edge_increase)
        self.isManifold = (
            np.all(unique_counts <= 2) and np.all(nb_inc <= 1) and np.all(nb_dec <= 1)
        )
        self.is_closed = self.isManifold and np.all(unique_counts == 2)

        self.edges_faces_Ones = sparse.coo_matrix(
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
        self.Faces_Edges = sparse.coo_matrix(
            (id_edge, (id_faces, v)), shape=(self.nb_faces, 3)
        ).todense()
        self.adjacency_vertices = (
            (self._vertices_Faces * self._vertices_Faces.T) > 0
        ) - sparse.eye(self.nb_vertices)
        self.degree_v_e = self.adjacency_vertices.dot(
            np.ones((self.nb_vertices))
        )  # degree_v_e(i)=j means that the vertex i appears in j edges
        self.Laplacian = (
            sparse.diags([self.degree_v_e], [0], (self.nb_vertices, self.nb_vertices))
            - self.adjacency_vertices
        )
        self.hasBoundaries = np.any(np.sum(self.edges_faces_Ones, axis=1) == 1)
        assert np.all(self.Laplacian * np.ones((self.nb_vertices)) == 0)
        self.store_backward = {}

    def id_edge(self, idv):

        return (
            np.maximum(idv[:, 0], idv[:, 1])
            + np.minimum(idv[:, 0], idv[:, 1]) * self.nb_vertices,
            idv[:, 0] < idv[:, 1],
        )

    def compute_face_normals(self, vertices):
        tris = vertices[self.faces, :]
        u = tris[:, 1, :] - tris[:, 0, :]
        v = tris[:, 2, :] - tris[:, 0, :]
        if self.clockwise:
            n = -np.cross(u, v)
        else:
            n = np.cross(u, v)
        normals = normalize(n, axis=1)
        self.store_backward["compute_face_normals"] = (u, v, n)
        return normals

    def compute_face_normals_backward(self, normals_b):
        u, v, n = self.store_backward["compute_face_normals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        if self.clockwise:
            u_b, v_b = cross_backward(u, v, -n_b)
        else:
            u_b, v_b = cross_backward(u, v, n_b)
        tris_b = np.stack((-u_b - v_b, u_b, v_b), axis=1)
        vertices_b = np.zeros((self.nb_vertices, 3))
        np.add.at(vertices_b, self.faces, tris_b)
        return vertices_b

    def compute_vertex_normals(self, face_normals):
        n = self._vertices_Faces * face_normals
        normals = normalize(n, axis=1)
        self.store_backward["compute_vertex_normals"] = n
        return normals

    def compute_vertex_normals_backward(self, normals_b):
        n = self.store_backward["compute_vertex_normals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        face_normals_b = self._vertices_Faces.T * n_b
        return face_normals_b

    def edge_on_silhouette(self, vertices_2d):
        """this computes the a boolean for each of edges of each face that is true if
        and only if the edge is one the silhouette of the mesh given a view point"""
        tris = vertices_2d[self.faces, :]
        u = tris[:, 1, :] - tris[:, 0, :]
        v = tris[:, 2, :] - tris[:, 0, :]
        if self.clockwise:
            face_visible = np.cross(u, v) > 0
        else:
            face_visible = np.cross(u, v) < 0
        edge_bool = (self.edges_faces_Ones * face_visible) == 1
        return edge_bool[self.Faces_Edges]


class TriMesh:
    def __init__(self, faces, vertices=None, clockwise=False):

        self.faces = faces
        self.nb_vertices = np.max(faces) + 1
        self.nb_faces = faces.shape[0]

        self.vertices = None
        self.faceNormals = None
        self.vertex_normals = None
        self.clockwise = clockwise
        self.compute_adjacencies()
        assert self.adjacencies.isManifold

        if vertices is not None:
            self.set_vertices(vertices)
            self.check_orientation()

    def compute_adjacencies(self):
        self.adjacencies = TriMeshAdjacencies(self.faces, self.clockwise)

    def set_vertices(self, vertices):
        self.vertices = vertices
        self.faceNormals = None
        self.vertex_normals = None

    def compute_volume(self):
        """Compute the volume enclosed by the triangulated surface. It assumes the
        surfaces is a closed manifold. This is done by summing the volumes of the
        simplices formed by joining the origin and the vertices of each triangle"""
        return (
            (1 if self.clockwise else -1)
            * np.sum(
                np.linalg.det(
                    np.dstack(
                        (
                            self.vertices[self.faces[:, 0]],
                            self.vertices[self.faces[:, 1]],
                            self.vertices[self.faces[:, 2]],
                        )
                    )
                )
            )
            / 6
        )

    def check_orientation(self):
        """check the mesh faces are properly oriented for the normals to point
         outward"""
        if self.compute_volume() > 0:
            raise (
                BaseException(
                    "The volume within the surface is negative. It seems that you faces"
                    "are not oriented cooreclt accourding to the clockwise flag"
                )
            )

    def compute_face_normals(self):
        self.faceNormals = self.adjacencies.compute_face_normals(self.vertices)

    def compute_vertex_normals(self):
        if self.faceNormals is None:
            self.compute_face_normals()
        self.vertex_normals = self.adjacencies.compute_vertex_normals(self.faceNormals)

    def compute_vertex_normals_backward(self, vertex_normals_b):
        self.faceNormals_b = self.adjacencies.compute_vertex_normals_backward(
            vertex_normals_b
        )
        self.vertices_b += self.adjacencies.compute_face_normals_backward(
            self.faceNormals_b
        )

    def edge_on_silhouette(self, points_2d):
        """this computes the a boolean for each of edges that is true if and only if
        the edge is one the silhouette of the mesh"""

        return self.adjacencies.edge_on_silhouette(points_2d)


class ColoredTriMesh(TriMesh):
    def __init__(
        self,
        faces,
        vertices=None,
        clockwise=False,
        faces_uv=None,
        uv=None,
        texture=None,
        colors=None,
        nb_colors=None,
    ):
        super(ColoredTriMesh, self).__init__(
            faces, vertices=vertices, clockwise=clockwise
        )
        self.faces_uv = faces_uv
        self.uv = uv

        self.texture = texture
        self.colors = colors
        self.textured = not (self.texture is None)
        self.nb_colors = nb_colors
        if nb_colors is None:
            if texture is None:
                self.nb_colors = colors.shape[1]
            else:
                self.nb_colors = texture.shape[2]

    def set_vertices_colors(self, colors):
        self.vertices_colors = colors

    def plot_uv_map(self, ax):
        ax.imshow(self.texture)
        ax.triplot(self.uv[:, 0], self.uv[:, 1], self.faces_uv)

    def plot(self, ax, plot_normals=False):
        x, y, z = self.vertices.T
        u, v, w = self.vertex_normals.T
        ax.plot_trisurf(
            self.vertices[:, 0],
            self.vertices[:, 1],
            Z=self.vertices[:, 2],
            triangles=self.faces,
        )
        ax.quiver(x, y, z, u, v, w, length=0.03, normalize=True, color=[0, 1, 0])

    @staticmethod
    def from_trimesh(mesh):  # inspired from pyrender
        """Gets the vertex colors, texture coordinates, and material properties
        from a :class:`~trimesh.base.Trimesh`.
        """
        colors = None
        uv = None
        texture = None

        # If the trimesh visual is undefined, return none for both

        # Process vertex colors
        if mesh.visual.kind == "vertex":
            colors = mesh.visual.vertex_colors.copy()

        # Process face colors
        elif mesh.visual.kind == "face":
            raise BaseException("not suported yet, will need antialisaing at the seams")

        # Process texture colors
        elif mesh.visual.kind == "texture":
            # Configure UV coordinates
            if mesh.visual.uv is not None:

                texture = np.array(mesh.visual.material.image) / 255
                texture.setflags(write=0)

                if texture.shape[2] == 4:
                    texture = texture[:, :, :3]  # removing alpha channel

                uv = np.column_stack(
                    (
                        (mesh.visual.uv[:, 0]) * texture.shape[0],
                        (1 - mesh.visual.uv[:, 1]) * texture.shape[1],
                    )
                )

        # merge identical 3D vertices even if their uv are different to keep surface
        # manifold. trimesh seem to split vertices that have different uvs (using
        # unmerge_faces texture.py), making the surface not watertight, while there
        # were only seems in the texture

        vertices, return_index, inv_ids = np.unique(
            mesh.vertices, axis=0, return_index=True, return_inverse=True
        )
        faces = inv_ids[mesh.faces].astype(np.uint32)
        if colors:
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
        )
