"""Implementation of triangulated meshes."""

import numpy as np

from scipy import sparse
from scipy.optimize import linprog

from .tools import cross_backward, normalize, normalize_backward


class TriMeshAdjacencies:
    """Class that stores adjacency matrices and methods that use this adjacencies.
    Unlike the TriMesh class there are no vertices stored in this class.
    """

    def __init__(self, faces, clockwise=False, nb_vertices=None):
        """
        unless specified explicitly the number of vertices is assumed to be np.max(faces)+1
        """
        assert np.issubdtype(faces.dtype, np.integer)
        assert faces.ndim == 2
        assert faces.shape[1] == 3

        self.faces = faces
        self.nb_faces = faces.shape[0]
        if nb_vertices is None:
            # unless specified explicitly the number of vertices is assumed to be np.max(faces.flat)+1
            nb_vertices = np.max(faces) + 1
        else:
            assert nb_vertices > np.max(faces)
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
        self.edge_vertices = np.zeros((self.nb_edges, 2), dtype=np.uint32)
        self.edge_vertices[id_edge] = edges

        nb_inc = np.zeros((self.nb_edges))
        np.add.at(nb_inc, id_edge, edge_increase)
        nb_dec = np.zeros((self.nb_edges))
        np.add.at(nb_dec, id_edge, ~edge_increase)
        self.is_non_manifold_edge = (unique_counts > 2) | (nb_inc > 1) | (nb_dec > 1)

        self.is_manifold = ~np.any(self.is_non_manifold_edge)
        self.is_closed = self.is_manifold and np.all(unique_counts == 2)

        self.edges_faces_signed = sparse.coo_matrix(
            (edge_increase * 2 - 1, (id_edge, id_faces)),
            shape=(self.nb_edges, self.nb_faces),
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
        self.Faces_Edges = sparse.coo_matrix(
            (id_edge, (id_faces, v)), shape=(self.nb_faces, 3)
        ).todense()
        self.adjacency_vertices = (
            (self._vertices_faces * self._vertices_faces.T) > 0
        ) - sparse.eye(self.nb_vertices)
        degree_v_f =  np.bincount(faces.flatten(),minlength=self.nb_vertices)
      
        self.degree_v_f = degree_v_f
        self.degree_v_e = self.adjacency_vertices.dot(
            np.ones((self.nb_vertices))
        )  # degree_v_e(i)=j means that the vertex i appears in j edges
        self.Laplacian = (
            sparse.diags([self.degree_v_e], [0], (self.nb_vertices, self.nb_vertices))
            - self.adjacency_vertices
        )
        self.hasBoundaries = np.any(np.sum(self.edges_faces_ones, axis=1) == 1)
        assert np.all(self.Laplacian * np.ones((self.nb_vertices)) == 0)
        self.store_backward = {}

    def boundary_edges(self):
        is_boundary_edge = np.array(
            np.sum(self.adjacencies.edges_faces_ones, axis=1) == 1
        ).squeeze(axis=1)
        return self.edge_vertices[is_boundary_edge, :]

    def id_edge(self, idv):

        return (
            np.maximum(idv[:, 0], idv[:, 1]).astype(np.uint64)
            + np.minimum(idv[:, 0], idv[:, 1]).astype(np.uint64) * self.nb_vertices,
            idv[:, 0] < idv[:, 1],
        )

    def compute_face_areas(self, vertices):
        triangles = vertices[self.faces, :]
        u = triangles[:, 1, :] - triangles[:, 0, :]
        v = triangles[:, 2, :] - triangles[:, 0, :]
        areas = np.sqrt(np.sum(np.cross(u, v) ** 2, axis=1))
        return areas

    def compute_face_normals(self, vertices):
        triangles = vertices[self.faces, :]
        u = triangles[:, 1, :] - triangles[:, 0, :]
        v = triangles[:, 2, :] - triangles[:, 0, :]
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
        triangles_b = np.stack((-u_b - v_b, u_b, v_b), axis=1)
        vertices_b = np.zeros((self.nb_vertices, 3))
        np.add.at(vertices_b, self.faces, triangles_b)
        return vertices_b

    def compute_vertex_normals(self, face_normals):
        n = self._vertices_faces * face_normals
        normals = normalize(n, axis=1)
        self.store_backward["compute_vertex_normals"] = n
        return normals

    def compute_vertex_normals_backward(self, normals_b):
        n = self.store_backward["compute_vertex_normals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        face_normals_b = self._vertices_faces.T * n_b
        return face_normals_b

    def edge_on_silhouette(self, vertices_2d):
        """Compute the a boolean for each of edges of each face that is true if
        and only if the edge is one the silhouette of the mesh given a view point
        """
        triangles = vertices_2d[self.faces, :]
        u = triangles[:, 1, :] - triangles[:, 0, :]
        v = triangles[:, 2, :] - triangles[:, 0, :]
        if self.clockwise:
            face_visible = np.cross(u, v) > 0
        else:
            face_visible = np.cross(u, v) < 0
        edge_bool = (self.edges_faces_ones * face_visible) == 1
        return edge_bool[self.Faces_Edges]


class TriMesh:
    """Class that implements a triangulated mesh."""

    def __init__(
        self,
        faces,
        vertices=None,
        nb_vertices=None,
        clockwise=False,
        compute_adjacencies=True,
    ):

        assert np.issubdtype(faces.dtype, np.integer)
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        assert np.all(faces >= 0)

        self.faces = faces
        self.max_face_index = np.max(faces)
        self.nb_faces = faces.shape[0]

        if nb_vertices is None:
            if vertices is None:
                raise BaseException(
                    "you need to not provide vertices or nb_vertices so that the number of vertices is known"
                )
            assert len(vertices.shape) == 2
            nb_vertices = vertices.shape[0]
        elif vertices is not None:
            assert len(vertices.shape) == 2
            assert nb_vertices == vertices.shape[0]

        self.nb_vertices = nb_vertices
        self.vertices = None
        self.face_normals = None
        self.vertex_normals = None
        self.clockwise = clockwise
        if vertices is not None:
            self.set_vertices(vertices)
        if compute_adjacencies:
            self.compute_adjacencies()

    def compute_adjacencies(self):
        self.adjacencies = TriMeshAdjacencies(
            self.faces, self.clockwise, nb_vertices=self.nb_vertices
        )
        if self.vertices is not None:
            if self.adjacencies.is_closed:
                self.check_orientation()

    def set_vertices(self, vertices):
        assert len(vertices.shape) == 2
        assert vertices.shape[1] == 3
        if vertices.shape[0] != self.nb_vertices:
            raise ValueError(f"Expecting {self.max_face_index+1} vertices.")

        self.vertices = vertices
        self.face_normals = None
        self.vertex_normals = None

    def compute_volume(self):
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

    def compute_face_normals(self):
        self.face_normals = self.adjacencies.compute_face_normals(self.vertices)

    def compute_vertex_normals(self):
        if self.face_normals is None:
            self.compute_face_normals()
        self.vertex_normals = self.adjacencies.compute_vertex_normals(self.face_normals)

    def compute_vertex_normals_backward(self, vertex_normals_b):
        self.face_normals_b = self.adjacencies.compute_vertex_normals_backward(
            vertex_normals_b
        )
        self.vertices_b += self.adjacencies.compute_face_normals_backward(
            self.face_normals_b
        )

    def edge_on_silhouette(self, points_2d):
        """Compute the a boolean for each of edges that is true if and only if
        the edge is one the silhouette of the mesh.
        """
        assert self.adjacencies.is_manifold
        return self.adjacencies.edge_on_silhouette(points_2d)

    def largest_manifold_subset_faces(self):
        """Return face indices to to keep to get the largest area manifold subset of the surface.
        This is formulated as a linear programming problem."""
        faces_candidates = (
            self.adjacencies.is_non_manifold_edge * self.adjacencies.edges_faces_ones
            >= 1
        )
        bounds = np.column_stack(
            (~faces_candidates, np.ones((self.adjacencies.nb_faces)))
        )
        b_ub = np.hstack(
            (np.ones((self.adjacencies.nb_edges)), np.ones((self.adjacencies.nb_edges)))
        )
        c = -self.adjacencies.compute_face_areas(self.vertices)
        if not isinstance(c, np.ndarray):
            c = self._data_as_numpy_array(c)
        A_ub = sparse.vstack(
            (
                self.adjacencies.edges_faces_signed > 0,
                self.adjacencies.edges_faces_signed < 0,
            )
        )
        result = linprog(
            c,
            A_ub,
            b_ub=b_ub,
            A_eq=None,
            b_eq=None,
            bounds=bounds,
            method="interior-point",
        )
        keep_face = result.x > 0.99
        return keep_face

    def largest_manifold_subset(self):
        keep_faces = self.largest_manifold_subset_faces()
        new_faces = self.faces[keep_faces, :]
        return TriMesh(new_faces, vertices=self.vertices)


class ColoredTriMesh(TriMesh):
    """Class that implements a colored triangulated mesh."""

    def __init__(
        self,
        faces,
        vertices=None,
        nb_vertices=None,
        clockwise=False,
        faces_uv=None,
        uv=None,
        texture=None,
        colors=None,
        nb_colors=None,
        compute_adjacencies=True,
    ):
        super(ColoredTriMesh, self).__init__(
            faces,
            vertices=vertices,
            nb_vertices=nb_vertices,
            clockwise=clockwise,
            compute_adjacencies=compute_adjacencies,
        )
        self.faces_uv = faces_uv
        self.uv = uv

        self.texture = texture
        self.vertices_colors = colors
        self.textured = not (self.texture is None)
        self.nb_colors = nb_colors
        if nb_colors is None and texture is  None and colors is None:
            raise BaseException("You need to provide either nb_colors, texture of colors")
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
    def from_trimesh(mesh, compute_adjacencies=True):  # inspired from pyrender
        """Get the vertex colors, texture coordinates, and material properties
        from a :class:`~trimesh.base.Trimesh`.
        """
        colors = None
        uv = None
        texture = None

        # If the trimesh visual is undefined, return none for both

        # Process vertex colors
        if mesh.visual.kind == "vertex":
            colors = mesh.visual.vertex_colors.copy()
            if colors.ndim == 2 and colors.shape[1] == 4:
                colors = colors[:, :3]
            colors = colors.astype(np.float) / 255

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
                texture.setflags(write=0)

                if texture.shape[2] == 4:
                    texture = texture[:, :, :3]  # removing alpha channel

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
        # manifold. trimesh seem to split vertices that have different uvs (using
        # unmerge_faces texture.py), making the surface not watertight, while there
        # were only seems in the texture

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

    def to_trimesh(self):
        # lazy modules loading
        import PIL
        import trimesh

        # largely inspired from trimesh's load_obj function

        v = self.vertices
        if not isinstance(v, np.ndarray):
            v = self._data_as_numpy_array(v)
        faces = self.faces
        if self.vertices_colors is not None:

            trimesh_mesh = trimesh.Trimesh(
                vertices=v, faces=faces, vertices_colors=self.vertices_colors
            )

        elif self.texture is not None:
            faces_tex = self.faces_uv

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

            trimesh_mesh = trimesh.Trimesh(
                vertices=new_vertices, faces=new_faces, visual=visual
            )
        else:
            trimesh_mesh = trimesh.Trimesh(
                vertices=v, faces=faces
            )            
        return trimesh_mesh
    

    def submesh(self, keep_faces):
        new_faces = self.faces[keep_faces, :]
        if self.vertices is None:
            raise BaseException("You need to provide the vertices first")
        if self.faces_uv is None:
            new_faces_uv = None
        else:
            new_faces_uv = self.faces_uv[keep_faces, :]
        compute_adjacencies = self.adjacencies is not None
        return self.__class__(
            faces=new_faces,
            vertices=self.vertices,
            clockwise=self.clockwise,
            faces_uv=new_faces_uv,
            uv=self.uv,
            texture=self.texture,
            colors=self.vertices_colors,
            nb_colors=self.nb_colors,
            compute_adjacencies=compute_adjacencies,
        )

    def largest_manifold_subset(self):
        keep_faces = self.largest_manifold_subset_faces()
        return self.submesh(keep_faces)
