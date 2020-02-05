from scipy import sparse
import numpy as np
import torch
from ..triangulated_mesh import *


def print_grad(name):
    def hook(grad):
        print(f"grad {name} = {grad}")

    return hook


class TriMeshAdjacenciesPytorch(TriMeshAdjacencies):
    def __init__(self, faces, clockwise=False):
        super().__init__(faces, clockwise)
        self.faces_torch = torch.LongTensor(faces)
        i = self.faces_torch.flatten()
        j = torch.LongTensor(np.tile(np.arange(self.nbF)[:, None], [1, 3]).flatten())
        v = np.ones((self.nbF, 3)).flatten()
        self.Vertices_Faces_torch = torch.sparse.DoubleTensor(
            torch.stack((i, j)),
            torch.ones((self.nbF, 3), dtype=torch.float64).flatten(),
            torch.Size((self.nbV, self.nbF)),
        )

    def computeFaceNormals(self, vertices):
        tris = vertices[self.faces_torch, :]
        u = tris[::, 1] - tris[::, 0]
        v = tris[::, 2] - tris[::, 0]
        if self.clockwise:
            n = -torch.cross(u, v)
        else:
            n = torch.cross(u, v)
        l2 = (n ** 2).sum(dim=1)
        l = l2.sqrt()
        nn = n / l[:, None]
        return nn

    def computeVertexNormals(self, faceNormals):
        n = self.Vertices_Faces_torch.mm(faceNormals)
        l2 = (n ** 2).sum(dim=1)
        l = l2.sqrt()
        return n / l[:, None]

    def edgeOnSilhouette(self, vertices, faceNormals, viewpoint):
        return super().edgeOnSilhouette(
            vertices.detach().numpy(), faceNormals.detach().numpy(), viewpoint
        )


class TriMeshPytorch(TriMesh):
    def __init__(self, faces, vertices=None,clockwise=False):
        super().__init__(faces,vertices,clockwise)

    def computeAdjacencies(self):
        self.adjacencies = TriMeshAdjacenciesPytorch(self.faces)
        
class ColoredTriMeshPytorch(TriMeshPytorch):
    def __init__(self, faces, vertices=None, clockwise=False,faces_uv=None,uv=None,texture=None,colors=None):
        super(ColoredTriMeshPytorch, self).__init__(faces,vertices=vertices,clockwise=clockwise)
        self.faces_uv = faces_uv
        self.uv = uv
        self.texture = texture    
        self.colors = colors
        self.textured = not (self.texture is None)