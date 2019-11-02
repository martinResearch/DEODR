from  scipy import sparse
import numpy as np
import torch
from ..triangulated_mesh import *

class TriMeshAdjacenciesPytorch(TriMeshAdjacencies):
	def __init__(self,faces):
		super().__init__(faces)
		self.faces_torch = torch.LongTensor(faces)
		i = self.faces_torch.flatten()
		j = torch.LongTensor(np.tile(np.arange(self.nbF)[:,None],[1,3]).flatten())
		v = np.ones((self.nbF,3)).flatten()		
		self.Vertices_Faces_torch = torch.sparse.DoubleTensor(torch.stack((i,j)),torch.ones((self.nbF,3),dtype=torch.float64).flatten(), torch.Size((self.nbV,self.nbF)))
	
	def computeFaceNormals(self,vertices):
		tris = vertices[self.faces_torch,:]
		n = torch.cross( tris[::,1 ] - tris[::,0], tris[::,2 ] - tris[::,0] )
		l = ((n**2).sum(dim = 1)).sqrt()
		return n/l[:,None]
		
	def computeVertexNormals(self,faceNormals):
		n = self.Vertices_Faces_torch.mm(faceNormals)
		l = ((n**2).sum(dim = 1)).sqrt()
		return  n/l[:,None]
	
	def edgeOnSilhouette(self, vertices, faceNormals, viewpoint):
		return super().edgeOnSilhouette( vertices.detach().numpy(), faceNormals.detach().numpy(), viewpoint)	

class TriMeshPytorch(TriMesh):
	def __init__(self, faces):
		super().__init__( faces)
	def computeAdjacencies(self):
		self.adjacencies = TriMeshAdjacenciesPytorch(self.faces)	