from  scipy import sparse
import numpy as np
import torch

class Mesh():
	def __init__(self, vertices, faces, computeAdjacencies = True):
		self.vertices = torch.tensor(vertices)
		self.faces = np.array(faces)
		self.faces_torch = torch.LongTensor(faces)
		self.faceNormals = None
		self.vertexNormals = None
		self.nbV = vertices.shape[0]
		self.nbF = faces.shape[0]		
		if computeAdjacencies:
			self.computeAdjacencies()
			
	def setVertices(self, vertices):
		self.vertices = vertices
		
	def setVerticesColors(self, colors):
		self.verticesColors = colors

	def computeFaceNormals(self):
		if isinstance(self.vertices, torch.Tensor):	
			tris = self.vertices[self.faces_torch,:]
			n = torch.cross( tris[::,1 ] - tris[::,0], tris[::,2 ] - tris[::,0] )
		else:
			tris = self.vertices[self.faces,:]
			n = np.cross( tris[::,1 ] - tris[::,0] , tris[::,2 ] - tris[::,0] )
		l = ((n**2).sum(dim = 1)).sqrt()
		self.faceNormals = n/l[:,None]

	def computeVertexNormals(self):
		self.computeFaceNormals()
		if isinstance(self.faceNormals, torch.Tensor):
			n = self.Vertices_Faces_torch.mm(self.faceNormals)
		else:
			n = self.Vertices_Faces*self.faceNormals
		l = ((n**2).sum(dim = 1)).sqrt()
		self.vertexNormals =  n/l[:,None]
		
	def idEdge(self, idv1, idv2):
		return np.maximum(idv1,idv2) + np.minimum(idv1,idv2) * self.nbV
	
	def invIdEdge(self,idE):
		idv1 = np.mod(idE, self.nbV)
		idv2 = (idE - idv1)/self.nbV
		return idv1,idv2
		
	def computeAdjacencies(self):		
		i = self.faces_torch.flatten()
		j = torch.LongTensor(np.tile(np.arange(self.nbF)[:,None],[1,3]).flatten())
		v = np.ones((self.nbF,3)).flatten()
		self.Vertices_Faces = sparse.coo_matrix((v,(i.numpy(),j.numpy())), shape=(self.nbV,self.nbF))
		self.Vertices_Faces_torch = torch.sparse.DoubleTensor(torch.stack((i,j)),torch.ones((self.nbF,3),dtype=torch.float64).flatten(), torch.Size((self.nbV,self.nbF)))		
		idE = np.hstack((self.idEdge(self.faces_torch[:,0],self.faces_torch[:,1]),self.idEdge(self.faces_torch[:,1],self.faces_torch[:,2]),self.idEdge(self.faces_torch[:,2],self.faces_torch[:,0])))
		idF = np.hstack((np.arange(self.nbF),np.arange(self.nbF),np.arange(self.nbF)))
		v = np.hstack((np.full((self.nbF),1),np.full((self.nbF),2),np.full((self.nbF),3)))		
		argsort_idE = np.argsort(idE)
		idE = idE[argsort_idE]
		idF = idF[argsort_idE]
		v = v[argsort_idE]
		idE = np.cumsum(np.hstack((0,np.diff(idE)))>0)
		self.nbE = idE[-1]+1		
		self.Edges_Faces = sparse.coo_matrix((v,(idE,idF)), shape=(self.nbE,self.nbF))
		self.Edges_Faces_Ones = sparse.coo_matrix((np.ones((len(idE))),(idE,idF)), shape=(self.nbE,self.nbF))
		self.Faces_Edges = sparse.coo_matrix((idE,(idF,v-1)),shape=(self.nbF,3)).todense()
		self.Adjacency_Vertices = ((self.Vertices_Faces*self.Vertices_Faces.T)>0)-sparse.eye(self.nbV) 
		self.DegreeVE = self.Adjacency_Vertices.dot(np.ones((self.nbV))) #DegreeVE(i)=j means that the vertex i appears in j edges
		self.Laplacian = sparse.diags([self.DegreeVE],[0],(self.nbV,self.nbV))-self.Adjacency_Vertices
		self.hasBoundaries = np.any(np.sum(self.Edges_Faces_Ones,axis=1)==1)		
		assert np.all(self.Laplacian*np.ones((self.nbV))==0)
		return self.Laplacian		

def SilhouetteEdges(mesh,viewpoint):
	"""this computes the a boolean for each of edges that is true if and only if the edge is one the silhouette of the mesh given a view point"""
	face_visible = (torch.sum(mesh.faceNormals *(mesh.vertices[self.faces_torch[:,0],:] -torch.tensor(viewpoint)[None,:]),dim=1)>0).numpy()			
	return ((mesh.Edges_Faces_Ones * face_visible)==1)		
		
		
	
		