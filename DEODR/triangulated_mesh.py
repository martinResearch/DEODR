from  scipy import sparse
import numpy as np


class TriMeshAdjacencies():
	"""this class stores adjacency matrices and methods that use this adjacencies. Unlike the TriMesh class there are no vertices stored in this class"""
	def __init__(self,faces):
		self.faces = faces
		self.nbF = faces.shape[0]
		self.nbV = np.max(faces.flat)+1
		i = self.faces.flatten()
		j = np.tile(np.arange(self.nbF)[:,None],[1,3]).flatten()
		v = np.ones((self.nbF,3)).flatten()		
		self.Vertices_Faces = sparse.coo_matrix((v,(i,j)), shape=(self.nbV,self.nbF))
		idF = np.hstack((np.arange(self.nbF),np.arange(self.nbF),np.arange(self.nbF)))			
		idEtmp = np.hstack((self.idEdge(self.faces[:,[0,1]]), self.idEdge(self.faces[:,[1,2]]), self.idEdge(self.faces[:,[2,0]])))
		idE = np.unique(idEtmp,return_inverse = True)[1]
		self.nbE = np.max(idE)+1		
		self.Edges_Faces_Ones = sparse.coo_matrix((np.ones((len(idE))),(idE,idF)), shape=(self.nbE,self.nbF))
		v = np.hstack((np.full((self.nbF),0),np.full((self.nbF),1),np.full((self.nbF),2)))
		self.Faces_Edges = sparse.coo_matrix((idE,(idF,v)),shape=(self.nbF,3)).todense()
		self.Adjacency_Vertices = ((self.Vertices_Faces*self.Vertices_Faces.T)>0)-sparse.eye(self.nbV) 
		self.DegreeVE = self.Adjacency_Vertices.dot(np.ones((self.nbV))) #DegreeVE(i)=j means that the vertex i appears in j edges
		self.Laplacian = sparse.diags([self.DegreeVE],[0],(self.nbV,self.nbV))-self.Adjacency_Vertices
		self.hasBoundaries = np.any(np.sum(self.Edges_Faces_Ones,axis=1)==1)		
		assert np.all(self.Laplacian*np.ones((self.nbV))==0)
		
	def idEdge(self, idv):
		return np.maximum(idv[:,0],idv[:,1]) + np.minimum(idv[:,0],idv[:,1]) * self.nbV		
	
	def computeFaceNormals(self,vertices):
		tris = vertices[self.faces,:]
		n = np.cross( tris[::,1 ] - tris[::,0] , tris[::,2 ] - tris[::,0] )
		l = ((n**2).sum(dim = 1)).sqrt()
		normals = n/l[:,None]
		return normals
	
	def computeVertexNormals(self,faceNormals):
		n = self.Vertices_Faces * faceNormals
		l = ((n**2).sum(dim = 1)).sqrt()
		normals = n/l[:,None]
		return normals  
	
	def computeFaceNormals_backard(self, faceNormals, normals_grad):		
		faceNormals_grad = self.Vertices_Faces.T* n_grad
		# = self.Vertices_Faces * 
		return faceNormals_grad
	
	def edgeOnSilhouette(self, vertices, faceNormals, viewpoint):		
		"""this computes the a boolean for each of edges of each face that is true if and only if the edge is one the silhouette of the mesh given a view point"""	
		face_visible = np.sum(faceNormals *(vertices[self.faces[:,0],:] -viewpoint),axis=1)>0		
		edge_bool =  ((self.Edges_Faces_Ones * face_visible)==1)		
		return edge_bool[self.Faces_Edges]		
class TriMesh():
	def __init__(self, faces):
		
		self.faces = faces
		self.nbV = np.max(faces)+1
		self.nbF = faces.shape[0]
		self.vertices = None
		self.faceNormals = None
		self.vertexNormals = None
		self.computeAdjacencies()
		
	def computeAdjacencies(self):
		self.adjacencies = TriMeshAdjacencies(self.faces)

	def setVertices(self, vertices):
		self.vertices = vertices
		self.faceNormals = None
		self.vertexNormals = None
		
	def setVerticesColors(self, colors):
		self.verticesColors = colors		

	def computeFaceNormals(self):
		self.faceNormals = self.adjacencies.computeFaceNormals(self.vertices)

	def computeVertexNormals(self):
		if self.faceNormals is None:
			self.computeFaceNormals()		
		self.vertexNormals = self.adjacencies.computeVertexNormals(self.faceNormals)				

	def edgeOnSilhouette(self,viewpoint):
		"""this computes the a boolean for each of edges that is true if and only if the edge is one the silhouette of the mesh given a view point"""
		if self.faceNormals is None:
			self.computeFaceNormals()		
		return self.adjacencies.edgeOnSilhouette(self.vertices, self.faceNormals,viewpoint)	
			
