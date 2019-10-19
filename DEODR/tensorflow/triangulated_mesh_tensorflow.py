from  scipy import sparse
import numpy as np
import tensorflow as tf
from ..triangulated_mesh import *
from .tools import scipy_sparse_matrix_to_tensorflow


class TriMeshAdjacenciesTensorflow(TriMeshAdjacencies):
	def __init__(self,faces):
		super().__init__(faces)
		self.faces_tf = tf.constant(faces)				
		self.Vertices_Faces_tf = scipy_sparse_matrix_to_tensorflow(self.Vertices_Faces) 
		
	def computeFaceNormals(self,vertices):
		tris = tf.gather(vertices,self.faces)
		n = tf.linalg.cross( tris[::,1 ] - tris[::,0], tris[::,2 ] - tris[::,0] )
		l = tf.sqrt(tf.reduce_sum(n**2, axis = 1))
		return n/l[:,None]
	
	def computeVertexNormals(self,faceNormals):
		n = tf.sparse.sparse_dense_matmul(self.Vertices_Faces_tf,faceNormals)
		l = tf.sqrt(tf.reduce_sum(n**2, axis = 1))
		return  n/l[:,None]
	
	def edgeOnSilhouette(self, vertices, faceNormals, viewpoint):
		"""this computes the a boolean for each of edges of each face that is true if and only if the edge is one the silhouette of the mesh given a view point"""	
		face_visible = (tf.reduce_sum(faceNormals *(tf.gather(vertices,self.faces[:,0]) - tf.constant(viewpoint)[None,:]),axis = 1) > 0).numpy()			
		edge_bool =  ((self.Edges_Faces_Ones * face_visible)==1)
		return edge_bool[self.Faces_Edges]		

class TriMeshTensorflow(TriMesh):
	def __init__(self, faces):
		super().__init__( faces)
	def computeAdjacencies(self):
		self.adjacencies = TriMeshAdjacenciesTensorflow(self.faces)	