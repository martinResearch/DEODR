from scipy import sparse
import numpy as np

from .tools import normalize,normalize_backward,cross_backward
import inspect

class TriMeshAdjacencies:
    """this class stores adjacency matrices and methods that use this adjacencies. Unlike the TriMesh class there are no vertices stored in this class"""

    def __init__(self, faces,clockwise=False):
        self.faces = faces
        self.nbF = faces.shape[0]
        self.nbV = np.max(faces.flat) + 1
        i = self.faces.flatten()
        j = np.tile(np.arange(self.nbF)[:, None], [1, 3]).flatten()
        v = np.ones((self.nbF, 3)).flatten()
        self.Vertices_Faces = sparse.coo_matrix((v, (i, j)), shape=(self.nbV, self.nbF))
        idF = np.hstack((np.arange(self.nbF), np.arange(self.nbF), np.arange(self.nbF)))
        self.clockwise=clockwise
        edges = np.vstack((
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]])
        )   
        
        idEtmp,edgeIncrease=self.idEdge(edges)         

        _,idE,unique_counts  = np.unique(idEtmp, return_inverse=True,return_counts =True)        
        
        self.nbE = np.max(idE) + 1
        
        nbInc = np.zeros((self.nbE))
        np.add.at(nbInc,idE,edgeIncrease)
        nbDec = np.zeros((self.nbE))
        np.add.at(nbDec,idE,~edgeIncrease)        
        self.isManifold = np.all(unique_counts<=2) and np.all(nbInc<=1) and np.all(nbDec<=1)
        self.isClosed = self.isManifold  and np.all(unique_counts==2) 
        
        self.Edges_Faces_Ones = sparse.coo_matrix(
            (np.ones((len(idE))), (idE, idF)), shape=(self.nbE, self.nbF)
        )
        v = np.hstack(
            (np.full((self.nbF), 0), np.full((self.nbF), 1), np.full((self.nbF), 2))
        )
        self.Faces_Edges = sparse.coo_matrix(
            (idE, (idF, v)), shape=(self.nbF, 3)
        ).todense()
        self.Adjacency_Vertices = (
            (self.Vertices_Faces * self.Vertices_Faces.T) > 0
        ) - sparse.eye(self.nbV)
        self.DegreeVE = self.Adjacency_Vertices.dot(
            np.ones((self.nbV))
        )  # DegreeVE(i)=j means that the vertex i appears in j edges
        self.Laplacian = (
            sparse.diags([self.DegreeVE], [0], (self.nbV, self.nbV))
            - self.Adjacency_Vertices
        )
        self.hasBoundaries = np.any(np.sum(self.Edges_Faces_Ones, axis=1) == 1)
        assert np.all(self.Laplacian * np.ones((self.nbV)) == 0)
        self.store_backward = {}

    def idEdge(self, idv):
        
        return (
            np.maximum(idv[:, 0], idv[:, 1])
            + np.minimum(idv[:, 0], idv[:, 1]) * self.nbV,
            idv[:, 0]< idv[:, 1]
        )

    def computeFaceNormals(self, vertices):
        tris = vertices[self.faces, :]
        u = tris[:, 1, :] - tris[:, 0, :]
        v = tris[:, 2, :] - tris[:, 0, :]
        if self.clockwise:
            n = -np.cross(u, v)
        else:
            n = np.cross(u, v)
        normals = normalize(n, axis=1)
        self.store_backward["computeFaceNormals"] = (u, v, n)
        return normals

    def computeFaceNormals_backward(self, normals_b):
        u, v, n = self.store_backward["computeFaceNormals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        if self.clockwise:
            u_b, v_b = cross_backward(u, v, -n_b)
        else:
            u_b, v_b = cross_backward(u, v, n_b)
        tris_b = np.stack((-u_b - v_b, u_b, v_b), axis=1)
        vertices_b = np.zeros((self.nbV, 3))
        np.add.at(vertices_b, self.faces, tris_b)
        return vertices_b

    def computeVertexNormals(self, faceNormals):
        n = self.Vertices_Faces * faceNormals
        normals = normalize(n, axis=1)
        self.store_backward["computeVertexNormals"] = n
        return normals

    def computeVertexNormals_backward(self, normals_b):
        n = self.store_backward["computeVertexNormals"]
        n_b = normalize_backward(n, normals_b, axis=1)
        faceNormals_b = self.Vertices_Faces.T * n_b
        return faceNormals_b

    def edgeOnSilhouette(self, vertices2D):
        """this computes the a boolean for each of edges of each face that is true if and only if the edge is one the silhouette of the mesh given a view point"""
        tris = vertices2D[self.faces, :]
        u = tris[:, 1, :] - tris[:, 0, :]
        v = tris[:, 2, :] - tris[:, 0, :] 
        if self.clockwise:
            face_visible = np.cross(u,v)>0
        else:
            face_visible = np.cross(u,v)<0
        edge_bool = (self.Edges_Faces_Ones * face_visible) == 1
        return edge_bool[self.Faces_Edges]


class TriMesh:
    def __init__(self, faces,vertices=None,clockwise=False):

        self.faces = faces
        self.nbV = np.max(faces) + 1
        self.nbF = faces.shape[0]
        
        self.vertices = None
        self.faceNormals = None
        self.vertexNormals = None
        self.clockwise = clockwise
        self.computeAdjacencies()
        assert(self.adjacencies.isManifold)
        
        if not vertices is None:
            self.setVertices(vertices)
            self.checkOrientation()

    def computeAdjacencies(self):
        self.adjacencies = TriMeshAdjacencies(self.faces,self.clockwise)        

    def setVertices(self, vertices):
        self.vertices = vertices
        self.faceNormals = None
        self.vertexNormals = None
        
    def computeVolume(self):
        """Compute the volume enclosed by the triangulated surface. It assumes the surfaces is a closed manifold. 
        This is done by summing the volumes of the simplices formed by joining the origin and the vertices of each triangle"""
        return (1 if self.clockwise else -1)*np.sum(np.linalg.det(np.dstack((self.vertices[self.faces[:,0]],self.vertices[self.faces[:,1]],self.vertices[self.faces[:,2]]))))/6        
        
    def checkOrientation(self):
        """check the mesh faces are properly oriented for the normals to point outward"""
        if (self.computeVolume()>0):
            raise(BaseException('The volume within the surface is negative. It seems that you faces are not oriented cooreclt accourding to the clockwise flag'))        



    def computeFaceNormals(self):
        self.faceNormals = self.adjacencies.computeFaceNormals(self.vertices)

    def computeVertexNormals(self):
        if self.faceNormals is None:
            self.computeFaceNormals()
        self.vertexNormals = self.adjacencies.computeVertexNormals(self.faceNormals)

    def computeVertexNormals_backward(self, vertexNormals_b):
        self.faceNormals_b = self.adjacencies.computeVertexNormals_backward(
            vertexNormals_b
        )
        self.vertices_b += self.adjacencies.computeFaceNormals_backward(
            self.faceNormals_b
        )

    def edgeOnSilhouette(self, points2D):
        """this computes the a boolean for each of edges that is true if and only if the edge is one the silhouette of the mesh"""
       
        return self.adjacencies.edgeOnSilhouette(points2D)
    
    
class ColoredTriMesh(TriMesh):
    def __init__(self, faces, vertices=None, clockwise=False,faces_uv=None,uv=None,texture=None,colors=None,nbColors=None):
        super(ColoredTriMesh, self).__init__(faces,vertices=vertices,clockwise=clockwise)
        self.faces_uv = faces_uv
        self.uv = uv
        
        self.texture = texture    
        self.colors = colors
        self.textured = not (self.texture is None)
        self.nbColors=nbColors
        if nbColors is None: 
            if texture is None:
                self.nbColors=colors.shape[1]
            else:
                self.nbColors= texture.shape[2]
                
    def setVerticesColors(self, colors):
        self.verticesColors = colors                
            
    def plot_uv_map(self,ax):
        ax.imshow(self.texture)
        ax.triplot(self.uv[:, 0], self.uv[:, 1], self.faces_uv)    
        
    def plot(self,ax,plot_normals=False):
        x,y,z = self.vertices.T
        u,v,w = self.vertexNormals.T
        ax.plot_trisurf(self.vertices[:,0], self.vertices[:,1], Z= self.vertices[:,2], triangles=self.faces)
        ax.quiver(x, y, z, u, v, w, length=0.03, normalize=True,color=[0,1,0])        
    @staticmethod
    def from_trimesh(mesh):# inpired from pyrender
        import trimesh # get trimesh module here instead of importing at the top of the file to keep the dependency on trimesh optional
        """Gets the vertex colors, texture coordinates, and material properties
        from a :class:`~trimesh.base.Trimesh`.
        """
        colors = None
        uv = None
        texture = None

        # If the trimesh visual is undefined, return none for both

        # Process vertex colors
        if mesh.visual.kind == 'vertex':
            colors = mesh.visual.vertex_colors.copy()
            
        # Process face colors
        elif mesh.visual.kind == 'face':
            raise BaseException("not suported yet, will need antialisaing at the seams")
          
        # Process texture colors
        elif mesh.visual.kind == 'texture':
            # Configure UV coordinates
            if mesh.visual.uv is not None:
                
                texture = np.array(mesh.visual.material.image)/255
                if texture.shape[2]==4:
                    texture=texture[:,:,:3]# removing alpha channel
                
                uv=np.column_stack(((mesh.visual.uv[:,0])*texture.shape[0],(1-mesh.visual.uv[:,1])*texture.shape[1]))
                
                
        #merge identical 3D vertices even if their uv are different to keep surface manifold 
        # trimesh seem to split vertices that have different uvs (using unmerge_faces texture.py), making the surface not watertight, while there were only seems in the texture        
       
        vertices,return_index,inv_ids = np.unique( mesh.vertices,axis=0,return_index=True,return_inverse=True)
        faces = inv_ids[mesh.faces].astype(np.uint32)  
        if colors:
            colors2=colors[return_index,:]
            if np.any(colors != colors2[inv_ids,:]):
                raise(BaseException("vertices at the same 3D location should have the same color for the rendering to be differentiable"))
        else:
            colors2=None
            
        return ColoredTriMesh(faces,vertices, clockwise=False, faces_uv=mesh.faces, 
                      uv=uv, texture=texture, 
                      colors=colors2)
         
        