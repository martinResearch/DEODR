import numpy as np
from . import differentiable_renderer_cython
import copy

class Scene2D():
    """this class is a simple class that contiains a triangles soup with additional field to store the gradients"""
    def __init__(self, ij, depths, textured, uv, shade, colors, shaded, edgeflags, image_H, image_W, nbColors, texture, background):        
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
        self.texture =  texture
        self.background = background     
        
    def render(self, sigma = 1, antialiaseError = False, Aobs = None):
        if antialiaseError:
            Abuffer = np.zeros((self.image_H, self.image_W, self.nbColors))
            ErrBuffer = np.sum((Aobs - self.background)**2, axis=2)
            Zbuffer = np.zeros((self.image_H, self.image_W))                 
            differentiable_renderer_cython.renderScene(self, sigma, Abuffer, Zbuffer, antialiaseError, Aobs, ErrBuffer) 
            return Abuffer, Zbuffer, ErrBuffer
        else:
            Abuffer = np.zeros((self.image_H, self.image_W, self.nbColors))
            Zbuffer = np.zeros((self.image_H, self.image_W))     
            ErrBuffer = None
            differentiable_renderer_cython.renderScene(self, sigma, Abuffer, Zbuffer, antialiaseError, Aobs, ErrBuffer)             
            return Abuffer, Zbuffer, ErrBuffer

class Scene2DWithBackward(Scene2D):
    def __init__(self, ij, depths, textured, uv, shade, colors, shaded, edgeflags, image_H, image_W, nbColors, texture, background):
        super().__init__(ij, depths, textured, uv, shade, colors, shaded, edgeflags, image_H, image_W, nbColors, texture, background)    
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
    
    def render_compare_and_backward(self, Aobs, sigma = 1, antialiaseError = False, mask = None, clear_gradients = True, make_copies=True):        
        if mask is None:
            mask = np.ones((Aobs.shape[0],Aobs.shape[1]))            
        Abuffer, Zbuffer, ErrBuffer = self.render(sigma, antialiaseError, Aobs)
        if clear_gradients:
            self.clear_gradients()        
        if antialiaseError:
            ErrBuffer = ErrBuffer * mask    
            Err = np.sum(ErrBuffer)      
            ErrBuffer_B = copy.copy(mask)  
            if make_copies:
                differentiable_renderer_cython.renderSceneB(self, sigma, Abuffer, Zbuffer, None, antialiaseError, Aobs, ErrBuffer.copy(), ErrBuffer_B)
            else: # if we don't make copies the image Abuffer will 
                differentiable_renderer_cython.renderSceneB(self, sigma, Abuffer, Zbuffer, None, antialiaseError, Aobs, ErrBuffer, ErrBuffer_B)
            return Abuffer, Zbuffer, ErrBuffer, Err            
        else:              
            diffImage = (Abuffer - Aobs) * mask[:,:,None]  
            Err = np.sum(diffImage**2)    
            Abuffer_b = 2 * diffImage          
            ErrBuffer_B = None
            if make_copies: # if we make copies we keep the antialized image unchanged Abuffer along the occlusion boundaries
                differentiable_renderer_cython.renderSceneB(self, sigma, Abuffer.copy(), Zbuffer, Abuffer_b, antialiaseError, Aobs, ErrBuffer, ErrBuffer_B)
            else:
                differentiable_renderer_cython.renderSceneB(self, sigma, Abuffer, Zbuffer, Abuffer_b, antialiaseError, Aobs, ErrBuffer, ErrBuffer_B)
            return Abuffer, Zbuffer, diffImage, Err    

class Scene3D():
    def __init__(self):
        self.mesh = None
        self.ligthDirectional = None
        self.ambiantLight = None
        pass
    
    def setLight(self, ligthDirectional, ambiantLight):  
        self.ligthDirectional = ligthDirectional
        self.ambiantLight = ambiantLight
        
    def setMesh(self,mesh):
        self.mesh = mesh
        
    def setBackground(self,backgroundImage):
        self.background = backgroundImage
    
    def camera_project(self,cameraMatrix,P3D, get_jacobians = False) : 
        r = np.column_stack((P3D, np.ones((P3D.shape[0],1), dtype = np.double))).dot(cameraMatrix.T)
        depths = r[:,2]
        P2D = r[:,:2]/depths[:,None]
        if get_jacobians:
            J_r = cameraMatrix[:,:3]
            J_d = J_r[2,:]
            J_PD2 = (J_r[:2,:]*depths[:,None,None]-r[:,:2,None]*J_d)/((depths**2)[:,None,None])
            return P2D,depths,J_PD2
        else:           
            return P2D,depths        
           
    def render(self,CameraMatrix,resolution):
        self.mesh.computeVertexNormals()
        
        P2D,depths = self.camera_project(CameraMatrix, self.mesh.vertices)        
        cameraCenter3D = -np.linalg.solve(CameraMatrix[:3,:3], CameraMatrix[:,3]) 
        colorsV =  self.computeVerticesColors()
        
        #compute silhouette edges 
        edge_bool = self.mesh.edgeOnSilhouette(cameraCenter3D)
        
        # construct triangle soup        
        ij = self.gather_faces(P2D)
        colors = self.gather_faces(colorsV)
        self.depths = self.gather_faces(depths)       
        self.edgeflags = edge_bool
        self.uv = np.zeros((self.mesh.nbF,3,2))
        self.textured = np.zeros((self.mesh.nbF),dtype=np.bool)
        self.shade = np.zeros((self.mesh.nbF,3),dtype=np.bool) # eventually used when using texture
        self.image_H = resolution[1]
        self.image_W = resolution[0]
        self.shaded = np.zeros((self.mesh.nbF),dtype=np.bool) # eventually used when using texture
        self.texture = np.zeros((0,0))          
        Abuffer = self.render2D(ij,colors)
        return Abuffer    
    
    def gather_faces(self,X):
        return X[self.mesh.faces]
    
    def renderDepth(self,CameraMatrix,resolution,depth_scale):        
        P2D, depths = self.camera_project(CameraMatrix, self.mesh.vertices)        
        cameraCenter3D = -np.linalg.solve(CameraMatrix[:3,:3], CameraMatrix[:,3])        
    
        #compute silhouette edges 
        self.mesh.computeFaceNormals()
        edge_bool = self.mesh.edgeOnSilhouette(cameraCenter3D)
    
        # construct triangle soup        
        ij = self.gather_faces(P2D)
        colors = self.gather_faces(depths)[:,:,None]*depth_scale
        self.depths = self.gather_faces(depths)
        self.edgeflags = edge_bool
        self.uv = np.zeros((self.mesh.nbF,3,2))
        self.textured = np.zeros((self.mesh.nbF), dtype = np.bool)
        self.shade = np.zeros((self.mesh.nbF,3), dtype = np.bool) # eventually used when using texture
        self.image_H = resolution[1]
        self.image_W = resolution[0]
        self.shaded = np.zeros((self.mesh.nbF),dtype=np.bool) # eventually used when using texture
        self.texture = np.zeros((0,0))
        #colors[:,:,:]=1
        Abuffer = self.render2D(ij,colors)
        return Abuffer
    
    def projectionsJacobian(self,CameraMatrix, vertices):
        P2D,depths,J_P2D = self.camera_project(CameraMatrix, vertices, get_jacobians=True) 
        return J_P2D        

class Scene3DWithBackward(Scene3D):
    def __init__(self):
        super().__init__()        
    def clear_gradients():
        pass
    def render_compare_and_backward():
        pass
        
