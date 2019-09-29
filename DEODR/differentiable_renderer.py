import numpy as np
from . import differentiable_renderer_cython
import torch
import copy

class Scene2D():
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
        # fields to store gradients  
        self.uv_b = np.zeros(self.uv.shape)
        self.ij_b = np.zeros(self.ij.shape)
        self.shade_b = np.zeros(self.shade.shape)
        self.colors_b = np.zeros(self.colors.shape)         

    def clear_gradients(self):
        self.uv_b.fill(0)
        self.ij_b.fill(0)
        self.shade_b.fill(0)
        self.colors_b.fill(0)        
        
    def render_and_compare(self, sigma, Aobs, antialiaseError = False, mask = None, clear_gradients = True):
    
        if mask is None:
            mask = np.ones((Aobs.shape[0],Aobs.shape[1]))
                
        if antialiaseError:
            Abuffer = np.zeros((self.image_H, self.image_W, self.nbColors))
            ErrBuffer = np.sum((Aobs - self.background)**2, axis=2)
            Zbuffer = np.zeros((self.image_H, self.image_W))                 
            differentiablerenderer_cython.renderScene(self, sigma, Abuffer, Zbuffer, antialiaseError, Aobs, ErrBuffer)    
            ErrBuffer = ErrBuffer * mask    
            Err = np.sum(ErrBuffer)      
            ErrBuffer_B = copy.copy(mask)  
            if clear_gradients:
                self.clear_gradients()
            differentiablerenderer_cython.renderSceneB(self, sigma, Abuffer, Zbuffer, None, antialiaseError, Aobs, ErrBuffer, ErrBuffer_B)
            return Abuffer, Zbuffer, ErrBuffer, Err
            
        else:
            Abuffer = np.zeros((self.image_H, self.image_W, self.nbColors))
            Zbuffer = np.zeros((self.image_H, self.image_W))     
            ErrBuffer = None
            differentiable_renderer_cython.renderScene(self, sigma, Abuffer, Zbuffer, antialiaseError, Aobs, ErrBuffer)               
            diffImage = (Abuffer - Aobs) * mask[:,:,None]  
            Err = np.sum(diffImage**2)    
            Abuffer_b = 2*diffImage
            if clear_gradients:
                self.clear_gradients()            
            ErrBuffer_B = None
            differentiable_renderer_cython.renderSceneB(self, sigma, Abuffer, Zbuffer, Abuffer_b, antialiaseError, Aobs, ErrBuffer, ErrBuffer_B)
            return Abuffer, Zbuffer, diffImage, Err


class TorchDifferentiableRenderer2DFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ij, colors, scene):
        nbColorChanels = colors.shape[2]
        Abuffer = np.zeros((scene.image_H, scene.image_W, nbColorChanels))
        Zbuffer = np.zeros((scene.image_H, scene.image_W))
        ctx.scene = scene
        scene.ij = ij.detach()#should automatically detached according to https://pytorch.org/docs/master/notes/extending.html
        scene.colors = colors.detach() 
        differentiable_renderer_cython.renderScene(scene, 1, Abuffer, Zbuffer)
        ctx.save_for_backward(ij, colors)
        ctx.Abuffer = Abuffer
        ctx.Zbuffer = Zbuffer
        return torch.tensor(Abuffer)
    
    @staticmethod
    def backward(ctx, Abuffer_b):
        scene = ctx.scene   
        scene.uv_b = np.zeros(scene.uv.shape)
        scene.ij_b = np.zeros(scene.ij.shape)
        scene.shade_b = np.zeros(scene.shade.shape)
        scene.colors_b = np.zeros(scene.colors.shape)         
        differentiable_renderer_cython.renderSceneB(scene, 1, ctx.Abuffer, ctx.Zbuffer, Abuffer_b.numpy())         
        return torch.tensor(scene.ij_b), torch.tensor(scene.colors_b),None
       
TorchDifferentiableRenderer2D = TorchDifferentiableRenderer2DFunc.apply

def camera_project(cameraMatrix,P3D, get_jacobians = False) : 
    if isinstance(P3D,torch.Tensor): 
        r = torch.cat((P3D, torch.ones((P3D.shape[0], 1), dtype = torch.double)), dim = 1).mm(torch.tensor(cameraMatrix.T))
    else:
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
    
class Scene3D():
    def __init__(self):
        self.mesh = None
        self.ligthDirectional = None
        self.ambiantLight = None
        pass
    
    def setLight(self, ligthDirectional, ambiantLight):
        if not(isinstance(ligthDirectional, torch.Tensor)):
            ligthDirectional = torch.tensor(ligthDirectional)        
        self.ligthDirectional = ligthDirectional
        self.ambiantLight = ambiantLight
        
    def setMesh(self,mesh):
        self.mesh = mesh
        
    def setBackground(self,backgroundImage):
        self.background = backgroundImage
     
    def render(self,CameraMatrix,resolution):
        self.mesh.computeVertexNormals()
        
        P2D,depths = camera_project(CameraMatrix, self.mesh.vertices)        
        cameraCenter3D = -np.linalg.solve(CameraMatrix[:3,:3], CameraMatrix[:,3])        
          
        verticesLuminosity = torch.relu(-torch.sum(self.mesh.vertexNormals * self.ligthDirectional, dim = 1)) + self.ambiantLight
        colorsV = self.mesh.verticesColors * verticesLuminosity[:,None]
        
        #compute silhouette edges 
        silhouetteEdges=SilhouetteEddge(mesh,cameraCenter3D)        
        
        # construct triangle soup        
        ij = P2D[self.mesh.faces]
        colors = colorsV[self.mesh.faces]
        self.depths = depths[self.mesh.faces].detach()
        self.edgeflags = silhouetteEdges.edge_bool[self.mesh.Faces_Edges]
        self.uv = np.zeros((self.mesh.nbF,3,2))
        self.textured = np.zeros((self.mesh.nbF),dtype=np.bool)
        self.shade = np.zeros((self.mesh.nbF,3),dtype=np.bool) # eventually used when using texture
        self.image_H = resolution[1]
        self.image_W = resolution[0]
        self.shaded = np.zeros((self.mesh.nbF),dtype=np.bool) # eventually used when using texture
        self.texture = np.zeros((0,0))
     
        Abuffer = TorchDifferentiableRenderer2D(ij,colors,self)
        return Abuffer
    
    def renderDepth(self,CameraMatrix,resolution,depth_scale):
        
        P2D,depths = camera_project(CameraMatrix, self.mesh.vertices)        
        cameraCenter3D = -np.linalg.solve(CameraMatrix[:3,:3], CameraMatrix[:,3])        
    
        #compute silhouette edges 
        self.mesh.computeVertexNormals()
        self.mesh.computSilhouetteEddges(cameraCenter3D)  
    
        # construct triangle soup        
        ij = P2D[self.mesh.faces]
        colors = depths[self.mesh.faces][:,:,None]*depth_scale
        self.depths = depths[self.mesh.faces].detach()
        self.edgeflags = self.mesh.edge_bool[self.mesh.Faces_Edges]
        self.uv = np.zeros((self.mesh.nbF,3,2))
        self.textured = np.zeros((self.mesh.nbF), dtype = np.bool)
        self.shade = np.zeros((self.mesh.nbF,3), dtype = np.bool) # eventually used when using texture
        self.image_H = resolution[1]
        self.image_W = resolution[0]
        self.shaded = np.zeros((self.mesh.nbF),dtype=np.bool) # eventually used when using texture
        self.texture = np.zeros((0,0))
        #colors[:,:,:]=1
        Abuffer = TorchDifferentiableRenderer2D(ij,colors,self)
        return Abuffer
    
    def projectionsJacobian(self,CameraMatrix):
        P2D,depths,J_P2D = camera_project(CameraMatrix, self.mesh.vertices.detach().numpy(), get_jacobians=True) 
        return J_P2D
    
        