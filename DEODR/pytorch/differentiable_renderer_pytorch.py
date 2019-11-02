import numpy as np
from .. import differentiable_renderer_cython
import torch
import copy
from ..differentiable_renderer import Scene3D

class TorchDifferentiableRenderer2DFunc(torch.autograd.Function):
    
    @staticmethod    
    def forward(ctx, ij, colors, scene):
        nbColorChanels = colors.shape[1]
        Abuffer = np.empty((scene.image_H, scene.image_W, nbColorChanels))
        Zbuffer = np.empty((scene.image_H, scene.image_W))
        ctx.scene = scene
        scene.ij = ij.detach().numpy() #should automatically detached according to https://pytorch.org/docs/master/notes/extending.html
        scene.colors = colors.detach().numpy() 
        differentiable_renderer_cython.renderScene(scene, 1, Abuffer, Zbuffer)
        ctx.save_for_backward(ij, colors)
        ctx.Abuffer = Abuffer.copy() # making a copy to keep the antializaed image for visualization , could be optional
        ctx.Zbuffer = Zbuffer  
        return torch.as_tensor(Abuffer)
    
    @staticmethod    
    def backward(ctx, Abuffer_b):
        scene = ctx.scene   
        scene.uv_b = np.zeros(scene.uv.shape)
        scene.ij_b = np.zeros(scene.ij.shape)
        scene.shade_b = np.zeros(scene.shade.shape)
        scene.colors_b = np.zeros(scene.colors.shape) 
        scene.texture_b = np.zeros(scene.texture.shape) 
        differentiable_renderer_cython.renderSceneB(scene, 1, ctx.Abuffer, ctx.Zbuffer, Abuffer_b.numpy())         
        return torch.as_tensor(scene.ij_b), torch.as_tensor(scene.colors_b),None
       
TorchDifferentiableRender2D = TorchDifferentiableRenderer2DFunc.apply

class Scene3DPytorch(Scene3D):
    def __init__(self):
        super().__init__()
    
    def setLight(self, ligthDirectional, ambiantLight):
        if not(isinstance(ligthDirectional, torch.Tensor)):
            ligthDirectional = torch.tensor(ligthDirectional)        
        self.ligthDirectional = ligthDirectional
        self.ambiantLight = ambiantLight    
            
    def cameraProject(self,cameraMatrix,P3D) : 
        assert( isinstance(P3D,torch.Tensor)) 
        r = torch.cat((P3D, torch.ones((P3D.shape[0], 1), dtype = torch.double)), dim = 1).mm(torch.tensor(cameraMatrix.T))
        depths = r[:,2]
        P2D = r[:,:2]/depths[:,None]
        return P2D,depths     
 
    def computeVerticesColorsWithIllumination(self):
        verticesLuminosity = torch.relu(-torch.sum(self.mesh.vertexNormals * self.ligthDirectional, dim = 1)) + self.ambiantLight
        return self.mesh.verticesColors * verticesLuminosity[:,None]      
    
    def render2D(self,ij,colors):   
        self.depths = self.depths.detach()
        return TorchDifferentiableRender2D(ij,colors,self)
    
