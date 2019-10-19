import numpy as np
from .. import differentiable_renderer_cython
import tensorflow as tf
import copy
from ..differentiable_renderer import Scene3D

#class TorchDifferentiableRenderer2DFunc(torch.autograd.Function):
    #@staticmethod
    #def forward(ctx, ij, colors, scene):
        #nbColorChanels = colors.shape[2]
        #Abuffer = np.zeros((scene.image_H, scene.image_W, nbColorChanels))
        #Zbuffer = np.zeros((scene.image_H, scene.image_W))
        #ctx.scene = scene
        #scene.ij = ij.detach()#should automatically detached according to https://pytorch.org/docs/master/notes/extending.html
        #scene.colors = colors.detach() 
        #differentiable_renderer_cython.renderScene(scene, 1, Abuffer, Zbuffer)
        #ctx.save_for_backward(ij, colors)
        #ctx.Abuffer = Abuffer # this make a copy, we could try to avoid that 
        #ctx.Zbuffer = Zbuffer # this make a copy, we could try to avoid that 
        #return torch.tensor(Abuffer)
    
    #@staticmethod
    #def backward(ctx, Abuffer_b):
        #scene = ctx.scene   
        #scene.uv_b = np.zeros(scene.uv.shape)
        #scene.ij_b = np.zeros(scene.ij.shape)
        #scene.shade_b = np.zeros(scene.shade.shape)
        #scene.colors_b = np.zeros(scene.colors.shape)         
        #differentiable_renderer_cython.renderSceneB(scene, 1, ctx.Abuffer, ctx.Zbuffer, Abuffer_b.numpy())         
        #return torch.tensor(scene.ij_b), torch.tensor(scene.colors_b),None
       
#TorchDifferentiableRender2D = TorchDifferentiableRenderer2DFunc.apply


def TensorflowDifferentiableRender2D( ij, colors, scene):
    @tf.custom_gradient
    def forward(ij, colors):#using inner function as we don't differentate w.r.t scene
        nbColorChanels = colors.shape[2]
        Abuffer = np.zeros((scene.image_H, scene.image_W, nbColorChanels))
        Zbuffer = np.zeros((scene.image_H, scene.image_W))
        #ctx.scene = scene
        scene.ij = np.array (ij)#should automatically detached according to https://pytorch.org/docs/master/notes/extending.html
        scene.colors = np.array (colors) 
        scene.depths = np.array (scene.depths)
        differentiable_renderer_cython.renderScene(scene, 1, Abuffer, Zbuffer)
        #ctx.save_for_backward(ij, colors)
        #ctx.Abuffer = Abuffer # this make a copy, we could try to avoid that 
        #ctx.Zbuffer = Zbuffer # this make a copy, we could try to avoid that 
        def backward(Abuffer_b):
            
            scene.uv_b = np.zeros(scene.uv.shape)
            scene.ij_b = np.zeros(scene.ij.shape)
            scene.shade_b = np.zeros(scene.shade.shape)
            scene.colors_b = np.zeros(scene.colors.shape)         
            differentiable_renderer_cython.renderSceneB(scene, 1, Abuffer, Zbuffer, Abuffer_b.numpy())         
            return tf.constant(scene.ij_b),tf.constant(scene.colors_b)    
    
        return tf.constant(Abuffer),backward
    return forward(ij, colors)

class Scene3DTensorflow(Scene3D):
    def __init__(self):
        super().__init__()
    
    def setLight(self, ligthDirectional, ambiantLight):
        if not(isinstance(ligthDirectional, torch.Tensor)):
            ligthDirectional = torch.tensor(ligthDirectional)        
        self.ligthDirectional = ligthDirectional
        self.ambiantLight = ambiantLight    
            
    def camera_project(self,cameraMatrix,P3D) : 
        assert( isinstance(P3D,tf.Tensor)) 
        r = tf.linalg.matmul( tf.concat((P3D,  tf.ones((P3D.shape[0], 1),dtype=P3D.dtype)), axis = 1),tf.constant(cameraMatrix.T))
        depths = r[:,2]
        P2D = r[:,:2]/depths[:,None]
        return P2D,depths    

    def gather_faces(self,X):
        return tf.gather(X,self.mesh.faces)    
 
    def computeVerticesColors(self):
        verticesLuminosity = tf.nn.relu(-tf.sum(self.mesh.vertexNormals * self.ligthDirectional, axis = 1)) + self.ambiantLight
        return self.mesh.verticesColors * verticesLuminosity[:,None]      
    
    def render2D(self,ij,colors):   
        
        return TensorflowDifferentiableRender2D(ij,colors,self)
    
