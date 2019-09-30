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



