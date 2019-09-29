from DEODR import differentiable_renderer_cython
from DEODR.differentiable_renderer import Scene2D
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

def create_example_scene():
    
    Ntri = 30;
    SizeW = 200;
    SizeH = 200;
    material = np.double(imread('trefle.jpg'))/255
    Hmaterial = material.shape[0]
    Wmaterial = material.shape[1]
    
    scale_matrix = np.array([[SizeH,0],[0,SizeW]])
    scale_material = np.array([[Hmaterial-1,0],[0,Wmaterial-1]])
    
    triangles=[]
    for k in range(Ntri):
        
        tmp=scale_matrix.dot(np.random.rand(2,1).dot(np.ones((1,3)))+0.5*(-0.5+np.random.rand(2,3)))
        while np.abs(np.linalg.det(np.vstack((tmp,np.ones((3))))))<1500:
            tmp=scale_matrix.dot(np.random.rand(2,1).dot(np.ones((1,3)))+0.5*(-0.5+np.random.rand(2,3)))
        
        if np.linalg.det(np.vstack((tmp,np.ones((3)))))<0:
            tmp=np.fliplr(tmp) 
        triangle={}
        triangle['ij'] = tmp.T
        triangle['depths'] = np.random.rand(1)*np.ones((3)) # constant depth triangles to avoid collisions
        triangle['textured'] = np.random.rand(1)>0.5
        
        if triangle['textured']:
            triangle['uv'] = scale_material.dot(np.array([[0,1,0.2],[0,0.2,1]])).T+1# texture coordinate of the vertices
            triangle['shade'] = np.random.rand(3) # shade  intensity at each vertex
            triangle['colors'] = np.zeros((3,3))
            triangle['shaded'] = True
        else:
            triangle['uv'] = np.zeros((3,2))
            triangle['shade'] = np.zeros((3)) 
            triangle['colors'] = np.random.rand(3,3) # colors of the vertices (can be gray, rgb color,or even other dimension vectors) when using simple linear interpolation across triangles
            triangle['shaded'] = False
                
        triangle['edgeflags'] = np.array([True,True,True]) # all edges are discontinuity edges as no triangle pair share an edge
        triangles.append(triangle)    
           
    scene={}
    for key in triangles[0].keys(): 
        scene[key] = np.stack([triangle[key] for triangle in triangles]) 
    scene['image_H'] = SizeH
    scene['image_W'] = SizeW
    scene['texture'] = material     
    scene['nbColors'] = 3
    scene['background'] = np.tile(np.array([0.3,0.5,0.7])[None,None,:],(SizeH,SizeW,1))        
    return Scene2D(**scene)   
    
def main():
    display = True
    np.random.seed(2)
    scene1 = create_example_scene()
    display = True
    save_images = True
    antialiaseError = False
    sigma=1
    
    AbufferTarget = np.zeros((scene1.image_H,scene1.image_W,scene1.nbColors))
    
    Zbuffer = np.zeros((scene1.image_H,scene1.image_W))    
    differentiable_renderer_cython.renderScene(scene1,sigma,AbufferTarget,Zbuffer)

    Ntri = len(scene1.depths);
    scene2 = copy.deepcopy(scene1)
    scale_material = np.array([[scene1.texture.shape[0]-1,0],[0,scene1.texture.shape[1]-1]])

    displacement_magnitude_ij = 10
    displacement_magnitude_uv = 0
    displacement_magnitude_colors = 0    
   
    max_uv=np.array(scene1.texture.shape[:2])-1    

    scene2.ij = scene1.ij+np.random.randn(Ntri,3,2)*displacement_magnitude_ij
    scene2.uv = scene1.uv+np.random.randn(Ntri,3,2)*displacement_magnitude_uv
    scene2.uv = np.maximum( scene2.uv,0)
    scene2.uv = np.minimum( scene2.uv,max_uv)
    scene2.colors = scene1.colors+np.random.randn(Ntri,3,3)*displacement_magnitude_colors    

    alpha_ij = 0.01
    beta_ij = 0.80
    alpha_uv = 0.03
    beta_uv = 0.80
    alpha_color = 0.001
    beta_color = 0.70

    speed_ij = np.zeros((Ntri,3,2))
    speed_uv = np.zeros((Ntri,3,2))
    speed_color = np.zeros((Ntri,3,3))

    nbMaxIter = 500
    losses=[]
    for iter in range(nbMaxIter):
        Image,depth,lossImage,loss = scene2.render_and_compare(sigma,AbufferTarget,antialiaseError);
        
        #imsave(os.path.join(iterfolder,f'soup_{iter}.png'), combinedIMage)
        key = cv2.waitKey(1) 
        losses.append(loss)
        if lossImage.ndim==2:
            lossImage=np.broadcast_to(lossImage[:,:,None],Image.shape)
        cv2.imshow('animation',np.column_stack((AbufferTarget,Image,lossImage))[:,:,::-1]) 
        
        if displacement_magnitude_ij>0:
            speed_ij = beta_ij*speed_ij-scene2.ij_b*alpha_ij
            scene2.ij = scene2.ij+speed_ij
        
        if displacement_magnitude_colors>0:
            speed_color = beta_color*speed_color-scene2b.colors_b*alpha_color
            scene2.colors = scene2.colors+speed_color
    
        if displacement_magnitude_uv>0:
            speed_uv = beta_uv*speed_uv-scene2b.uv_b*alpha_uv
            scene2.uv = scene2.uv+speed_uv
            scene2.uv = max( scene2.uv,0)
            scene2.uv = min( scene2.uv,max_uv)
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    main()
