import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn import decomposition 
from scipy.spatial import Delaunay
import numpy as np
from DEODR.differentiable_renderer import Scene2DWithBackward
import cv2

faces=sklearn.datasets.fetch_olivetti_faces()

#plt.imshow(faces.images[0])

faces_pca = decomposition.PCA(n_components=150, whiten=True)
faces_pca.fit(faces.data)
plt.imshow(faces_pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.bone)

#coefs = faces_pca.transform(faces.data)
#print(faces)

#faces_pca.inverse_transform(coefs[[0]])


# create a regular grid and a triangulation  of the 2D image
N=5
points=np.column_stack([t.flatten() for t in np.meshgrid(np.arange(N+1)/N,np.arange(N+1)/N)])
tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices)
on_border= np.any((points==0)|(points==1),axis=1)

# deform the grid
max_displacement = 0.8
np.random.seed(0)
points_deformed_gt = points + (np.random.rand(*points.shape)-0.5)*max_displacement/N
points_deformed_gt[on_border]=points[on_border]
plt.triplot(points_deformed_gt[:,0], points_deformed_gt[:,1], tri.simplices)
#plt.show()

nbTriangles = tri.simplices.shape[0]
ij = points_deformed_gt[tri.simplices,:]*64-0.5
uv = points[tri.simplices,:]*64+0.5
textured = np.ones((nbTriangles), dtype=np.bool)
shaded = np.ones((nbTriangles), dtype=np.bool)
depths = np.ones((nbTriangles,3))
shade =  np.ones((nbTriangles,3))
colors = np.ones((nbTriangles,3,1))
edgeflags = np.zeros((nbTriangles,3),dtype=np.bool)
image_H = 64
image_W = 64
nbColors = 1
texture = faces.images[10][:,:,None]
background = np.zeros((image_H,image_W,1))

scene_gt=Scene2DWithBackward(ij, depths, textured, uv, shade, colors, shaded, 
                   edgeflags, image_H, image_W, nbColors, 
                   texture, background)

A_gt,_,_ = scene_gt.render(sigma = 1)
plt.subplot(3,1,1)
plt.imshow(np.squeeze(texture,axis=2))
plt.subplot(3,1,2)
plt.imshow(np.squeeze(A_gt,axis=2))
plt.subplot(3,1,3)
plt.imshow(np.squeeze(np.abs(A_gt-texture),axis=2))

#plt.show()
np.max(texture-A_gt)




scene = Scene2DWithBackward(ij, depths, textured, uv, shade, colors, shaded, 
                             edgeflags, image_H, image_W, nbColors, 
                   texture, background)
#cv2.namedWindow('animation',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('animation', 600,600)

rescale_factor = 10
def fun(points_deformed,pca_coefs):
    ij = points_deformed[tri.simplices,:] * 64-0.5
    #face=faces_pca.inverse_transform(coefs[:,None])
    face= (faces_pca.mean_+ pca_coefs.dot(faces_pca.components_)).reshape((64,64))
    scene.ij = ij
    scene.texture = face[:,:,None]
    print('render')
    Abuffer, Zbuffer, diffImage, Err   = scene.render_compare_and_backward(Aobs = A_gt,sigma = 1)
    print("np.max(np.abs(scene.ij_b))=%f"%np.max(np.abs(scene.ij_b)))
    print("E=%f"%Err)
    print('done')
    A_gt_zoomed = cv2.resize((A_gt.copy()*255).astype(np.uint8), None, fx = rescale_factor, fy = rescale_factor, interpolation=cv2.INTER_NEAREST)
    Abuffer_zoomed = cv2.resize((Abuffer.copy()*255).astype(np.uint8), None, fx = rescale_factor, fy = rescale_factor, interpolation=cv2.INTER_NEAREST)
    diffImage_zoomed = cv2.resize((np.abs(diffImage)*255).astype(np.uint8), None, fx = rescale_factor, fy = rescale_factor, interpolation=cv2.INTER_NEAREST)
    
    cv2.polylines(A_gt_zoomed, (points_deformed_gt[tri.simplices]*64*rescale_factor).astype(np.int32), isClosed=True,color=(0,0,0),lineType=cv2.LINE_AA) 
    
    cv2.polylines(Abuffer_zoomed, (points_deformed[tri.simplices]*64*rescale_factor).astype(np.int32), isClosed=True,color=(0,0,0),lineType=cv2.LINE_AA) 
    cv2.imshow('animation', np.column_stack((A_gt_zoomed,Abuffer_zoomed,diffImage_zoomed)))
    key = cv2.waitKey(1) 
    
    # get gradient on pca coefs
    coefs_grad= faces_pca.components_.dot(scene.texture_b.flatten())
    points_deformed_grad = np.zeros(points_deformed.shape)
    np.add.at(points_deformed_grad, tri.simplices, scene.ij_b*64)  
    print (np.max(np.abs(points_deformed_grad)))
    grads={'points_deformed':points_deformed_grad,'pca_coefs':coefs_grad}
    return Err,grads 

nbIter = 100

pca_coefs = np.zeros((faces_pca.n_components))
points_deformed = points.copy()



variables={'points_deformed':points_deformed,'pca_coefs':pca_coefs}
lambdas={'points_deformed':0.0001,'pca_coefs':0.5}


for iter in range(nbIter):

    E, grads = fun(**variables)
    print (f'E={E}')
    for name in variables.keys():
        variables[name] = variables[name]-lambdas[name]*grads[name] 
    
    variables['points_deformed'][on_border]=points[on_border]
    
key = cv2.waitKey() 












