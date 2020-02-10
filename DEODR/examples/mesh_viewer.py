from DEODR.obj import readObj
from DEODR.triangulated_mesh import ColoredTriMesh
from DEODR import differentiable_renderer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import cv2
import time

#obj_file="../../data/hand.obj"
obj_file="models/crate.obj"
obj_file="models/duck.obj"
#obj_file="models/drill.obj"
#obj_file="models/fuze.obj"

 
mesh_trimesh = trimesh.load(obj_file)
mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)

ax=plt.subplot(111)
if mesh.textured:
    mesh.plot_uv_map(ax)

SizeW = 640
SizeH = 480

objectCenter = 0.5*(mesh.vertices.max(axis=0)+mesh.vertices.min(axis=0))
objectRadius = np.max(mesh.vertices.max(axis=0)-mesh.vertices.min(axis=0))
cameraCenter = objectCenter + np.array([0, 0, 4]) * objectRadius
focal = 2 * SizeW

R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
T = -R.T.dot(cameraCenter)
extrinsics = np.column_stack((R, T))
intrinsics = np.array([[focal, 0, SizeW / 2], [0, focal, SizeH / 2], [0, 0, 1]])

CameraMatrix = intrinsics.dot(extrinsics)

handColor = np.array([200, 100, 100]) / 255
mesh.setVerticesColors(np.tile(handColor, [mesh.nbV, 1]))

scene = differentiable_renderer.Scene3D()
scene.setLight(ligthDirectional=np.array([-0.5, 0, -0.5]), ambiantLight=0.3)
scene.setMesh(mesh)
backgroundImage=np.ones((SizeH,SizeW,3))
scene.setBackground(backgroundImage)

mesh.texture=mesh.texture[:,:,::-1]# convert texture to GBR to avoid future conversion when ploting in Opencv

fps=0
fps_decay=0.1
windowname=  f"DEODR mesh viewer:{obj_file}"


def mouseCallback(event,x,y,flags,param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print('left button down')
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:    
        print('left button up')    
cv2.namedWindow(windowname)
cv2.setMouseCallback(windowname,mouseCallback)
        
while True:
    #mesh.setVertices(mesh.vertices+np.random.randn(*mesh.vertices.shape)*0.001)
    start= time.clock()
    Abuffer = scene.render(CameraMatrix, resolution=(SizeW, SizeH)) 
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20,SizeH-20)
    fontScale              = 1
    fontColor              = (0,0,255)
    thickness               = 2
     
    cv2.putText(Abuffer,'fps:%0.1f'%fps, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness)
    
    cv2.imshow(windowname, Abuffer)    
    
    stop=time.clock()
    fps= (1-fps_decay)*fps+fps_decay*( 1/(stop-start))    
    key = cv2.waitKey(1)


