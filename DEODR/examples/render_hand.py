from DEODR.obj import readObj
from DEODR.triangulated_mesh import TriMesh
from DEODR import differentiable_renderer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


faces, vertices = readObj("../../data/hand.obj")

mesh = TriMesh(
     faces,
     vertices=vertices,
     clockwise=False
)  



SizeW = 640
SizeH = 480

objectCenter = 0.5*(vertices.max(axis=0)+vertices.min(axis=0))
objectRadius = np.max(vertices.max(axis=0)-vertices.min(axis=0))
cameraCenter = objectCenter + np.array([0, 0, 3]) * objectRadius
focal = 2 * SizeW

R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
T = -R.T.dot(cameraCenter)
extrinsics = np.column_stack((R, T))
intrinsics = np.array([[focal, 0, SizeW / 2], [0, focal, SizeH / 2], [0, 0, 1]])

CameraMatrix = intrinsics.dot(extrinsics)

handColor = np.array([200, 100, 100]) / 255
mesh.setVerticesColors(np.tile(handColor, [mesh.nbV, 1]))

scene = differentiable_renderer.Scene3D()
scene.setLight(ligthDirectional=np.array([-0.8, 0, -0.8]), ambiantLight=0.3)
scene.setMesh(mesh)
backgroundImage=np.ones((SizeH,SizeW,3))
scene.setBackground(backgroundImage)

Abuffer = scene.render(CameraMatrix, resolution=(SizeW, SizeH))
plt.figure()
plt.imshow(Abuffer)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z = vertices.T
u,v,w = mesh.vertexNormals.T
ax.plot_trisurf(vertices[:,0], vertices[:,1], Z= vertices[:,2], triangles=mesh.faces)
ax.quiver(x, y, z, u, v, w, length=0.03, normalize=True,color=[0,1,0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
u,v,w= scene.ligthDirectional
ax.quiver(np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([u]),np.array([v]),np.array([w]),color=[1,1,0.5])
plt.show()

