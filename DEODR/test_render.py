import obj
from DEODR.triangulated_mesh import TriMesh
from scipy.misc import imread
import numpy as np
from DEODR import differentiable_renderer
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import torch
import copy
import cv2

faces, vertices = obj.readObj("../data/hand.obj")

mesh = TriMesh(
    faces[:, ::-1],  # flip the faces to get the right normals orientation
    vertices=vertices,
)

SizeW = 640
SizeH = 480

objectCenter = vertices.mean(axis=0)
objectRadius = np.max(np.std(vertices, axis=0))
cameraCenter = objectCenter + np.array([0, 0, 9]) * objectRadius
focal = 2 * SizeW

R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
T = -R.T.dot(cameraCenter)
extrinsics = np.column_stack((R, T))
intrinsics = np.array([[focal, 0, SizeW / 2], [0, focal, SizeH / 2], [0, 0, 1]])

CameraMatrix = intrinsics.dot(extrinsics)

handColor = np.array([200, 100, 100]) / 255
mesh.setVerticesColors(np.tile(handColor, [mesh.nbV, 1]))

scene = differentiable_renderer.Scene3D()
scene.setLight(ligthDirectional=np.array([0.8, 0.5, 0.5]), ambiantLight=0.3)
scene.setMesh(mesh)
backgroundImage = np.ones((SizeH, SizeW, 3))
scene.setBackground(backgroundImage)

Abuffer = scene.render(CameraMatrix, resolution=(SizeW, SizeH))
plt.imshow(Abuffer)
plt.show()
