import obj
from triangulated_mesh import Mesh
from scipy.misc import imread
import numpy as np
import differentiablerenderer
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import torch
import copy
import cv2


faces, vertices = obj.readObj("../../../data/hand.obj")
# vertices=npAD.array(vertices)
mesh = Mesh(
    vertices, faces[:, ::-1].copy()
)  # we do a copy to avoid negative stride not support by pytorch


handImage = imread("../../../data/hand.png").astype(np.double) / 255

SizeW = handImage.shape[1]
SizeH = handImage.shape[0]
# SizeW=500
# SizeH=300
objectCenter = vertices.mean(axis=0)
objectRadius = np.max(np.std(vertices, axis=0))
cameraCenter = objectCenter + np.array([0, 0, 9]) * objectRadius
focal = 2 * SizeW
# focal=600
R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
T = -R.T.dot(cameraCenter)
CameraMatrix = np.array([[focal, 0, SizeW / 2], [0, focal, SizeH / 2], [0, 0, 1]]).dot(
    np.column_stack((R, T))
)

handRegion = handImage[90:120, 80:120, :]
# plt.imshow(handRegion)
# plt.show()
handColor = np.array([200, 100, 100]) / 255
mesh.setVerticesColors(np.tile(handColor, [mesh.nbV, 1]))


scene = differentiable_renderer.Scene()
scene.setLight(ligthDirectional=np.array([0.1, 0.5, 0.5]), ambiantLight=0.3)
scene.setMesh(mesh)
# Abuffer,Zbuffer=scene.render(CameraMatrix,resolution=(SizeW,SizeH))
# plt.imshow(Abuffer)
# plt.show()
cregu = 5000
gamma = 300
alpha = 0.01
beta = 0.01
inertia = 0.85
cT = cregu * sparse.kron(mesh.Laplacian.T * mesh.Laplacian, sparse.eye(3))
cTplusGama = cT + gamma * sparse.eye(cT.shape[0])

solvecTplusGama = scipy.sparse.linalg.factorized(cTplusGama)
nbMaxIter = 400
V = mesh.vertices
Vref = copy.copy(V)
speed = np.zeros(V.shape)
# plt.ion()


for iter in range(nbMaxIter):

    V_with_grad = torch.tensor(V, requires_grad=True)
    mesh.setVertices(V_with_grad)

    Abuffer = scene.render(CameraMatrix, resolution=(SizeW, SizeH))

    diffImage = torch.sum((Abuffer - torch.tensor(handImage)) ** 2, dim=2)
    # plt.imshow(diffImage.detach().numpy())

    cv2.imshow("animation", diffImage.detach().numpy())
    cv2.waitKey(1)
    loss = torch.sum(diffImage)
    loss.backward()
    GradData = V_with_grad.grad

    G = GradData + torch.tensor(cT * (V - Vref).numpy().flatten()).reshape_as(V)

    step = -solvecTplusGama(G.numpy().flatten()).reshape(V.shape)
    speed = speed * inertia + (1 - inertia) * step
    V = V + torch.tensor(speed)
