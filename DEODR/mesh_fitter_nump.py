from DEODR.pytorch import Scene3DPytorch, LaplacianRigidEnergyPytorch
from DEODR import LaplacianRigidEnergy
from DEODR.pytorch import TriMeshPytorch as TriMesh
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy.spatial.transform.rotation
import torch
import copy
import cv2

def qrot(q, v):
    qr=q[None,:].repeat(v.shape[0],1)
    qvec = qr[:,:-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (qr[:, [3]] * uv + uuv))



class MeshRGBFitterWithPose():
    
    def __init__(self, vertices, faces, euler_init, translation_init, defaultColor, defaultLight, cregu=2000, inertia=0.96, damping=0.05, updateLights=True, updateColor=True):
        self.cregu = cregu

        self.inertia = inertia   
        self.damping = damping      
        self.step_factor_vertices= 0.0005
        self.step_max_vertices = 0.5
        self.step_factor_quaternion= 0.00006
        self.step_max_quaternion = 0.05
        self.step_factor_translation= 0.00005
        self.step_max_translation = 0.1     
        
        self.defaultColor = defaultColor
        self.defaultLight = defaultLight          
        self.updateLights = updateLights
        self.updateColor = updateColor        
        self.mesh = TriMesh(faces[:,::-1].copy())#we do a copy to avoid negative stride not support by pytorch 
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices,axis=0))
        self.cameraCenter = objectCenter + np.array([0,0,9]) * objectRadius    
        
        self.scene = Scene3DPytorch()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy=LaplacianRigidEnergyPytorch(self.mesh, vertices, cregu)
        self.vertices_init = torch.tensor(copy.copy(vertices))        
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.setMeshTransformInit(euler = euler_init, translation = translation_init)
        self.reset()        
        
    def setBackgroundColor(self,backgroundColor):        
        self.scene.setBackground(np.tile(backgroundColor[None,None,:], (self.SizeH,self.SizeW,1)))        
    def setMeshTransformInit(self,euler,translation):        
        self.transformQuaternionInit = scipy.spatial.transform.Rotation.from_euler('zyx',euler).as_quat()
        self.transformTranslationInit = translation
        
    def reset(self):
        self.vertices = copy.copy(self.vertices_init)      
        self.speed_vertices = np.zeros(self.vertices.shape)
        self.transformQuaternion = copy.copy(self.transformQuaternionInit)
        self.transformTranslation = copy.copy(self.transformTranslationInit)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)
        
        self.handColor= copy.copy(self.defaultColor)
        self.ligthDirectional = copy.copy(self.defaultLight['directional'])
        self.ambiantLight = copy.copy(self.defaultLight['ambiant'])

        self.speed_ligthDirectional = np.zeros(self.ligthDirectional.shape)
        self.speed_ambiantLight = np.zeros(self.ambiantLight.shape)
        self.speed_handColor = np.zeros(self.handColor.shape)      

    def setImage(self,handImage,focal=None):   
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]   
        assert(handImage.ndim == 3)
        self.handImage = handImage
        if focal is None:
            focal=2*self.SizeW
    
        R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        T = -R.T.dot(self.cameraCenter)
        self.CameraMatrix = np.array([[focal,0,self.SizeW/2],[0,focal,self.SizeH/2],[0,0,1]]).dot(np.column_stack((R,T)))
        self.iter=0        
   
    def step(self):
        self.vertices = self.vertices - torch.mean(self.vertices, dim = 0)[None,:]
        vertices_with_grad = torch.tensor(self.vertices, dtype = torch.float64, requires_grad = True)
        vertices_with_grad_centered = vertices_with_grad - torch.mean(vertices_with_grad, dim = 0)[None,:]
        quaternion_with_grad = torch.tensor(self.transformQuaternion, dtype=torch.float64, requires_grad = True)
        translation_with_grad = torch.tensor(self.transformTranslation, dtype=torch.float64, requires_grad = True)
               
        ligthDirectional_with_grad = torch.tensor(self.ligthDirectional, dtype = torch.float64, requires_grad=True)
        ambiantLight_with_grad = torch.tensor(self.ambiantLight, dtype = torch.float64, requires_grad = True)
        handColor_with_grad = torch.tensor(self.handColor, dtype=torch.float64, requires_grad = True)
    
        q_normalized = quaternion_with_grad/quaternion_with_grad.norm() # that will lead to a gradient that is in the tangeant space
        vertices_with_grad_transformed = qrot(q_normalized, vertices_with_grad_centered) + translation_with_grad
        self.mesh.setVertices(vertices_with_grad_transformed)
        
        self.scene.setLight(ligthDirectional = ligthDirectional_with_grad, ambiantLight = ambiantLight_with_grad )
        self.mesh.setVerticesColors(handColor_with_grad.repeat( [self.mesh.nbV,1]))
       
        Abuffer = self.scene.render(self.CameraMatrix, resolution = (self.SizeW,self.SizeH))    
        
        diffImage = torch.sum((Abuffer-torch.tensor(self.handImage))**2, dim = 2)
        loss = torch.sum(diffImage)

        loss.backward()
        EData = loss.detach().numpy()        

        GradData = vertices_with_grad.grad
        
        E_rigid,grad_rigidity,approx_hessian_rigidity = self.rigidEnergy.eval(self.vertices)
        Energy = EData + E_rigid.numpy()
        print('Energy=%f : EData=%f E_rigid=%f'%(Energy, EData,E_rigid))

        #update v
        G = GradData + grad_rigidity
        
        def mult_and_clamp(x,a,t):
            return np.minimum(np.maximum(x * a,-t), t)        
        
        inertia=self.inertia
   
        #update vertices
        step_vertices = mult_and_clamp(-G.numpy(), self.step_factor_vertices, self.step_max_vertices)        
        self.speed_vertices = (1 - self.damping) * (self.speed_vertices * inertia+ ( 1 - inertia ) * step_vertices)
        self.vertices = self.vertices + torch.tensor(self.speed_vertices) 
        #update rotation
        step_quaternion = mult_and_clamp(-quaternion_with_grad.grad.numpy(), self.step_factor_quaternion, self.step_max_quaternion)  
        self.speed_quaternion = (1-self.damping) * (self.speed_quaternion * inertia + ( 1 - inertia ) *step_quaternion)   
        self.transformQuaternion=  self.transformQuaternion + self.speed_quaternion
        #update translation
        self.transformQuaternion = self.transformQuaternion/np.linalg.norm(self.transformQuaternion)         
        step_translation = mult_and_clamp(-translation_with_grad.grad.numpy(), self.step_factor_translation, self.step_max_translation)
        self.speed_translation = (1 - self.damping)*(self.speed_translation * inertia + ( 1 - inertia ) * step_translation)
        self.transformTranslation = self.transformTranslation + self.speed_translation
        #update directional light
        step = - ligthDirectional_with_grad.grad.numpy() * 0.0001 
        self.speed_ligthDirectional = (1 - self.damping) * (self.speed_ligthDirectional * inertia+ ( 1 - inertia ) * step)
        self.ligthDirectional = self.ligthDirectional + self.speed_ligthDirectional
        #update ambiant light
        step = - ambiantLight_with_grad.grad.numpy() * 0.0001
        self.speed_ambiantLight = (1-self.damping)*(self.speed_ambiantLight * inertia + ( 1 - inertia ) * step)
        self.ambiantLight = self.ambiantLight + self.speed_ambiantLight
        #update hand color
        step = - handColor_with_grad.grad.numpy() *0.00001
        self.speed_handColor = (1 - self.damping) * (self.speed_handColor * inertia+ ( 1 - inertia ) * step)
        self.handColor = self.handColor + self.speed_handColor         

        self.iter += 1
        return Energy, Abuffer.detach().numpy(), diffImage.detach().numpy()        