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

class MeshRGBFitter():
    
    def __init__(self, vertices, faces, defaultColor, defaultLight, cregu=2000, gamma=0.01, beta=0.01, inertia=0.9, damping=0.1, updateLights=True, updateColor=True):
        self.cregu = cregu
        self.gamma = gamma
        self.beta = beta
        self.inertia = inertia   
        self.damping = damping
        self.defaultColor = defaultColor
        self.defaultLight = defaultLight  
        self.updateLights = updateLights
        self.updateColor = updateColor
        self.mesh = TriMesh(faces[:,::-1].copy())#we do a copy to avoid negative stride not support by pytorch 
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices,axis=0))
        self.cameraCenter = objectCenter+np.array([0,0,9]) * objectRadius        
        self.scene = Scene3DPytorch()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergyPytorch(self.mesh, vertices, cregu)
        self.Vinit = copy.copy(vertices)        
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.reset()
        
    def reset(self):
        self.V = copy.copy(self.Vinit)
        self.handColor= copy.copy(self.defaultColor)
        self.ligthDirectional = copy.copy(self.defaultLight['directional'])
        self.ambiantLight = copy.copy(self.defaultLight['ambiant'])
        self.speed = np.zeros(self.V.shape)
        self.speed_ligthDirectional = np.zeros(self.ligthDirectional.shape)
        self.speed_ambiantLight = np.zeros(self.ambiantLight.shape)
        self.speed_handColor = np.zeros(self.handColor.shape)  

    def setImage(self,handImage):   
        self.SizeW=handImage.shape[1]
        self.SizeH=handImage.shape[0]        
        self.handImage=handImage
        focal=2*self.SizeW
    
        R=np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        T=-R.T.dot(self.cameraCenter)
        self.CameraMatrix=np.array([[focal,0,self.SizeW/2],[0,focal,self.SizeH/2],[0,0,1]]).dot(np.column_stack((R,T)))
        self.iter=0
        
    def setBackgroundColor(self,backgroundColor):        
        self.scene.setBackground(np.tile(backgroundColor[None,None,:],(self.SizeH,self.SizeW,1)))

    def step(self):
        V_with_grad = torch.tensor(self.V, dtype=torch.float64,requires_grad=True)
        ligthDirectional_with_grad = torch.tensor(self.ligthDirectional, dtype=torch.float64,requires_grad=True)
        ambiantLight_with_grad = torch.tensor(self.ambiantLight, dtype=torch.float64,requires_grad=True)
        handColor_with_grad = torch.tensor(self.handColor, dtype=torch.float64,requires_grad=True)

        self.mesh.setVertices(V_with_grad)
        self.scene.setLight(ligthDirectional = ligthDirectional_with_grad, ambiantLight = ambiantLight_with_grad )
        self.mesh.setVerticesColors(handColor_with_grad.repeat( [self.mesh.nbV,1]))

        Abuffer = self.scene.render(self.CameraMatrix,resolution=(self.SizeW,self.SizeH))
        projJac = self.scene.projectionsJacobian(self.CameraMatrix,self.V)
        #projJacSp=sparse.block_diag(projJac,format='csr')# horribly slow as it uses python loops
        i = np.tile((np.arange(projJac.shape[0])[:,None] * projJac.shape[1] + np.arange(projJac.shape[1])[None,:])[:,:,None], (1, 1, projJac.shape[2]))
        j = np.tile((np.arange(projJac.shape[0])[:,None] * projJac.shape[2] + np.arange(projJac.shape[2])[None,:])[:,:,None], (1, projJac.shape[1], 1))
        projJacSp = sparse.coo_matrix((projJac.flatten(), (i.flatten(), j.flatten())))
        
        diffImage = torch.sum((Abuffer-torch.tensor(self.handImage))**2, dim=2)
        loss = torch.sum(diffImage)

        loss.backward()
        EData = loss.detach().numpy()        

        GradData = V_with_grad.grad        
 
        E_rigid, grad_rigidity,approx_hessian_rigidity = self.rigidEnergy.eval(self.V)
        Energy = EData + E_rigid.numpy()
        print('Energy=%f : EData=%f E_rigid=%f'%(Energy,EData, E_rigid))

        #update vertices
        G = GradData.numpy() + grad_rigidity.numpy()
        
        if self.beta == 0:
            if self.Hfactorized is None :# it is the allways the same for the moment            
                self.H = approx_hessian_rigidity + self.gamma*sparse.eye(approx_hessian_rigidity.shape[0])
                self.Hfactorized = scipy.sparse.linalg.factorized(self.H.tocsc()) 
            step = - self.Hfactorized(G.numpy().flatten()).reshape(self.V.shape)
        else:   
            #if self.Hpreconditioner is None :
            #    Happrox=approx_hessian_rigidity+self.gamma*sparse.eye(approx_hessian_rigidity.shape[0])+self.beta*(projJacSp.T*projJacSp)
            #    self.Hpreconditioner = sparse.linalg.inv(Happrox.tocsc())       
                     
            self.H = approx_hessian_rigidity + self.gamma * sparse.eye(approx_hessian_rigidity.shape[0]) + self.beta * (projJacSp.T * projJacSp) * sparse.kron(sparse.diags(self.mesh.DegreeVE),sparse.eye(3))
            #step = - sparse.linalg.cg(self.H,G.numpy().flatten(),tol=0.001,maxiter=50,M=self.Hpreconditioner)[0].reshape(self.V.shape)
            step = - sparse.linalg.spsolve(self.H,G.flatten()).reshape(self.V.shape)
            #step = - sparse.linalg.cg(self.H,G.numpy().flatten(),tol=0.01,maxiter=20)[0].reshape(self.V.shape)
            #self.Hfactorized = scipy.sparse.linalg.factorized(self.H.tocsc()) 
            #step = - self.Hfactorized(G.numpy().flatten()).reshape(self.V.shape)

        self.speed = (1 - self.damping) * (self.speed * self.inertia + ( 1 - self.inertia ) * step)
        self.V = self.V + torch.tensor(self.speed) 

        #update lights
        if self.updateLights:
            step = - ligthDirectional_with_grad.grad.numpy() * 0.0001 
            self.speed_ligthDirectional = (1 - self.damping) * (self.speed_ligthDirectional * self.inertia+ ( 1 - self.inertia ) * step)
            self.ligthDirectional = self.ligthDirectional + self.speed_ligthDirectional    
            step = - ambiantLight_with_grad.grad.numpy() * 0.0001
            self.speed_ambiantLight = (1-self.damping)*(self.speed_ambiantLight * self.inertia+ ( 1 - self.inertia ) * step)
            self.ambiantLight = self.ambiantLight + self.speed_ambiantLight
        if self.updateColor:
            #update hand color
            step = - handColor_with_grad.grad.numpy() *0.00001
            self.speed_handColor = (1 - self.damping)*(self.speed_handColor * self.inertia + ( 1 - self.inertia ) * step)
            self.handColor = self.handColor + self.speed_handColor
        self.iter += 1
        return Energy, Abuffer.detach().numpy(), diffImage.detach().numpy()  
    
class MeshDepthFitterEnergy(torch.nn.Module):

    def __init__(self, vertices, faces, euler_init, translation_init, cregu = 2000): 
        super(MeshDepthFitterEnergy, self).__init__()
        self.mesh = TriMesh(faces[:,::-1].copy())#we do a copy to avoid negative stride not supported by pytorch 
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices,axis=0))
        self.cameraCenter = objectCenter+np.array([-0.5,0,5])*objectRadius      
        self.scene = Scene()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergyPytorch(self.mesh, vertices, cregu)
        self.Vinit = copy.copy(self.mesh.vertices)        
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.transformQuaternionInit = scipy.spatial.transform.Rotation.from_euler('zyx',euler_init).as_quat() 
        self.transformTranslationInit = translation_init
        self.Vertices = torch.nn.Parameter( torch.tensor(self.Vinit, dtype = torch.float64))
        self.quaternion =  torch.nn.Parameter( torch.tensor(self.transformQuaternionInit, dtype=torch.float64))
        self.translation =   torch.nn.Parameter( torch.tensor(self.transformTranslationInit, dtype=torch.float64))      
        
    def setMaxDepth(self,maxDepth):  
        self.scene.maxDepth = maxDepth
        self.scene.setBackground(np.full((self.SizeH,self.SizeW,1), maxDepth, dtype=np.float))       
        
    def setDepthScale(self,depthScale):
        self.depthScale = depthScale

    def setImage(self, handImage, focal=None):   
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]   
        assert(handImage.ndim == 2)
        self.handImage = handImage
        if focal is None:
            focal = 2 * self.SizeW
    
        R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        T = -R.T.dot(self.cameraCenter)
        self.CameraMatrix = np.array([[focal, 0, self.SizeW/2], [0, focal, self.SizeH/2],[0,0,1]]).dot(np.column_stack((R,T)))
        self.iter = 0        
         
    def forward(self):   
        q_normalized = self.quaternion/self.quaternion.norm()
        print (self.quaternion.norm())
        Vertices_centered = self.Vertices-torch.mean(self.Vertices,dim = 0)[None,:]
        V_transformed = qrot(q_normalized, Vertices_centered)+self.translation
        self.mesh.setVertices(V_transformed)        
        depth_scale = 1 * self.depthScale
        Depth = self.scene.renderDepth(self.CameraMatrix,resolution=(self.SizeW,self.SizeH),depth_scale = depth_scale)
        Depth = torch.clamp(Depth,0,self.scene.maxDepth)
        diffImage = torch.sum((Depth-torch.tensor(self.handImage[:,:,None]))**2, dim = 2)
        self.Depth = Depth
        self.diffImage = diffImage
        EData = torch.sum(diffImage)            
        E_rigid = self.rigidEnergy.eval(self.Vertices,return_grad = False, return_hessian = False)
        Energy = EData + E_rigid   
        self.loss = EData + E_rigid          
        print('Energy=%f : EData=%f E_rigid=%f'%(Energy, EData, E_rigid))
        return self.loss  
    
class MeshDepthFitterPytorchOptim():
    
    def __init__(self, vertices, faces, euler_init, translation_init, cregu=2000, lr=0.8):
        self.energy = MeshDepthFitterEnergy(vertices, faces, euler_init, translation_init, cregu)
        params = self.energy.parameters()
        self.optimizer = torch.optim.LBFGS(self.energy.parameters(), lr=0.8, max_iter=1)
        #self.optimizer = torch.optim.SGD(params, lr=0.000005, momentum=0.1,  dampening=0.1        )
        #self.optimizer =torch.optim.RMSprop(params, lr=1e-3, alpha=0.99,  eps=1e-8,  weight_decay=0,  momentum=0.001)   
        #self.optimizer = torch.optim.Adadelta(params, lr=0.1, rho=0.95,     eps=1e-6,  weight_decay=0)
        #self.optimizer = torch.optim.Adagrad(self.energy.parameters(), lr=0.02)
    def setImage(self,depth_image,focal):
        self.energy.setImage(depth_image, focal = focal)
    def setMaxDepth(self,maxDepth):
        self.energy.setMaxDepth(maxDepth)
    def setDepthScale(self,depthScale):
        self.energy.setDepthScale(depthScale)        
    def step(self):     
        def closure():
            self.optimizer.zero_grad()
            loss = self.energy()
            loss.backward()
            return loss        
        self.optimizer.step(closure)
        #self.iter += 1
        return self.energy.loss, self.energy.Depth[:,:,0].detach().numpy(), self.energy.diffImage.detach().numpy()    
    
class MeshDepthFitter():
    
    def __init__(self, vertices, faces, euler_init, translation_init, cregu=2000, inertia=0.96, damping=0.05):
        self.cregu = cregu
        self.inertia = inertia   
        self.damping = damping      
        self.step_factor_vertices= 0.0005
        self.step_max_vertices = 0.5
        self.step_factor_quaternion= 0.00006
        self.step_max_quaternion = 0.1
        self.step_factor_translation= 0.00005
        self.step_max_translation = 0.1     
        
        self.mesh = TriMesh(faces[:,::-1].copy())#we do a copy to avoid negative stride not support by pytorch 
        objectCenter = vertices.mean(axis=0)
        objectRadius = np.max(np.std(vertices,axis = 0))
        self.cameraCenter = objectCenter+np.array([-0.5,0,5]) * objectRadius  
        
        self.scene = Scene3DPytorch()
        self.scene.setMesh(self.mesh)
        self.rigidEnergy = LaplacianRigidEnergy(self.mesh, vertices, cregu)
        self.vertices_init = torch.tensor(copy.copy(vertices))        
        self.Hfactorized = None
        self.Hpreconditioner = None
        self.setMeshTransformInit(euler=euler_init, translation=translation_init)
        self.reset()
        
    def setMeshTransformInit(self,euler,translation):        
        self.transformQuaternionInit = scipy.spatial.transform.Rotation.from_euler('zyx',euler).as_quat()
        self.transformTranslationInit = translation
        
    def reset(self):
        self.vertices = copy.copy(self.vertices_init)      
        self.speed_vertices = np.zeros(self.vertices_init.shape)
        self.transformQuaternion = copy.copy(self.transformQuaternionInit)
        self.transformTranslation = copy.copy(self.transformTranslationInit)
        self.speed_translation = np.zeros(3)
        self.speed_quaternion = np.zeros(4)        
    
    def setMaxDepth(self, maxDepth):  
        self.scene.maxDepth = maxDepth
        self.scene.setBackground(np.full((self.SizeH, self.SizeW,1), maxDepth, dtype=np.float))       
        
    def setDepthScale(self,depthScale):
        self.depthScale = depthScale

    def setImage(self, handImage, focal=None):   
        self.SizeW = handImage.shape[1]
        self.SizeH = handImage.shape[0]   
        assert(handImage.ndim == 2)
        self.handImage = handImage
        if focal is None:
            focal = 2*self.SizeW
    
        R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        T = -R.T.dot(self.cameraCenter)
        self.CameraMatrix = np.array([[focal,0,self.SizeW/2],[0,focal,self.SizeH/2],[0,0,1]]).dot(np.column_stack((R,T)))
        self.iter = 0        
    
    def step(self):
        self.vertices = self.vertices - torch.mean(self.vertices, dim=0)[None,:]
        #vertices_with_grad = self.vertices.clone().requires_grad(True)
        vertices_with_grad = self.vertices.clone().detach().requires_grad_(True)        
        vertices_with_grad_centered = vertices_with_grad-torch.mean(vertices_with_grad, dim = 0)[None,:]
        quaternion_with_grad = torch.tensor(self.transformQuaternion, dtype = torch.float64, requires_grad = True)
        translation_with_grad = torch.tensor(self.transformTranslation, dtype = torch.float64, requires_grad = True)        
        
        q_normalized = quaternion_with_grad/quaternion_with_grad.norm() # that will lead to a gradient that is in the tangeant space
        vertices_with_grad_transformed = qrot(q_normalized, vertices_with_grad_centered)+translation_with_grad
      
        self.mesh.setVertices(vertices_with_grad_transformed)

        depth_scale = 1 * self.depthScale
        Depth = self.scene.renderDepth(self.CameraMatrix,resolution=(self.SizeW,self.SizeH),depth_scale=depth_scale)
        Depth = torch.clamp(Depth,0,self.scene.maxDepth)
        
        diffImage = torch.sum((Depth-torch.tensor(self.handImage[:,:,None]))**2,dim=2)
        loss = torch.sum(diffImage)

        loss.backward()
        EData = loss.detach().numpy()        

        GradData = vertices_with_grad.grad.numpy()
        
        E_rigid, grad_rigidity, approx_hessian_rigidity = self.rigidEnergy.eval(self.vertices.numpy())
        Energy = EData + E_rigid
        print('Energy=%f : EData=%f E_rigid=%f'%(Energy,EData,E_rigid))

        #update v
        G = GradData + grad_rigidity
        
        def mult_and_clamp(x,a,t):
            return np.minimum(np.maximum(x*a,-t),t)
   
        #update vertices
        step_vertices = mult_and_clamp(-G ,self.step_factor_vertices, self.step_max_vertices)        
        self.speed_vertices = (1 - self.damping) * (self.speed_vertices * self.inertia+ ( 1 - self.inertia ) * step_vertices)
        self.vertices = self.vertices + torch.tensor(self.speed_vertices) 
        #update rotation
        step_quaternion = mult_and_clamp(-quaternion_with_grad.grad.numpy(), self.step_factor_quaternion, self.step_max_quaternion)  
        self.speed_quaternion = (1 - self.damping) * (self.speed_quaternion * self.inertia + ( 1 - self.inertia ) * step_quaternion)   
        self.transformQuaternion =  self.transformQuaternion + self.speed_quaternion
        #update translation
        self.transformQuaternion = self.transformQuaternion/np.linalg.norm(self.transformQuaternion)         
        step_translation = mult_and_clamp( -translation_with_grad.grad.numpy(), self.step_factor_translation, self.step_max_translation)
        self.speed_translation = (1 - self.damping) * (self.speed_translation * self.inertia + ( 1 - self.inertia ) * step_translation)
        self.transformTranslation = self.transformTranslation + self.speed_translation         

        self.iter += 1
        return Energy, Depth[:,:,0].detach().numpy(), diffImage.detach().numpy()        
    
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