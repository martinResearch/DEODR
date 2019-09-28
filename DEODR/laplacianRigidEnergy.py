import numpy as np
import scipy
import torch
import copy
import scipy.sparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def scipySparseToTorch(sparse_matrix):
    coo = sparse_matrix.tocoo()   
    values = coo.data
    indices = np.vstack((coo.row, coo.col))    
    i = torch.LongTensor(indices.astype(np.int32))
    v = torch.DoubleTensor(values)
    shape = coo.shape    
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape))

class LaplacianRigidEnergy():
    def __init__(self,mesh,cregu):
        self.cT = scipy.sparse.kron(mesh.Laplacian.T*mesh.Laplacian,scipy.sparse.eye(3))
        self.cT_torch = scipySparseToTorch(self.cT)
        self.Vref = copy.copy(mesh.vertices)
        self.mesh=mesh
        self.cregu=cregu      
        self.approx_hessian = self.cregu*self.cT 
        n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=self.mesh.Adjacency_Vertices, directed=False, return_labels=True)
        if n_components>1:
            raise(BaseException("you have more than one connected component in your mesh"))
    
    def eval(self,V, return_grad=True,return_hessian=True,refresh_rotations=True):

        if isinstance(V ,torch.Tensor):
            if  V.requires_grad:            
                diff=(V-self.Vref).flatten()
                gradV = self.cregu*(self.cT_torch.mm(diff[:,None])).reshape_as(V)
                E= 0.5*diff.dot(gradV.flatten())
                return E
            else:
                diff=(V-self.Vref).flatten()
                gradV = self.cregu*(self.cT_torch.mm(diff[:,None])).reshape_as(V)
                E= 0.5*diff.dot(gradV.flatten())
                if not(return_grad):
                    assert(not(return_hessian))
                    return E
                if not(return_hessian):
                    return E,gradV          
                              
                return E,gradV,self.approx_hessian
        else:
            diff=(V-self.Vref).flatten()               
            gradV =self.cregu*(self.cT*diff).reshape((V.shape[0],3))
            E= 0.5*diff.dot(gradV.flatten())
            if not(return_grad):
                assert(not(return_hessian))
                return E
            if not(return_hessian):
                return E,gradV                
                       
            return E,gradV,self.approx_hessian                
          
   