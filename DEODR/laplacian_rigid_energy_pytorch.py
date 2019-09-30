import numpy as np
import torch
from .laplacian_rigid_energy import LaplacianRigidEnergy

def scipySparseToTorch(sparse_matrix):
    coo = sparse_matrix.tocoo()   
    values = coo.data
    indices = np.vstack((coo.row, coo.col))    
    i = torch.LongTensor(indices.astype(np.int32))
    v = torch.DoubleTensor(values)
    shape = coo.shape    
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape))

class LaplacianRigidEnergyPytorch(LaplacianRigidEnergy):
    def __init__(self, mesh, vertices, cregu):
        super().__init__(mesh, vertices, cregu)
        self.cT_torch = scipySparseToTorch(self.cT)

    def eval(self,V, return_grad=True,return_hessian=True,refresh_rotations=True):
        assert( isinstance(V ,torch.Tensor))
        if  V.requires_grad:            
            diff=(V-self.Vref).flatten()
            gradV = self.cregu*(self.cT_torch.mm(diff[:,None])).reshape_as(V)
            E= 0.5*diff.dot(gradV.flatten())
            return E
        else:
            diff=(V - torch.tensor(self.Vref)).flatten()
            gradV = self.cregu*(self.cT_torch.mm(diff[:,None])).reshape_as(V)
            E= 0.5*diff.dot(gradV.flatten())
            if not(return_grad):
                assert(not(return_hessian))
                return E
            if not(return_hessian):
                return E,gradV  
            return E,gradV,self.approx_hessian
  