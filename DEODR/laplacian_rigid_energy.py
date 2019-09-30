import numpy as np
import scipy
import copy
import scipy.sparse

class LaplacianRigidEnergy():
    def __init__(self,mesh,vertices,cregu):
        self.cT = scipy.sparse.kron(mesh.adjacencies.Laplacian.T*mesh.adjacencies.Laplacian,scipy.sparse.eye(3))
        self.Vref = copy.copy(vertices)
        self.mesh=mesh
        self.cregu=cregu      
        self.approx_hessian = self.cregu*self.cT 
        n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=self.mesh.adjacencies.Adjacency_Vertices, directed=False, return_labels=True)
        if n_components>1:
            raise(BaseException("you have more than one connected component in your mesh"))
    
    def eval(self,V, return_grad=True,return_hessian=True,refresh_rotations=True):

        diff=(V-self.Vref).flatten()               
        gradV =self.cregu*(self.cT*diff).reshape((V.shape[0],3))
        E= 0.5*diff.dot(gradV.flatten())
        if not(return_grad):
            assert(not(return_hessian))
            return E
        if not(return_hessian):
            return E,gradV                
                   
        return E,gradV,self.approx_hessian                
      
   