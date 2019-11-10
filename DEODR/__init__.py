# -*- coding: utf-8 -*-
"""DEODR

Discontinuity-Edge-Overdraw Differentiable Rendering

If you use this library please cite
Model-based 3D Hand Pose Estimation from Monocular Video.
M. de la Gorce, N. Paragios and David Fleet. PAMI 2011

Martin de La Gorce. 2019.
"""

from .differentiable_renderer import Scene2D, Scene3D
from .obj import readObj
from .laplacian_rigid_energy import LaplacianRigidEnergy
from .triangulated_mesh import TriMesh

# from .tensorflow.triangulated_mesh_tensorflow import *
