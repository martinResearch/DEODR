# -*- coding: utf-8 -*-
"""deodr

Discontinuity-Edge-Overdraw Differentiable Rendering

If you use this library please cite
Model-based 3D Hand Pose Estimation from Monocular Video.
M. de la Gorce, N. Paragios and David Fleet. PAMI 2011

Martin de La Gorce. 2019.
"""
__all__ = [
    "Scene2D",
    "Scene3D",
    "Camera",
    "read_obj",
    "LaplacianRigidEnergy",
    "TriMesh",
    "ColoredTriMesh",
]

import os

from .differentiable_renderer import Camera, Scene2D, Scene3D
from .laplacian_rigid_energy import LaplacianRigidEnergy
from .obj import read_obj
from .triangulated_mesh import ColoredTriMesh, TriMesh

data_path = os.path.join(os.path.dirname(__file__), "data")
