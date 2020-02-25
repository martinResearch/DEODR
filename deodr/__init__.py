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
from .differentiable_renderer import Scene2D, Scene3D, Camera
from .obj import read_obj
from .laplacian_rigid_energy import LaplacianRigidEnergy
from .triangulated_mesh import TriMesh, ColoredTriMesh
import os

data_path = os.path.join(os.path.dirname(__file__), "data")
