# distutils: language=c++
from libcpp cimport bool
cdef extern from "../c++/DifferentiableRenderer.h":
	ctypedef struct Scene:
		double* depths
		double* uv
		double* ij
		double* shade
		double* colors
		bool* edgeflags
		bool* textured
		bool* shaded
		int     nbTriangles
		int     image_H
		int     image_W
		int     nbColors
		double* texture
		int  texture_H
		int  texture_W
		double* background
		double* uv_b
		double* ij_b
		double* shade_b
		double* colors_b
		double* texture_b
	void renderScene(Scene scene,double* Abuffer,double* Zbuffer,double sigma,bool antialiaseError ,double* Aobs,double*  ErrBuffer)
	void renderScene_B(Scene scene,double* Abuffer,double* Zbuffer,double* Abuffer_b,double sigma,bool antialiaseError ,double* Aobs,double*  ErrBuffer, double* ErrBuffer_b)
