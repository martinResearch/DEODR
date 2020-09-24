# distutils: language=c++
from libcpp cimport bool
cdef extern from "../C++/DifferentiableRenderer.h":
	ctypedef struct Scene:
		unsigned int* faces;
		unsigned int* faces_uv;
		double* depths
		double* uv
		double* ij
		double* shade
		double* colors
		bool* edgeflags
		bool* textured
		bool* shaded
		int     nb_triangles
		int nb_vertices;
		bool clockwise;
		bool backface_culling;
		int nb_uv;
		int     height
		int     width
		int     nb_colors
		double* texture
		int  texture_height
		int  texture_width
		double* background
		double* uv_b
		double* ij_b
		double* shade_b
		double* colors_b
		double* texture_b
		bool strict_edge
	void renderScene(Scene scene,double* image,double* z_buffer,double sigma,bool antialiase_error ,double* obs,double*  err_buffer)
	void renderScene_B(Scene scene,double* image,double* z_buffer,double* image_b,double sigma,bool antialiase_error ,double* obs,double*  err_buffer, double* err_buffer_b)
