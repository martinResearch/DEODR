# distutils: language=c++
from libcpp cimport bool
from libcpp.vector cimport vector
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
		double* background_image
		double* background_color
		double* uv_b
		double* ij_b
		double* shade_b
		double* colors_b
		double* texture_b
		bool strict_edge
		bool perspective_correct
		bool integer_pixel_centers

	ctypedef struct FragmentsDouble:
		vector[int] list_x
		vector[int] list_y
		vector[double] list_values
		vector[double] list_alpha
		int nb_channels
		int width
		int height

	void renderScene(Scene scene,double* image,double* z_buffer,double sigma,bool antialiase_error ,double* obs,double*  err_buffer)
	void renderScene_B(Scene scene,double* image,double* z_buffer,double* image_b,double sigma,bool antialiase_error ,double* obs,double*  err_buffer, double* err_buffer_b)
	void renderSceneFragments(Scene scene,double* z_buffer, double sigma, FragmentsDouble fragments)
