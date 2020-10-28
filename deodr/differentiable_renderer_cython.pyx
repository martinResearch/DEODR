# distutils: language = c++
from libcpp cimport bool
cimport _differentiable_renderer 

import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np



@cython.boundscheck(False)
@cython.wraparound(False)
def renderScene(scene, 
		double sigma,
		np.ndarray[double,ndim = 3,mode = "c"] image, 
		np.ndarray[double,ndim = 2,mode = "c"] z_buffer,
		bool antialiase_error  = 0,
		np.ndarray[double,ndim = 3,mode = "c"] obs = None,
		np.ndarray[double,ndim = 2,mode = "c"] err_buffer = None, 
		bool check_valid = 1):
 
	cdef _differentiable_renderer.Scene scene_c 

	if check_valid:
		assert (not(image is None))
		assert (not(z_buffer is None))

	heigth  =  image.shape[0]
	width  =  image.shape[1]
	nb_colors  =  image.shape[2]
		
	nb_triangles  =  scene.faces.shape[0]
	assert(nb_triangles  ==  scene.faces_uv.shape[0])
	nb_vertices  =  scene.depths.shape[0]
	nb_vertices_uv  =  scene.uv.shape[0]

	if check_valid:	
		assert(scene.faces.dtype  ==  np.uint32)
		assert(np.all(scene.faces < nb_vertices))
		assert(np.all(scene.faces_uv < nb_vertices_uv))
				
		assert(scene.colors.ndim  ==  2)
		assert(scene.uv.ndim  ==  2)
		assert(scene.ij.ndim  ==  2)
		assert(scene.shade.ndim  ==  1)
		assert(scene.edgeflags.ndim  ==  2)
		assert(scene.textured.ndim  ==  1)
		assert(scene.shaded.ndim  ==  1)		
		assert(scene.uv.shape[1]  ==  2)	
		assert(scene.ij.shape[0]  ==  nb_vertices)
		assert(scene.ij.shape[1]  ==  2)
		assert(scene.shade.shape[0]  ==  nb_vertices)
		assert(scene.colors.shape[0]  ==  nb_vertices)
		assert(scene.colors.shape[1]  ==  nb_colors)
		assert(scene.edgeflags.shape[0]  ==  nb_triangles)
		assert(scene.edgeflags.shape[1]  ==  3)
		assert(scene.textured.shape[0]  ==  nb_triangles)
		assert(scene.shaded.shape[0]  ==  nb_triangles)
		assert(scene.background.ndim  ==  3)
		assert(scene.background.shape[0]  ==  heigth)
		assert(scene.background.shape[1]  ==  width)
		assert(scene.background.shape[2]  ==  nb_colors)
		
		if scene.texture.size>0:
			assert(scene.texture.ndim  ==  3)
			assert(scene.texture.shape[0]>0)
			assert(scene.texture.shape[1]>0)
			assert(scene.texture.shape[2]  ==  nb_colors)
		
		assert z_buffer.shape[0]  ==  heigth 
		assert z_buffer.shape[1]  ==  width 

	scene_c.nb_colors = nb_colors
	cdef np.ndarray[np.uint32_t, mode = "c"] faces_c  =  np.ascontiguousarray(scene.faces.flatten(), dtype = np.uint32)
	cdef np.ndarray[np.uint32_t, mode = "c"] faces_uv_c  =  np.ascontiguousarray(scene.faces_uv.flatten(), dtype = np.uint32)		
	cdef np.ndarray[np.double_t, mode = "c"] depths_c  =  np.ascontiguousarray(scene.depths.flatten(), dtype = np.double)	
	cdef np.ndarray[np.double_t, mode = "c"] uv_c  =  np.ascontiguousarray(scene.uv.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] ij_c  =  np.ascontiguousarray(scene.ij.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] shade_c  =  np.ascontiguousarray(scene.shade.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] colors_c  =  np.ascontiguousarray(scene.colors.flatten(), dtype = np.double)
	cdef np.ndarray[np.uint8_t, mode = "c"] edgeflags_c  =  np.ascontiguousarray(scene.edgeflags.flatten(), dtype = np.uint8)
	cdef np.ndarray[np.uint8_t, mode = "c"] textured_c  =  np.ascontiguousarray(scene.textured.flatten(), dtype = np.uint8)
	cdef np.ndarray[np.uint8_t, mode = "c"] shaded_c  =  np.ascontiguousarray(scene.shaded.flatten(), dtype = np.uint8)
	cdef np.ndarray[np.double_t, mode = "c"] texture_c  =  np.ascontiguousarray(scene.texture.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] background_c  =  np.ascontiguousarray(scene.background.flatten(), dtype = np.double)
	
	scene_c.height = <int> scene.height
	scene_c.width = <int> scene.width	
	scene_c.nb_triangles = nb_triangles
	scene_c.nb_vertices = nb_vertices
	scene_c.backface_culling = scene.backface_culling
	scene_c.clockwise = scene.clockwise
	scene_c.nb_uv = nb_vertices_uv
	scene_c.faces = <unsigned int*> faces_c.data
	scene_c.faces_uv = <unsigned int*> faces_uv_c.data
	scene_c.depths = <double*> depths_c.data
	scene_c.uv = <double*> uv_c.data
	scene_c.ij = <double*> ij_c.data
	scene_c.shade = <double*> shade_c.data
	scene_c.colors = <double*> colors_c.data
	scene_c.edgeflags = <bool*> edgeflags_c.data
	scene_c.textured = <bool*> textured_c.data
	scene_c.shaded = <bool*> shaded_c.data
	scene_c.texture = <double*> texture_c.data
	scene_c.background = <double*> background_c.data
	scene_c.texture_height = scene.texture.shape[0]
	scene_c.texture_width = scene.texture.shape[1]
	scene_c.strict_edge = scene.strict_edge
	


	cdef double* obs_ptr = NULL
	cdef double* err_buffer_ptr = NULL
	
	cdef double* image_ptr =  <double*> image.data
	cdef double* z_buffer_ptr =  <double*> z_buffer.data
	
	if image_ptr  ==  NULL:
		raise BaseException('image_ptr is NULL')
	if z_buffer_ptr  ==  NULL:
		raise BaseException('z_buffer_ptr is NULL')

	
	if antialiase_error:
		assert err_buffer.shape[0]  ==  heigth 
		assert err_buffer.shape[1]  ==  width
		assert obs.shape[0]  ==  heigth 
		assert obs.shape[1]  ==  width
		assert obs.shape[2]  ==  nb_colors 
		
		obs_ptr  =  <double*>obs.data
		err_buffer_ptr = <double*>err_buffer.data
		
		if err_buffer_ptr  ==  NULL:
			raise BaseException('err_buffer_ptr is NULL')
		if obs_ptr  ==  NULL:
			raise BaseException('obs_ptr is NULL')
	

	_differentiable_renderer.renderScene( scene_c,image_ptr, z_buffer_ptr, sigma, antialiase_error ,obs_ptr, err_buffer_ptr)
	
@cython.boundscheck(False)
@cython.wraparound(False)	
def renderSceneB(scene, 
		double sigma,
		np.ndarray[double,ndim = 3,mode = "c"] image, 
		np.ndarray[double,ndim = 2,mode = "c"] z_buffer,
		np.ndarray[double,ndim = 3,mode = "c"] image_b = None,
		bool antialiase_error  = 0,
		np.ndarray[double,ndim = 3,mode = "c"] obs = None,
		np.ndarray[double,ndim = 2,mode = "c"] err_buffer = None,
		np.ndarray[double,ndim = 2,mode = "c"] err_buffer_b = None,
		bool check_valid=1):

	cdef _differentiable_renderer.Scene scene_c 

	if check_valid:
		assert (not(image is None))
		assert (not(z_buffer is None))
		
	heigth = image.shape[0]
	width  = image.shape[1]
	nb_colors = image.shape[2]
	nb_triangles  =  scene.faces.shape[0]
	
	if check_valid:
		assert(nb_colors  ==  scene.colors.shape[1])			
		assert z_buffer.shape[0]  ==  heigth 
		assert z_buffer.shape[1]  ==  width 	
		assert(nb_triangles  ==  scene.faces_uv.shape[0])

	nb_vertices  =  scene.depths.shape[0]
	nb_vertices_uv  =  scene.uv.shape[0]
	
	if check_valid:
		assert(scene.faces.dtype == np.uint32)
		assert(np.all(scene.faces<nb_vertices))
		assert(np.all(scene.faces_uv<nb_vertices_uv))
				
		assert(scene.colors.ndim  ==  2)
		assert(scene.uv.ndim  ==  2)
		assert(scene.ij.ndim  ==  2)
		assert(scene.shade.ndim  ==  1)
		assert(scene.edgeflags.ndim  ==  2)
		assert(scene.textured.ndim  ==  1)
		assert(scene.shaded.ndim  ==  1)		
		assert(scene.uv.shape[1]  ==  2)	
		assert(scene.ij.shape[0]  ==  nb_vertices)
		assert(scene.ij.shape[1]  ==  2)
		assert(scene.shade.shape[0]  ==  nb_vertices)
		assert(scene.colors.shape[0]  ==  nb_vertices)
		assert(scene.colors.shape[1]  ==  nb_colors)
		assert(scene.edgeflags.shape[0]  ==  nb_triangles)
		assert(scene.edgeflags.shape[1]  ==  3)
		assert(scene.textured.shape[0]  ==  nb_triangles)
		assert(scene.shaded.shape[0]  ==  nb_triangles)
		assert(scene.background.ndim  ==  3)
		assert(scene.background.shape[0]  ==  heigth)
		assert(scene.background.shape[1]  ==  width)
		assert(scene.background.shape[2]  ==  nb_colors)
				
		assert(scene.uv_b.ndim  ==  2)
		assert(scene.ij_b.ndim  ==  2)
		assert(scene.shade_b.ndim  ==  1)
		assert(scene.edgeflags.ndim  ==  2)
		assert(scene.textured.ndim  ==  1)
		assert(scene.shaded.ndim  ==  1)	
		assert(scene.uv_b.shape[0]  ==  nb_vertices_uv)
		assert(scene.uv_b.shape[1]  ==  2)
		assert(scene.ij_b.shape[0]  ==  nb_vertices)
		assert(scene.ij_b.shape[1]  ==  2)
		assert(scene.shade_b.shape[0]  ==  nb_vertices)	
		assert(scene.colors_b.shape[0]  ==  nb_vertices)
		assert(scene.colors_b.shape[1]  ==  nb_colors)
		assert(scene.edgeflags.shape[0]  ==  nb_triangles)
		assert(scene.edgeflags.shape[1]  ==  3)
		assert(scene.textured.shape[0]  ==  nb_triangles)
		assert(scene.shaded.shape[0]  ==  nb_triangles)
		assert(scene.background.ndim  ==  3)
		assert(scene.background.shape[0]  ==  heigth)
		assert(scene.background.shape[1]  ==  width)
		assert(scene.background.shape[2]  ==  nb_colors)
			
		if scene.texture.size>0:
			assert(scene.texture.ndim  ==  3)
			assert(scene.texture_b.ndim  ==  3)	
			assert(scene.texture.shape[0]>0)
			assert(scene.texture.shape[1]>0)		
			assert(scene.texture.shape[0]  ==  scene.texture_b.shape[0])
			assert(scene.texture.shape[1]  ==  scene.texture_b.shape[1])
			assert(scene.texture.shape[2]  ==  nb_colors)	
			assert(scene.texture_b.shape[2]  ==  nb_colors)
			
	scene_c.nb_colors = nb_colors
	
	cdef np.ndarray[np.uint32_t, mode = "c"] faces_c  =  np.ascontiguousarray(scene.faces.flatten(), dtype = np.uint32)
	cdef np.ndarray[np.uint32_t, mode = "c"] faces_uv_c  =  np.ascontiguousarray(scene.faces_uv.flatten(), dtype = np.uint32)
	cdef np.ndarray[np.double_t, mode = "c"] depths_c =  np.ascontiguousarray(scene.depths.flatten(), dtype = np.double)	
	cdef np.ndarray[np.double_t, mode = "c"] uv_c =  np.ascontiguousarray(scene.uv.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] ij_c =  np.ascontiguousarray(scene.ij.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] uv_b_c =  np.ascontiguousarray(scene.uv_b.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] ij_b_c =  np.ascontiguousarray(scene.ij_b.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] shade_c =  np.ascontiguousarray(scene.shade.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] shade_b_c =  np.ascontiguousarray(scene.shade_b.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] colors_c =  np.ascontiguousarray(scene.colors.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] colors_b_c =  np.ascontiguousarray(scene.colors_b.flatten(), dtype = np.double)
	cdef np.ndarray[np.uint8_t, mode = "c"] edgeflags_c =  np.ascontiguousarray(scene.edgeflags.flatten(), dtype = np.uint8)
	cdef np.ndarray[np.uint8_t, mode = "c"] textured_c =  np.ascontiguousarray(scene.textured.flatten(), dtype = np.uint8)
	cdef np.ndarray[np.uint8_t, mode = "c"] shaded_c =  np.ascontiguousarray(scene.shaded.flatten(), dtype = np.uint8)
	cdef np.ndarray[np.double_t, mode = "c"] texture_c =  np.ascontiguousarray(scene.texture.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] texture_b_c =  np.ascontiguousarray(scene.texture_b.flatten(), dtype = np.double)
	cdef np.ndarray[np.double_t, mode = "c"] background_c =  np.ascontiguousarray(scene.background.flatten(), dtype = np.double)
	


	scene_c.height = <int> scene.height
	scene_c.width = <int> scene.width
	scene_c.nb_triangles = nb_triangles
	scene_c.nb_vertices = nb_vertices
	scene_c.backface_culling = scene.backface_culling
	scene_c.clockwise = scene.clockwise	
	scene_c.nb_uv = nb_vertices_uv
	scene_c.faces = <unsigned int*> faces_c.data
	scene_c.faces_uv = <unsigned int*> faces_uv_c.data	
	scene_c.depths = <double*> depths_c.data
	scene_c.uv = <double*> uv_c.data
	scene_c.uv_b = <double*> uv_b_c.data
	scene_c.ij = <double*> ij_c.data
	scene_c.ij_b = <double*> ij_b_c.data
	scene_c.shade = <double*> shade_c.data
	scene_c.shade_b = <double*> shade_b_c.data
	scene_c.colors = <double*> colors_c.data
	scene_c.colors_b = <double*> colors_b_c.data
	scene_c.edgeflags = <bool*> edgeflags_c.data
	scene_c.textured = <bool*> textured_c.data
	scene_c.shaded = <bool*> shaded_c.data
	scene_c.texture = <double*> texture_c.data
	scene_c.texture_b = <double*> texture_b_c.data
	scene_c.background = <double*> background_c.data
	scene_c.texture_height = scene.texture.shape[0]
	scene_c.texture_width = scene.texture.shape[1]
	scene_c.strict_edge = scene.strict_edge
	
	
	if scene_c.background  ==  NULL:
		raise BaseException('scene_c.background is NULL')

	cdef double* obs_ptr  =  NULL
	cdef double* err_buffer_ptr  =  NULL
	cdef double* err_buffer_b_ptr  =  NULL	
	cdef double* image_ptr  =  <double*> image.data
	
	cdef double* image_b_ptr  =  NULL
	cdef double* z_buffer_ptr  =  <double*> z_buffer.data
	
	if image_ptr  ==  NULL:
		raise BaseException('image_ptr is NULL')		
	if z_buffer_ptr  ==  NULL:
		raise BaseException('z_buffer_ptr is NULL')
	
	if antialiase_error:
		if check_valid:
			assert err_buffer.shape[0]  ==  heigth 
			assert err_buffer.shape[1]  ==  width 
			assert obs.shape[0]  ==  heigth 
			assert obs.shape[1]  ==  width 
	
		err_buffer_ptr = <double*>err_buffer.data
		err_buffer_b_ptr = <double*>err_buffer_b.data
		obs_ptr = <double*>obs.data
		
		if err_buffer_ptr  ==  NULL:
			raise BaseException('err_buffer_ptr is NULL')
		if err_buffer_b_ptr  ==  NULL:
			raise BaseException('err_buffer_b_ptr is NULL')
		if obs_ptr  ==  NULL:
			raise BaseException('obs_ptr is NULL')
	else:
		if check_valid:
			assert (not(image_b is None))
			assert image_b.shape[0]  ==  heigth 
			assert image_b.shape[1]  ==  width 
		image_b_ptr  =  <double*> image_b.data
		if image_b_ptr  ==  NULL:
			raise BaseException('image_b_ptr is NULL')
	
	_differentiable_renderer.renderScene_B( scene_c, image_ptr, z_buffer_ptr, image_b_ptr, sigma, antialiase_error ,obs_ptr, err_buffer_ptr, err_buffer_b_ptr)
	scene.uv_b = uv_b_c.reshape(scene.uv_b.shape)
	scene.ij_b = ij_b_c.reshape(scene.ij_b.shape)
	scene.shade_b = shade_b_c.reshape(scene.shade_b.shape)
	scene.colors_b = colors_b_c.reshape(scene.colors_b.shape)
	scene.texture_b = texture_b_c.reshape(scene.texture_b.shape)
	
