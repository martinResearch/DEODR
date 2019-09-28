# distutils: language=c++
from libcpp cimport bool
cimport _differentiablerenderer 

import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np



@cython.boundscheck(False)
@cython.wraparound(False)
def renderScene(scene, 
		double sigma,
		np.ndarray[double,ndim=3,mode="c"] Abuffer, 
		np.ndarray[double,ndim=2,mode="c"] Zbuffer,
		bool antialiaseError =0,
		np.ndarray[double,ndim=3,mode="c"] Aobs=None,
		np.ndarray[double,ndim=2,mode="c"] ErrBuffer=None):
    
	cdef _differentiablerenderer.Scene scene_c 	
	assert (not(Abuffer is None))
	assert (not(Zbuffer is None))
	heigth=Abuffer.shape[0]
	width =Abuffer.shape[1]
	nbColors=Abuffer.shape[2]
	
		
		
	assert(nbColors==scene.colors.shape[2])
	
	
	assert Zbuffer.shape[0]==heigth 
	assert Zbuffer.shape[1]==width 



	
	scene_c.nbColors=nbColors
	cdef np.ndarray[np.double_t, mode="c"] depths_c= np.ascontiguousarray(scene.depths.flatten(), dtype=np.double)	
	cdef np.ndarray[np.double_t, mode="c"] uv_c= np.ascontiguousarray(scene.uv.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] ij_c= np.ascontiguousarray(scene.ij.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] shade_c= np.ascontiguousarray(scene.shade.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] colors_c= np.ascontiguousarray(scene.colors.flatten(), dtype=np.double)
	cdef np.ndarray[np.uint8_t, mode="c"] edgeflags_c= np.ascontiguousarray(scene.edgeflags.flatten(), dtype=np.uint8)
	cdef np.ndarray[np.uint8_t, mode="c"] textured_c= np.ascontiguousarray(scene.textured.flatten(), dtype=np.uint8)
	cdef np.ndarray[np.uint8_t, mode="c"] shaded_c= np.ascontiguousarray(scene.shaded.flatten(), dtype=np.uint8)
	cdef np.ndarray[np.double_t, mode="c"] texture_c= np.ascontiguousarray(scene.texture.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] background_c= np.ascontiguousarray(scene.background.flatten(), dtype=np.double)
	
	nbTriangles=scene.depths.shape[0]
	assert(scene.uv.shape[0]==nbTriangles)
	assert(scene.uv.shape[0]==nbTriangles)
	assert(scene.ij.shape[0]==nbTriangles)
	assert(scene.shade.shape[0]==nbTriangles)
	assert(scene.colors.shape[0]==nbTriangles)
	assert(scene.edgeflags.shape[0]==nbTriangles)
	assert(scene.textured.shape[0]==nbTriangles)
	assert(scene.shaded.shape[0]==nbTriangles)


	scene_c.image_H=<int> scene.image_H
	scene_c.image_W=<int> scene.image_W
	scene_c.nbTriangles=nbTriangles
	scene_c.depths=<double*> depths_c.data
	scene_c.uv=<double*> uv_c.data
	scene_c.ij=<double*> ij_c.data
	scene_c.shade=<double*> shade_c.data
	scene_c.colors=<double*> colors_c.data
	scene_c.edgeflags=<bool*> edgeflags_c.data
	scene_c.textured=<bool*> textured_c.data
	scene_c.shaded=<bool*> shaded_c.data
	scene_c.texture=<double*> texture_c.data
	scene_c.background=<double*> background_c.data
	scene_c.texture_H=scene.texture.shape[0]
	scene_c.texture_W=scene.texture.shape[1]
	


	cdef double* Aobs_ptr=NULL
	cdef double* ErrBuffer_ptr=NULL
	
	cdef double* Abuffer_ptr= <double*> Abuffer.data
	cdef double* Zbuffer_ptr= <double*> Zbuffer.data
	
	if Abuffer_ptr==NULL:
		raise BaseException('Abuffer_ptr is NULL')
	if Zbuffer_ptr==NULL:
		raise BaseException('Zbuffer_ptr is NULL')

	
	if antialiaseError:
		assert ErrBuffer.shape[0]==heigth 
		assert ErrBuffer.shape[1]==width
		assert Aobs.shape[0]==heigth 
		assert Aobs.shape[1]==width
		assert Aobs.shape[2]==nbColors 
		
		Aobs_ptr = <double*>Aobs.data
		ErrBuffer_ptr=<double*>ErrBuffer.data
		
		if ErrBuffer_ptr==NULL:
			raise BaseException('ErrBuffer_ptr is NULL')
		if Aobs_ptr==NULL:
			raise BaseException('Aobs_ptr is NULL')
	

	_differentiablerenderer.renderScene( scene_c,Abuffer_ptr, Zbuffer_ptr, sigma, antialiaseError ,Aobs_ptr, ErrBuffer_ptr)
	
@cython.boundscheck(False)
@cython.wraparound(False)	
def renderSceneB(scene, 
		double sigma,
		np.ndarray[double,ndim=3,mode="c"] Abuffer, 
		np.ndarray[double,ndim=2,mode="c"] Zbuffer,
		np.ndarray[double,ndim=3,mode="c"] Abuffer_b=None,
		bool antialiaseError =0,
		np.ndarray[double,ndim=3,mode="c"] Aobs=None,
		np.ndarray[double,ndim=2,mode="c"] ErrBuffer=None,
		np.ndarray[double,ndim=2,mode="c"] ErrBuffer_b=None):

	cdef _differentiablerenderer.Scene scene_c 
	
	assert (not(Abuffer is None))
	assert (not(Zbuffer is None))
	
	
	heigth=Abuffer.shape[0]
	width =Abuffer.shape[1]
	nbColors=Abuffer.shape[2]
	

	assert(nbColors==scene.colors.shape[2])
	
	
	assert Zbuffer.shape[0]==heigth 
	assert Zbuffer.shape[1]==width 



	
	scene_c.nbColors=nbColors
	cdef np.ndarray[np.double_t, mode="c"] depths_c= np.ascontiguousarray(scene.depths.flatten(), dtype=np.double)	
	cdef np.ndarray[np.double_t, mode="c"] uv_c= np.ascontiguousarray(scene.uv.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] ij_c= np.ascontiguousarray(scene.ij.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] uv_b_c= np.ascontiguousarray(scene.uv_b.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] ij_b_c= np.ascontiguousarray(scene.ij_b.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] shade_c= np.ascontiguousarray(scene.shade.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] shade_b_c= np.ascontiguousarray(scene.shade_b.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] colors_c= np.ascontiguousarray(scene.colors.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] colors_b_c= np.ascontiguousarray(scene.colors_b.flatten(), dtype=np.double)
	cdef np.ndarray[np.uint8_t, mode="c"] edgeflags_c= np.ascontiguousarray(scene.edgeflags.flatten(), dtype=np.uint8)
	cdef np.ndarray[np.uint8_t, mode="c"] textured_c= np.ascontiguousarray(scene.textured.flatten(), dtype=np.uint8)
	cdef np.ndarray[np.uint8_t, mode="c"] shaded_c= np.ascontiguousarray(scene.shaded.flatten(), dtype=np.uint8)
	cdef np.ndarray[np.double_t, mode="c"] texture_c= np.ascontiguousarray(scene.texture.flatten(), dtype=np.double)
	cdef np.ndarray[np.double_t, mode="c"] background_c= np.ascontiguousarray(scene.background.flatten(), dtype=np.double)
	
	nbTriangles=scene.depths.shape[0]
	assert(scene.uv.shape[0]==nbTriangles)
	assert(scene.uv.shape[0]==nbTriangles)
	assert(scene.ij.shape[0]==nbTriangles)
	assert(scene.ij_b.shape[0]==nbTriangles)
	assert(scene.uv_b.shape[0]==nbTriangles)
	
	assert(scene.shade.shape[0]==nbTriangles)
	assert(scene.shade_b.shape[0]==nbTriangles)
	
	assert(scene.colors.shape[0]==nbTriangles)
	assert(scene.colors_b.shape[0]==nbTriangles)
	assert(scene.edgeflags.shape[0]==nbTriangles)
	assert(scene.textured.shape[0]==nbTriangles)
	assert(scene.shaded.shape[0]==nbTriangles)


	scene_c.image_H=<int> scene.image_H
	scene_c.image_W=<int> scene.image_W
	scene_c.nbTriangles=nbTriangles
	scene_c.depths=<double*> depths_c.data
	scene_c.uv=<double*> uv_c.data
	scene_c.uv_b=<double*> uv_b_c.data
	scene_c.ij=<double*> ij_c.data
	scene_c.ij_b=<double*> ij_b_c.data
	scene_c.shade=<double*> shade_c.data
	scene_c.shade_b=<double*> shade_b_c.data
	scene_c.colors=<double*> colors_c.data
	scene_c.colors_b=<double*> colors_b_c.data
	scene_c.edgeflags=<bool*> edgeflags_c.data
	scene_c.textured=<bool*> textured_c.data
	scene_c.shaded=<bool*> shaded_c.data
	scene_c.texture=<double*> texture_c.data
	scene_c.background=<double*> background_c.data
	scene_c.texture_H=scene.texture.shape[0]
	scene_c.texture_W=scene.texture.shape[1]
	


	cdef double* Aobs_ptr = NULL
	cdef double* ErrBuffer_ptr = NULL
	cdef double* ErrBuffer_b_ptr = NULL	
	cdef double* Abuffer_ptr = <double*> Abuffer.data
	
	cdef double* Abuffer_b_ptr = NULL
	cdef double* ZBuffer_ptr = <double*> Zbuffer.data
	
	if Abuffer_ptr==NULL:
		raise BaseException('Abuffer_ptr is NULL')		
	if ZBuffer_ptr==NULL:
		raise BaseException('ZBuffer_ptr is NULL')
	
	if antialiaseError:
		assert ErrBuffer.shape[0]==heigth 
		assert ErrBuffer.shape[1]==width 
		assert Aobs.shape[0]==heigth 
		assert Aobs.shape[1]==width 
	
		ErrBuffer_ptr=<double*>ErrBuffer.data
		ErrBuffer_b_ptr=<double*>ErrBuffer_b.data
		Aobs_ptr=<double*>Aobs.data
		
		if ErrBuffer_ptr==NULL:
			raise BaseException('ErrBuffer_ptr is NULL')
		if ErrBuffer_b_ptr==NULL:
			raise BaseException('ErrBuffer_b_ptr is NULL')
		if Aobs_ptr==NULL:
			raise BaseException('Aobs_ptr is NULL')
	else:
		assert (not(Abuffer_b is None))
		assert Abuffer_b.shape[0]==heigth 
		assert Abuffer_b.shape[1]==width 
		Abuffer_b_ptr = <double*> Abuffer_b.data
		if Abuffer_b_ptr==NULL:
			raise BaseException('Abuffer_b_ptr is NULL')
	
	_differentiablerenderer.renderScene_B( scene_c, Abuffer_ptr, ZBuffer_ptr, Abuffer_b_ptr, sigma, antialiaseError ,Aobs_ptr, ErrBuffer_ptr, ErrBuffer_b_ptr)
	scene.uv_b=uv_b_c.reshape(scene.uv_b.shape)
	scene.ij_b=ij_b_c.reshape(scene.ij_b.shape)
	scene.shade_b=shade_b_c.reshape(scene.shade_b.shape)
	scene.colors_b=colors_b_c.reshape(scene.colors_b.shape)
	
