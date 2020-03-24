#include <mex.h>
#include "array2D.h"
#include "../C++/DifferentiableRenderer.h"

void error(const char* msg)
{
	mexErrMsgTxt(msg);
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{



	const char **fnames;       /* pointers to field names */
	double* z_buffer;
	double* Zvertex;
	double *image;
	double* image_b;
	char* type;


	int height;
	int width;
	int SizeA;

	const mxArray * source;
	double *pr;
	int   number_of_dims;
	const mwSize  *dim_array;
	mwSize strlen;  
	const mxArray * matlab_scene;
	Scene scene;
	bool antialiaseError;
	double* obs;
	double* err_buffer;
	double* err_buffer_b;

    // loading type 
	int karg=0;
        matlab_scene=prhs[karg];
	if (!mxIsStruct(matlab_scene))
		error("type should a struct");

	    /* get input arguments */
    	int nfields = mxGetNumberOfFields(matlab_scene);
        int NStructElems = mxGetNumberOfElements(matlab_scene);
	if (!NStructElems>1)
		error("expect a single element struct");



 
	source=mxGetField(matlab_scene,0,"faces");
    if (!source)
        error("missing field faces");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.faces is not well sized , should be of dimension 3 (3xNbTriangles)");
	if (dim_array[0]!=3)
		error("the input scene.faces is not well sized should be of size 3xNbTriangles");
    if (!mxIsUint32(source))
        error("the input scene.faces should be of type uint32");
	scene.nb_triangles = dim_array[1];
	scene.faces= (uint32_T *)mxGetData(source);
    
    
    source=mxGetField(matlab_scene,0,"faces_uv");
    if (!source)
        error("missing field faces_uv"); 
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.faces_uv is not well sized , should be of dimension 2 (3xNbTriangles)");
	if (dim_array[0]!=3)
		error("the input scene.faces_uv is not well sized should be of size 3xNbTriangles");
	if (dim_array[1]!=scene.nb_triangles)
        error("the input scene.faces_uv is not well sized should be of size 3xNbTriangles");
    if (!mxIsUint32(source))
        error("the input scene.faces_uv should be of type uint32");
	scene.faces_uv= (uint32_T *)mxGetData(source);
    
    	
	source=mxGetField(matlab_scene,0,"depths");
    if (!source)
        error("missing field depths"); 
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.depths is not well sized , should be of dimension 2 (1xNb_vertices)");
	if (dim_array[0]!=1)
		error("the input scene.depths is not well sized should be of size 1xNb_vertices");
    scene.nb_vertices=dim_array[1];
	scene.depths=mxGetPr(source);
    
	source=mxGetField(matlab_scene,0,"uv");
    if (!source)
        error("missing field uv"); 
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.uv is not well sized , should be of dimension 3 (2xNb_vertices)");
	if (dim_array[0]!=2)
		error("the input scene.uv is not well sized should be of size 2xNb_vertices");
	scene.nb_uv=dim_array[1];	
	scene.uv= mxGetPr(source);
    
	source=mxGetField(matlab_scene,0,"ij");
    if (!source)
        error("missing field ij");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.ij is not well sized , should be of dimension 3 (2xNb_vertices)");
	if (dim_array[0]!=2)
		error("the input scene.ij is not well sized should be of size 2xNb_vertices");
	if (dim_array[1]!=scene.nb_vertices)		
		error("the input scene.ij is not well sized should be of size 2xNb_vertices");
	scene.ij= mxGetPr(source);

	source=mxGetField(matlab_scene,0,"shade");
    if (!source)
        error("missing field shade");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.shade is not well sized , should be of dimension 2 (1xNb_vertices)");
	if (dim_array[0]!=1)
		error("the input scene.shade is not well sized should be of size 1xNb_vertices");
	if (dim_array[1]!=scene.nb_vertices)		
		error("the input scene.shade is not well sized should be of size 1xNb_vertices");
	scene.shade= mxGetPr(source);

	source=mxGetField(matlab_scene,0,"colors");
    if (!source)
        error("missing field colors");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=2)
		error("the input scene.colors is not well sized , should be of dimension 2 (nb_colors x Nb_vertices)");
	scene.nb_colors=dim_array[0];		
	if (dim_array[1]!=scene.nb_vertices)		
		error("the input scene.colors is not well sized should be of size nb_colors x Nb_vertices");
	scene.colors= mxGetPr(source);

	source=mxGetField(matlab_scene,0,"textured");
    if (!source)
        error("missing field textured");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);

	if (number_of_dims!=2)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");
	if (dim_array[0]!=1)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");	
    if (dim_array[1]!=scene.nb_triangles)
        error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");
	scene.textured= mxGetLogicals(source);

    source=mxGetField(matlab_scene,0,"edgeflags");
    if (!source)
        error("missing field edgeflags");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xNbTriangles");
	if (dim_array[0]!=3)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xNbTriangles");	
	if (dim_array[1]!=scene.nb_triangles)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xNbTriangles");
	scene.edgeflags= mxGetLogicals(source);
    
	source=mxGetField(matlab_scene,0,"shaded");
    if (!source)
        error("missing field shaded");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");
	if (dim_array[0]!=1)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");	
	if (dim_array[1]!=scene.nb_triangles)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");	
	scene.shaded= mxGetLogicals(source);

	
	source=mxGetField(matlab_scene,0,"background");
    if (!source)
        error("missing field background");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	pr  = mxGetPr(source);
	if (number_of_dims!=3)
		error("the input scene.background is not well sized , should be of dimension  nb_colors x H x W");	
	if (dim_array[0]!=scene.nb_colors)
		error("the input scene.background is not well sized , should be of dimension  nb_colors x H x W ");
	scene.width=dim_array[1];
	scene.height=dim_array[2];	
	
	scene.background= mxGetPr(source);	

	source=mxGetField(matlab_scene,0,"texture");
    if (!source)
        error("missing field texture");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	pr  = mxGetPr(source);
	if (number_of_dims!=3)
		error("the input scene.texture is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[0]!=scene.nb_colors)
		error("the input scene.texture is not well sized , should be of dimension  H x W x nb_colors");	
	scene.texture_width=dim_array[1];
	scene.texture_height=dim_array[2];
    scene.texture= mxGetPr(source);
    
    source=mxGetField(matlab_scene,0,"texture_b");
    if (!source)
        error("missing field texture_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	pr  = mxGetPr(source);
	if (number_of_dims!=3)
		error("the input scene.texture_b is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[0]!=scene.nb_colors)
		error("the input scene.texture_b is not well sized , should be of dimension  H x W x nb_colors");
    if (dim_array[1]!=scene.texture_width)
        error("the input scene.texture_b is not well sized , should be of dimension  H x W x nb_colors");
    if (dim_array[2]!=scene.texture_height)
        error("the input scene.texture_b is not well sized , should be of dimension  H x W x nb_colors");
    scene.texture_b= mxGetPr(source);


	source=mxGetField(matlab_scene,0,"uv_b");
    if (!source)
        error("missing field uv_b");
    number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.uv_b is not well sized , should be of dimension 2 (2xNb_vertices)");
	if (dim_array[0]!=2)
		error("the input scene.uv_b is not well sized should be of size 2xNb_vertices");
	if (scene.nb_uv!=dim_array[1])
		error("the input scene.uv_b is not well sized should be of size 2xNb_vertices");
	scene.uv_b= mxGetPr(source);

	source=mxGetField(matlab_scene,0,"ij_b");
    if (!source)
        error("missing field ij_b");
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.ij_b is not well sized , should be of dimension 2 (2xNb_vertices)");
	if (dim_array[0]!=2)
		error("the input scene.ij_b is not well sized should be of size 2xNb_vertices");
	if (dim_array[1]!=scene.nb_vertices)		
		error("the input scene.ij_b is not well sized should be of size 2xNb_vertices");
	scene.ij_b= mxGetPr(source);

	source=mxGetField(matlab_scene,0,"shade_b");
    if (!source)
        error("missing field shade_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.shade_b is not well sized , should be of dimension 2 (1xNb_vertices)");
	if (dim_array[0]!=1)
		error("the input scene.shade_b is not well sized should be of size 1xNb_vertices");
	if (dim_array[1]!=scene.nb_vertices)		
		error("the input scene.shade_b is not well sized should be of size 1xNb_vertices");
	scene.shade_b= mxGetPr(source);

	source=mxGetField(matlab_scene,0,"colors_b");
    if (!source)
        error("missing field colors_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.colors_b is not well sized , should be of dimension 2 (nb_colors x Nb_vertices)");
	scene.nb_colors=dim_array[0];		
	if (dim_array[1]!=scene.nb_vertices)		
		error("the input scene.colors_b is not well sized should be of size nb_colors x Nb_vertices");
	scene.colors_b= mxGetPr(source);


	source=prhs[1];
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=3)
		error("the input image is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[0]!=scene.nb_colors)
		error("the input image is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[1]!=scene.width)
		error("the input image is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[2]!=scene.height)
		error("the input image is not well sized , should be of dimension  H x W x nb_colors");
        image= mxGetPr(source);

	source=prhs[2];
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=2)
		error("a the input z_buffer is not well sized , should be of dimension  H x W");		
	if (dim_array[0]!=scene.width)
		error("b the input z_buffer is not well sized , should be of dimension  H x W");	
	if (dim_array[1]!=scene.height)
		error("c the input z_buffer is not well sized , should be of dimension  H x W");
        z_buffer= mxGetPr(source);

	double sigma = mxGetScalar(prhs[4]);
	
	 if (nrhs>5)
	  antialiaseError = mxGetScalar(prhs[5]);
	else
	  antialiaseError=false;
	  
	scene.clockwise=true;
	scene.backface_culling=true;
    
	
	if (antialiaseError)
	{
	  source=prhs[6];	
	  number_of_dims = mxGetNumberOfDimensions(source);
	  dim_array = mxGetDimensions(source);
	  pr  = mxGetPr(source);
	  if (number_of_dims!=3)
		  error("the input obs is not well sized , should be of dimension  H x W x nb_colors");	
	  if (dim_array[0]!=scene.nb_colors)
		  error("the input obs is not well sized , should be of dimension  H x W x nb_colors");
	  if (dim_array[1]!=scene.width)
		  error("the input obs is not well sized , should be of dimension  H x W x nb_colors");
	  if (dim_array[2]!=scene.height)
		error("the input obs is not well sized , should be of dimension  H x W x nb_colors");
	  obs= mxGetPr(source);
	  
	  source=prhs[7];	
	  number_of_dims = mxGetNumberOfDimensions(source);
	  dim_array = mxGetDimensions(source);
	  pr  = mxGetPr(source);
	  if (number_of_dims!=2)
		  error("the input error is not well sized , should be of dimension  H x W ");	
	  if (dim_array[0]!=scene.width)
		  error("the input error is not well sized , should be of dimension  H x W");
	  if (dim_array[1]!=scene.height)
		  error("the input error is not well sized , should be of dimension  H x W ");
	  err_buffer= mxGetPr(source);
	  
	  source=prhs[8];	
	  number_of_dims = mxGetNumberOfDimensions(source);
	  dim_array = mxGetDimensions(source);
	  pr  = mxGetPr(source);
	  if (number_of_dims!=2)
		  error("the input error is not well sized , should be of dimension  H x W ");	
	  if (dim_array[0]!=scene.width)
		  error("the input error is not well sized , should be of dimension  H x W");
	  if (dim_array[1]!=scene.height)
		  error("the input error is not well sized , should be of dimension  H x W ");
	  err_buffer_b= mxGetPr(source);
	 
	  renderScene_B(scene,image,z_buffer,image_b,sigma,antialiaseError,obs,err_buffer,err_buffer_b);
	}
      
    else
    {
      	
	source=prhs[3];
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=3)
		error("the input image_b is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[0]!=scene.nb_colors)
		error("the input image_b is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[1]!=scene.width)
		error("the input image_b is not well sized , should be of dimension  H x W x nb_colors");	
	if (dim_array[2]!=scene.height)
		error("the input image_b is not well sized , should be of dimension  H x W x nb_colors");
        image_b= mxGetPr(source);
	
	renderScene_B(scene,image,z_buffer,image_b,sigma); 
    }

 }

