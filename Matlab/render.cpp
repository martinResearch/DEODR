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
	double *image;
	char* type;


	int height;
	int width;


	const mxArray * source;
	double *pr;
	int   number_of_dims;
	const mwSize  *dim_array;
	mwSize strlen;  
	const mxArray * matlab_scene;
	Scene scene;
	double* obs;
	double* err_buffer; 
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
    
    
    scene.clockwise=true;
	scene.backface_culling=true;
    
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

	bool antialiaseError;
	double sigma = mxGetScalar(prhs[1]);

	if (nrhs>2)
	  antialiaseError = mxGetScalar(prhs[2]);
	else
	  antialiaseError=false;
	  
	  const mwSize dims_image[]         = {scene.nb_colors,scene.width,scene.height};
	  plhs[0]                     = mxCreateNumericArray(3, dims_image, mxDOUBLE_CLASS, mxREAL);
	  image=mxGetPr(plhs[0]);
	  const mwSize dims_z_buffer[]         = {scene.width,scene.height};
	  plhs[1]                     = mxCreateNumericArray(2, dims_z_buffer, mxDOUBLE_CLASS, mxREAL);
	  z_buffer=mxGetPr(plhs[1]);
	
	if (antialiaseError)
	{
	  source=prhs[3];	
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
	  
	  const mwSize dimsE[]         = {scene.width,scene.height};
	  plhs[2]                     = mxCreateNumericArray(2, dims_z_buffer, mxDOUBLE_CLASS, mxREAL);
	  err_buffer=mxGetPr(plhs[2]);
	  
	  renderScene(scene,image,z_buffer,sigma,antialiaseError,obs,err_buffer);
	}
	else
	{
	  
	  renderScene(scene,image,z_buffer,sigma,antialiaseError);
	}	
 }

