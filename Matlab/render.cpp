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
	double* Zbuffer;
	double *Abuffer;
	char* type;


	int SizeH;
	int SizeW;
	int SizeA;

	const mxArray * source;
	double *pr;
	int   number_of_dims;
	const mwSize  *dim_array;
	mwSize strlen;  
	const mxArray * matlabScene;
	Scene scene;
	double* Aobs;
	double* Errbuffer; 
    // loading type 
	int karg=0;
        matlabScene=prhs[karg];
	if (!mxIsStruct(matlabScene))
		error("type should a struct");

	    /* get input arguments */
    	int nfields = mxGetNumberOfFields(matlabScene);
        int NStructElems = mxGetNumberOfElements(matlabScene);
	if (!NStructElems>1)
		error("expect a single element struct");



 
	source=mxGetField(matlabScene,0,"faces");
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
	scene.nbTriangles = dim_array[1];
	scene.faces= (uint32_T *)mxGetData(source);
    
    
    scene.clockwise=true;
    
    source=mxGetField(matlabScene,0,"faces_uv");
    if (!source)
        error("missing field faces_uv");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.faces_uv is not well sized , should be of dimension 2 (3xNbTriangles)");
	if (dim_array[0]!=3)
		error("the input scene.faces_uv is not well sized should be of size 3xNbTriangles");
	if (dim_array[1]!=scene.nbTriangles)
        error("the input scene.faces_uv is not well sized should be of size 3xNbTriangles");
    if (!mxIsUint32(source))
        error("the input scene.faces_uv should be of type uint32");
	scene.faces_uv= (uint32_T *)mxGetData(source);
    
    	
	source=mxGetField(matlabScene,0,"depths");
    if (!source)
        error("missing field depths");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.depths is not well sized , should be of dimension 2 (1xNbVertices)");
	if (dim_array[0]!=1)
		error("the input scene.depths is not well sized should be of size 1xNbVertices");
    scene.nbVertices=dim_array[1];
	scene.depths=mxGetPr(source);
    
	source=mxGetField(matlabScene,0,"uv");
    if (!source)
        error("missing field uv");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.uv is not well sized , should be of dimension 3 (2xNbVertices)");
	if (dim_array[0]!=2)
		error("the input scene.uv is not well sized should be of size 2xNbVertices");
	scene.nbUV=dim_array[1];	
	scene.uv= mxGetPr(source);
    
	source=mxGetField(matlabScene,0,"ij");
    if (!source)
        error("missing field ij");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.ij is not well sized , should be of dimension 3 (2xNbVertices)");
	if (dim_array[0]!=2)
		error("the input scene.ij is not well sized should be of size 2xNbVertices");
	if (dim_array[1]!=scene.nbVertices)		
		error("the input scene.ij is not well sized should be of size 2xNbVertices");
	scene.ij= mxGetPr(source);

	source=mxGetField(matlabScene,0,"shade");
    if (!source)
        error("missing field shade");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.shade is not well sized , should be of dimension 2 (1xNbVertices)");
	if (dim_array[0]!=1)
		error("the input scene.shade is not well sized should be of size 1xNbVertices");
	if (dim_array[1]!=scene.nbVertices)		
		error("the input scene.shade is not well sized should be of size 1xNbVertices");
	scene.shade= mxGetPr(source);

	source=mxGetField(matlabScene,0,"colors");
    if (!source)
        error("missing field colors");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=2)
		error("the input scene.colors is not well sized , should be of dimension 2 (nbColors x NbVertices)");
	scene.nbColors=dim_array[0];		
	if (dim_array[1]!=scene.nbVertices)		
		error("the input scene.colors is not well sized should be of size nbColors x NbVertices");
	scene.colors= mxGetPr(source);

	source=mxGetField(matlabScene,0,"textured");
    if (!source)
        error("missing field textured");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");
	if (dim_array[0]!=1)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");	
    if (dim_array[1]!=scene.nbTriangles)
        error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");
	scene.textured= mxGetLogicals(source);

    source=mxGetField(matlabScene,0,"edgeflags");
    if (!source)
        error("missing field edgeflags");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xNbTriangles");
	if (dim_array[0]!=3)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xNbTriangles");	
	if (dim_array[1]!=scene.nbTriangles)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xNbTriangles");
	scene.edgeflags= mxGetLogicals(source);
    
	source=mxGetField(matlabScene,0,"shaded");
    if (!source)
        error("missing field shaded");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=2)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");
	if (dim_array[0]!=1)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");	
	if (dim_array[1]!=scene.nbTriangles)
		error("the input scene.textured is not well sized , should be of dimension  1xNbTriangles");	
	scene.shaded= mxGetLogicals(source);

	
	source=mxGetField(matlabScene,0,"background");
    if (!source)
        error("missing field background");	
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	pr  = mxGetPr(source);
	if (number_of_dims!=3)
		error("the input scene.background is not well sized , should be of dimension  nbColors x H x W");	
	if (dim_array[0]!=scene.nbColors)
		error("the input scene.background is not well sized , should be of dimension  nbColors x H x W ");
	scene.image_W=dim_array[1];
	scene.image_H=dim_array[2];	
	
	scene.background= mxGetPr(source);	

	source=mxGetField(matlabScene,0,"texture");
    if (!source)
        error("missing field texture");	
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	pr  = mxGetPr(source);
	if (number_of_dims!=3)
		error("the input scene.texture is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[0]!=scene.nbColors)
		error("the input scene.texture is not well sized , should be of dimension  H x W x nbColors");	
	scene.texture_W=dim_array[1];
	scene.texture_H=dim_array[2];
    scene.texture= mxGetPr(source);

	bool antialiaseError;
	double sigma = mxGetScalar(prhs[1]);

	if (nrhs>2)
	  antialiaseError = mxGetScalar(prhs[2]);
	else
	  antialiaseError=false;
	  
	  const mwSize dimsA[]         = {scene.nbColors,scene.image_W,scene.image_H};
	  plhs[0]                     = mxCreateNumericArray(3, dimsA, mxDOUBLE_CLASS, mxREAL);
	  Abuffer=mxGetPr(plhs[0]);
	  const mwSize dimsZ[]         = {scene.image_W,scene.image_H};
	  plhs[1]                     = mxCreateNumericArray(2, dimsZ, mxDOUBLE_CLASS, mxREAL);
	  Zbuffer=mxGetPr(plhs[1]);
	
	if (antialiaseError)
	{
	  source=prhs[3];	
	  number_of_dims = mxGetNumberOfDimensions(source);
	  dim_array = mxGetDimensions(source);
	  pr  = mxGetPr(source);
	  if (number_of_dims!=3)
		  error("the input Aobs is not well sized , should be of dimension  H x W x nbColors");	
	  if (dim_array[0]!=scene.nbColors)
		  error("the input Aobs is not well sized , should be of dimension  H x W x nbColors");
	  if (dim_array[1]!=scene.image_W)
		  error("the input Aobs is not well sized , should be of dimension  H x W x nbColors");
	  if (dim_array[2]!=scene.image_H)
		error("the input Aobs is not well sized , should be of dimension  H x W x nbColors");
	  Aobs= mxGetPr(source);
	  
	  const mwSize dimsE[]         = {scene.image_W,scene.image_H};
	  plhs[2]                     = mxCreateNumericArray(2, dimsZ, mxDOUBLE_CLASS, mxREAL);
	  Errbuffer=mxGetPr(plhs[2]);
	  
	  renderScene(scene,Abuffer,Zbuffer,sigma,antialiaseError,Aobs,Errbuffer);
	}
	else
	{
	  
	  renderScene(scene,Abuffer,Zbuffer,sigma,antialiaseError);
	}	
 }

