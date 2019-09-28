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
	double* Zvertex;
	double *Abuffer;
	double* Abuffer_b;
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
	bool antialiaseError;
	double* Aobs;
	double* ErrBuffer;
	double* ErrBuffer_b;

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


	
	source=mxGetField(matlabScene,0,"depths");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=3)
		error("the input scene.depths is not well sized , should be of dimension 3 (1x3xN)");
	if (dim_array[0]!=1)
		error("the input scene.depths is not well sized should be of size 1x3xN");
	if (dim_array[1]!=3)
		error("the input scene.depths is not well sized should be of size 1x3xN");
	scene.nbTriangles=dim_array[2];
	scene.depths=mxGetPr(source);
 
	
	source=mxGetField(matlabScene,0,"uv");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=3)
		error("the input scene.uv is not well sized , should be of dimension 3 (2x3xN)");
	if (dim_array[0]!=2)
		error("the input scene.uv is not well sized should be of size 2x3xN");
	if (dim_array[1]!=3)
		error("the input scene.uv is not well sized should be of size 2x3xN");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.uv is not well sized should be of size 2x3xN");
	scene.uv= mxGetPr(source);

	source=mxGetField(matlabScene,0,"ij");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=3)
		error("the input scene.ij is not well sized , should be of dimension 3 (2x3xN)");
	if (dim_array[0]!=2)
		error("the input scene.ij is not well sized should be of size 2x3xN");
	if (dim_array[1]!=3)
		error("the input scene.ij is not well sized should be of size 2x3xN");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.ij is not well sized should be of size 2x3xN");
	scene.ij= mxGetPr(source);

	source=mxGetField(matlabScene,0,"shade");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=3)
		error("the input scene.shade is not well sized , should be of dimension 3 (1x3xN)");
	if (dim_array[0]!=1)
		error("the input scene.shade is not well sized should be of size 1x3xN");
	if (dim_array[1]!=3)
		error("the input scene.shade is not well sized should be of size 1x3xN");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.shade is not well sized should be of size 1x3xN");
	scene.shade= mxGetPr(source);

	source=mxGetField(matlabScene,0,"colors");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=3)
		error("the input scene.colors is not well sized , should be of dimension 3 (nbColors x 3 x N)");
	scene.nbColors=dim_array[0];		
	if (dim_array[1]!=3)
		error("the input scene.colors is not well sized should be of size nbColors x 3 x N");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.colors is not well sized should be of size 2x3xN");
	scene.colors= mxGetPr(source);

	
	source=mxGetField(matlabScene,0,"edgeflags");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);

	if (number_of_dims!=2)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xN");
	if (dim_array[0]!=3)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xN");	
	if (dim_array[1]!=scene.nbTriangles)
		error("the input scene.edgeflag is not well sized , should be of dimension  3xN");
	scene.edgeflags= mxGetLogicals(source);

	source=mxGetField(matlabScene,0,"textured");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);

	if (number_of_dims!=2)
		error("the input scene.textured is not well sized , should be of dimension  1xN");
	if (dim_array[0]!=1)
		error("the input scene.textured is not well sized , should be of dimension  1xN");	
	if (dim_array[1]!=scene.nbTriangles)
		error("the input scene.textured is not well sized , should be of dimension  1xN");	
	scene.textured= mxGetLogicals(source);

	source=mxGetField(matlabScene,0,"shaded");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);

	if (number_of_dims!=2)
		error("the input scene.textured is not well sized , should be of dimension  1xN");
	if (dim_array[0]!=1)
		error("the input scene.textured is not well sized , should be of dimension  1xN");	
	if (dim_array[1]!=scene.nbTriangles)
		error("the input scene.textured is not well sized , should be of dimension  1xN");	
	scene.shaded= mxGetLogicals(source);

	
	source=mxGetField(matlabScene,0,"background");
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

	source=mxGetField(matlabScene,0,"material");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	pr  = mxGetPr(source);
	if (number_of_dims!=3)
		error("the input scene.material is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[0]!=scene.nbColors)
		error("the input scene.material is not well sized , should be of dimension  H x W x nbColors");	
	scene.texture_W=dim_array[1];
	scene.texture_H=dim_array[2];
        scene.texture= mxGetPr(source);


	source=mxGetField(matlabScene,0,"depths");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=3)
		error("the input scene.depths is not well sized , should be of dimension 3 (1x3xN)");
	if (dim_array[0]!=1)
		error("the input scene.depths is not well sized should be of size 1x3xN");
	if (dim_array[1]!=3)
		error("the input scene.depths is not well sized should be of size 1x3xN");
	scene.nbTriangles=dim_array[2];
	scene.depths=mxGetPr(source);
 
	
	source=mxGetField(matlabScene,0,"uv_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=3)
		error("the input scene.uv is not well sized , should be of dimension 3 (2x3xN)");
	if (dim_array[0]!=2)
		error("the input scene.uv is not well sized should be of size 2x3xN");
	if (dim_array[1]!=3)
		error("the input scene.uv is not well sized should be of size 2x3xN");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.uv is not well sized should be of size 2x3xN");
	scene.uv_b= mxGetPr(source);

	source=mxGetField(matlabScene,0,"ij_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);	
	if (number_of_dims!=3)
		error("the input scene.ij is not well sized , should be of dimension 3 (2x3xN)");
	if (dim_array[0]!=2)
		error("the input scene.ij is not well sized should be of size 2x3xN");
	if (dim_array[1]!=3)
		error("the input scene.ij is not well sized should be of size 2x3xN");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.ij is not well sized should be of size 2x3xN");
	scene.ij_b= mxGetPr(source);

	source=mxGetField(matlabScene,0,"shade_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	if (number_of_dims!=3)
		error("the input scene.shade is not well sized , should be of dimension 3 (1x3xN)");
	if (dim_array[0]!=1)
		error("the input scene.shade is not well sized should be of size 1x3xN");
	if (dim_array[1]!=3)
		error("the input scene.shade is not well sized should be of size 1x3xN");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.shade is not well sized should be of size 1x3xN");
	scene.shade_b= mxGetPr(source);

	source=mxGetField(matlabScene,0,"colors_b");
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=3)
		error("the input scene.colors is not well sized , should be of dimension 3 (nbColors x 3 x N)");
	scene.nbColors=dim_array[0];		
	if (dim_array[1]!=3)
		error("the input scene.colors is not well sized should be of size nbColors x 3 x N");
	if (dim_array[2]!=scene.nbTriangles)		
		error("the input scene.colors is not well sized should be of size 2x3xN");
	scene.colors_b= mxGetPr(source);


	source=prhs[1];
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=3)
		error("the input Abuffer is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[0]!=scene.nbColors)
		error("the input Abuffer is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[1]!=scene.image_W)
		error("the input Abuffer is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[2]!=scene.image_H)
		error("the input Abuffer is not well sized , should be of dimension  H x W x nbColors");
        Abuffer= mxGetPr(source);

	source=prhs[2];
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=2)
		error("a the input Zbuffer is not well sized , should be of dimension  H x W");		
	if (dim_array[0]!=scene.image_W)
		error("b the input Zbuffer is not well sized , should be of dimension  H x W");	
	if (dim_array[1]!=scene.image_H)
		error("c the input Zbuffer is not well sized , should be of dimension  H x W");
        Zbuffer= mxGetPr(source);

	double sigma = mxGetScalar(prhs[4]);
	
	 if (nrhs>5)
	  antialiaseError = mxGetScalar(prhs[5]);
	else
	  antialiaseError=false;
	  
	 
	
	if (antialiaseError)
	{
	   source=prhs[6];	
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
	  
	  source=prhs[7];	
	  number_of_dims = mxGetNumberOfDimensions(source);
	  dim_array = mxGetDimensions(source);
	  pr  = mxGetPr(source);
	  if (number_of_dims!=2)
		  error("the input error is not well sized , should be of dimension  H x W ");	
	  if (dim_array[0]!=scene.image_W)
		  error("the input error is not well sized , should be of dimension  H x W");
	  if (dim_array[1]!=scene.image_H)
		  error("the input error is not well sized , should be of dimension  H x W ");
	  ErrBuffer= mxGetPr(source);
	  
	  source=prhs[8];	
	  number_of_dims = mxGetNumberOfDimensions(source);
	  dim_array = mxGetDimensions(source);
	  pr  = mxGetPr(source);
	  if (number_of_dims!=2)
		  error("the input error is not well sized , should be of dimension  H x W ");	
	  if (dim_array[0]!=scene.image_W)
		  error("the input error is not well sized , should be of dimension  H x W");
	  if (dim_array[1]!=scene.image_H)
		  error("the input error is not well sized , should be of dimension  H x W ");
	  ErrBuffer_b= mxGetPr(source);
	 
	  renderScene_B(scene,Abuffer,Zbuffer,Abuffer_b,sigma,antialiaseError,Aobs,ErrBuffer,ErrBuffer_b);
	}
      
    else
    {
      	
	 source=prhs[3];
	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	
	if (number_of_dims!=3)
		error("the input Abuffer_b is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[0]!=scene.nbColors)
		error("the input Abuffer_b is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[1]!=scene.image_W)
		error("the input Abuffer_b is not well sized , should be of dimension  H x W x nbColors");	
	if (dim_array[2]!=scene.image_H)
		error("the input Abuffer_b is not well sized , should be of dimension  H x W x nbColors");
        Abuffer_b= mxGetPr(source);
	
	renderScene_B(scene,Abuffer,Zbuffer,Abuffer_b,sigma); 
    }

 }

