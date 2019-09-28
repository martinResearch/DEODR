#ifndef _array2D_h_
#define _array2D_h_



template <class T> class array2D
{

private:
	T* data;
	int nRows;
	int nCols;
	int nElms;

public :
	T* get_data(){return data;}
	int get_nRows(){return nRows;};
	int get_nCols(){return nCols;};

	array2D(void) {data=NULL;nRows=0;nCols=0;nElms=0;}
	array2D(int nR,int nC) {nRows=nR;nCols=nC;nElms=nR*nC;if (nElms>0) {data=new T[nElms];} else {data=NULL;}}
	void alloc(int nR,int nC)   
	{ if ((nRows>0)&&(nR!=nRows)){mexErrMsgTxt("nRows different from the previous declared nRows");}
	  if ((nCols>0)&&(nC!=nCols)){mexErrMsgTxt("nCols different from the previous declared nCols");}
	  if (nElms==0)
	  {nRows=nR;nCols=nC;nElms=nR*nC;if (nElms>0) {data=new T[nElms];}
	  }

	}
	void free()   
	{ 
			delete [] data;
	}
	~array2D(void) {delete [] data;}
	//inline T  operator() (const int rowIndx,const int columnIndx) const {return data[rowIndx*nCols+columnIndx];}
	//array2D(int nRows,int nColumns,T* data) {this.data=data;this.nRows=nRows;this.nCols=nCols;}

	#ifdef CHECK_LIMITS_ARRAY2D
	inline T& operator() (const int rowIndx,const int columnIndx) {
			if ( rowIndx < 0  || nRows <= rowIndx || columnIndx < 0 || nCols <= columnIndx )
			{
			throw 0;
			}
			return data[rowIndx*nCols+columnIndx];
	}
	#else
	inline T& operator() (const int rowIndx,const int columnIndx) {return data[rowIndx*nCols+columnIndx];}
	#endif
	inline T& operator [] (const int num){return data[num];}

};

//template <class T> void createCopyArray2DFromMatlab(const mxArray * source,  array2D<T> &target);


template <class T> void createCopyArray2DFromMatlab(const mxArray * source,  array2D<T> &target,int flag)
	{
	int number_of_elements,number_of_dims;
	double *pr;

	const mwSize  *dim_array;         
	

	number_of_dims = mxGetNumberOfDimensions(source);
	dim_array = mxGetDimensions(source);
	number_of_elements = mxGetNumberOfElements(source);
	pr = mxGetPr(source);

	if (number_of_dims>2)	{mexErrMsgTxt("the array should be of dimensions 2");}
	
	if (flag==0) //array store row by row , so values need beeing reodered
	{target.alloc(dim_array[0],dim_array[1]);
		if (target.get_data()==NULL)
		mexErrMsgTxt("not able to allocate memory\n");			
		for(int c=0;c< target.get_nCols();c++) 
		for(int r=0;r< target.get_nRows();r++) 
			target(r,c)=(T) *(pr++);
	}else //we swap dimensions , such that values are not reodered
	{ 
		target.alloc(dim_array[1],dim_array[0]);
		if (target.get_data()==NULL)
		mexErrMsgTxt("not able to allocate memory\n");
		T* pr2;
		pr2= target.get_data();
		for(int r=0;r< target.get_nRows();r++) 
		for(int c=0;c< target.get_nCols();c++) 
		 *pr2++=(T) *(pr++); 
	
	
	
	
	}


	}

#endif