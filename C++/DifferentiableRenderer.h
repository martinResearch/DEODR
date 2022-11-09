/* License FreeBSD:

	Copyright (c) 2016  Martin de La Gorce
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, this
	   list of conditions and the following disclaimer.
	2. Redistributions in binary form must reproduce the above copyright notice,
	   this list of conditions and the following disclaimer in the documentation
	   and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	The views and conclusions contained in the software and documentation are those
	of the authors and should not be interpreted as representing official policies,
	either expressed or implied, of the FreeBSD Project.

*/
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <limits>
#include <vector>
#include <algorithm>

//#define CHECK_LIMITS_ARRAY2D

using namespace std;

#define SWAP(_a_, _b_, _c_) \
	{                       \
		_c_ = _a_;          \
		_a_ = _b_;          \
		_b_ = _c_;          \
	}

void get_edge_xrange_from_ineq(double ineq[12], int width, int y, int &x_begin, int &x_end);
inline void render_part_interpolated(double *image, double *z_buffer, int y_begin, int x_min, int x_max, int y_end, bool strict_edge, double *xy1_to_A, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, bool perspective_correct);
inline void render_part_interpolated_B(double *image, double *image_B, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_A, double *xy1_to_A_B, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, bool perspective_correct);
inline void render_part_textured_gouraud(double *image, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_UV, double *xy1_to_L, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, double *Texture, int *Texture_size, bool perspective_correct);
inline void render_part_textured_gouraud_B(double *image, double *image_B, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_UV, double *xy1_to_UV_B, double *xy1_to_L, double *xy1_to_L_B, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, double *Texture, double *Texture_B, int *Texture_size, bool perspective_correct);

struct Scene
{
	unsigned int *faces;
	unsigned int *faces_uv;
	double *depths;
	double *uv;
	double *ij;
	double *shade;
	double *colors;
	bool *edgeflags;
	bool *textured;
	bool *shaded;
	int nb_triangles;
	int nb_vertices;
	bool clockwise;
	bool backface_culling;
	int nb_uv;
	int height;
	int width;
	int nb_colors;
	double *texture;
	int texture_height;
	int texture_width;
	double *background_image;
	double *background_color;
	// fields to store adjoint
	double *uv_b;
	double *ij_b;
	double *shade_b;
	double *colors_b;
	double *texture_b;
	bool strict_edge;
	bool perspective_correct;
	bool integer_pixel_centers;
};

void inv_matrix_3x3(double *S, double *T)
{
	//	S=	|S[0] S[1] S[2]|
	//		|S[3] S[4] S[5]|
	//		|S[6] S[7] S[8]|

	// compute transposed cofactors matrix

	T[0] = (S[4] * S[8] - S[7] * S[5]);
	T[3] = -(S[3] * S[8] - S[6] * S[5]);
	T[6] = (S[3] * S[7] - S[6] * S[4]);
	T[1] = -(S[1] * S[8] - S[7] * S[2]);
	T[4] = (S[0] * S[8] - S[6] * S[2]);
	T[7] = -(S[0] * S[7] - S[6] * S[1]);
	T[2] = (S[1] * S[5] - S[4] * S[2]);
	T[5] = -(S[0] * S[5] - S[3] * S[2]);
	T[8] = (S[0] * S[4] - S[3] * S[1]);

	// compute determinant

	double inv_det = 1 / (S[0] * T[0] + S[1] * T[3] + S[2] * T[6]);

	// rescale the cofactor matrix to get the inverse matrix
	for (int k = 0; k < 9; k++)
		T[k] *= inv_det;
}

//void transpose(double* S,double* T)
//{
//S(i*)
//}

void inv_matrix_3x3_B(double *S, double *S_B, double *T, double *T_B)
{
	double Tp[9] = {0};
	double Tp_B[9] = {0};
	//	S=	|S[0] S[1] S[2]|
	//		|S[3] S[4] S[5]|
	//		|S[6] S[7] S[8]|

	// mult_matrix(3,3,3,R,T_b,T');
	//T_b=-mult_matrix(3,3,3, T',R)

	// compute transposed cofactors matrix

	Tp[0] = (S[4] * S[8] - S[7] * S[5]);
	Tp[3] = -(S[3] * S[8] - S[6] * S[5]);
	Tp[6] = (S[3] * S[7] - S[6] * S[4]);
	Tp[1] = -(S[1] * S[8] - S[7] * S[2]);
	Tp[4] = (S[0] * S[8] - S[6] * S[2]);
	Tp[7] = -(S[0] * S[7] - S[6] * S[1]);
	Tp[2] = (S[1] * S[5] - S[4] * S[2]);
	Tp[5] = -(S[0] * S[5] - S[3] * S[2]);
	Tp[8] = (S[0] * S[4] - S[3] * S[1]);

	// compute determinant

	double inv_det = 1 / (S[0] * Tp[0] + S[1] * Tp[3] + S[2] * Tp[6]);

	// rescale the cofactor matrix to get the inverse matrix
	for (int k = 0; k < 9; k++)
		T[k] = Tp[k] * inv_det;

	double inv_det_b = 0;
	for (int k = 0; k < 9; k++)
	{
		inv_det_b += Tp[k] * T_B[k];
		Tp_B[k] += inv_det * T_B[k];
	}
	// double t = (S[0] * Tp[0] + S[1] * Tp[3] + S[2] * Tp[6]);
	// double inv_det=1/t
	double t_B = inv_det_b * (-inv_det * inv_det); //?

	S_B[0] += Tp[0] * t_B;
	Tp_B[0] += S[0] * t_B;
	S_B[1] += Tp[3] * t_B;
	Tp_B[3] += S[1] * t_B;
	S_B[2] += Tp[6] * t_B;
	Tp_B[6] += S[2] * t_B;

	//Tp[0]=(S[4]*S[8]-S[7]*S[5]);

	S_B[4] += S[8] * Tp_B[0];
	S_B[8] += S[4] * Tp_B[0];
	S_B[7] += -S[5] * Tp_B[0];
	S_B[5] += -S[7] * Tp_B[0];

	//Tp[3]=-(S[3]*S[8]-S[6]*S[5]);

	S_B[3] += -S[8] * Tp_B[3];
	S_B[8] += -S[3] * Tp_B[3];
	S_B[6] += S[5] * Tp_B[3];
	S_B[5] += S[6] * Tp_B[3];

	// Tp[6]=(S[3]*S[7]-S[6]*S[4]);

	S_B[3] += S[7] * Tp_B[6];
	S_B[7] += S[3] * Tp_B[6];
	S_B[6] += -S[4] * Tp_B[6];
	S_B[4] += -S[6] * Tp_B[6];

	//	Tp[1]=-(S[1]*S[8]-S[7]*S[2]);

	S_B[1] += -S[8] * Tp_B[1];
	S_B[8] += -S[1] * Tp_B[1];
	S_B[7] += S[2] * Tp_B[1];
	S_B[2] += S[7] * Tp_B[1];

	//	Tp[4]=(S[0]*S[8]-S[6]*S[2]);

	S_B[0] += S[8] * Tp_B[4];
	S_B[8] += S[0] * Tp_B[4];
	S_B[6] += -S[2] * Tp_B[4];
	S_B[2] += -S[6] * Tp_B[4];

	//	Tp[7]=-(S[0]*S[7]-S[6]*S[1]);

	S_B[0] += -S[7] * Tp_B[7];
	S_B[7] += -S[0] * Tp_B[7];
	S_B[6] += S[1] * Tp_B[7];
	S_B[1] += S[6] * Tp_B[7];

	//	Tp[2]=(S[1]*S[5]-S[4]*S[2]);
	S_B[1] += S[5] * Tp_B[2];
	S_B[5] += S[1] * Tp_B[2];
	S_B[4] += -S[2] * Tp_B[2];
	S_B[2] += -S[4] * Tp_B[2];

	//	Tp[5]=-(S[0]*S[5]-S[3]*S[2]);

	S_B[0] += -S[5] * Tp_B[5];
	S_B[5] += -S[0] * Tp_B[5];
	S_B[3] += S[2] * Tp_B[5];
	S_B[2] += S[3] * Tp_B[5];

	//	Tp[8]=(S[0]*S[4]-S[3]*S[1]);
	S_B[0] += S[4] * Tp_B[8];
	S_B[4] += S[0] * Tp_B[8];
	S_B[3] += -S[1] * Tp_B[8];
	S_B[1] += -S[3] * Tp_B[8];
}

inline void mul_matrix3x3_vect(double R[3], double M[9], double V[3])
{
	for (int i = 0; i < 3; i++) //v:vertex , d:dimension
	{
		R[i] = 0;
		for (int j = 0; j < 3; j++)
			R[i] += M[3 * i + j] * V[j];
	}
}

inline void mul_matrix3x3_vect_B(double R_B[3], double M_B[9], const double V[3])
{
	for (int i = 0; i < 3; i++) //v:vertex , d:dimension
	{
		for (int j = 0; j < 3; j++)
			M_B[3 * i + j] += R_B[i] * V[j];
	}
}

inline void mul_matrixNx3_vect(int N, double R[], double M[], double V[3])
{
	for (int i = 0; i < N; i++) //v:vertex , d:dimension
	{
		R[i] = 0;
		for (int j = 0; j < 3; j++)
			R[i] += M[3 * i + j] * V[j];
	}
}

inline void mul_matrixNx3_vect_B(int N, double R_B[], double M_B[], const double V[3])
{
	for (int i = 0; i < N; i++) //v:vertex , d:dimension
	{
		for (int j = 0; j < 3; j++)
			M_B[3 * i + j] += R_B[i] * V[j];
	}
}

inline void mul_vect_matrix3x3(double R[3], double V[3], double M[9])
{
	for (int i = 0; i < 3; i++) //v:vertex , d:dimension
	{
		R[i] = 0;
		for (int j = 0; j < 3; j++)
			R[i] += M[3 * j + i] * V[j];
	}
}

inline void mul_vect_matrix3x3_B(const double R[3], const double R_B[3], const double V[3], double V_B[3], double M[9], double M_B[9])
{
	for (int i = 0; i < 3; i++) //v:vertex , d:dimension
	{

		for (int j = 0; j < 3; j++)
		{
			M_B[3 * j + i] += R_B[i] * V[j];
			V_B[j] += R_B[i] * M[3 * j + i];
		}
		//	R[i]+=M[3*j+i]*V[j];
	}
}

inline void mul_matrix(const int I, const int J, const int K, double *AB, double *A, double *B)
// A matrix of size (I,J)
// B matrix of size (J,K)
// AB=A*B
{
	for (int i = 0; i < I; i++) //v:vertex , d:dimension
		for (int k = 0; k < K; k++)
		{
			double *ptr = &AB[K * i + k];
			*ptr = 0;
			for (int j = 0; j < J; j++)
				*ptr += A[i * J + j] * B[j * K + k];
		}
}

inline void mul_matrix_B(const int I, const int J, const int K, double *AB, double *AB_B, double *A, double *A_B, double *B, double *B_B)
// A matrix of size (I,J)
// B matrix of size (J,K)
// AB=A*B
{
	for (int i = 0; i < I; i++) //v:vertex , d:dimension
		for (int k = 0; k < K; k++)
		{
			double *ptr;
			ptr = &AB[K * i + k];
			*ptr = 0;
			for (int j = 0; j < J; j++)
				*ptr += A[i * J + j] * B[j * K + k];

			ptr = &AB_B[K * i + k];
			for (int j = 0; j < J; j++)
			//*ptr+=A[i*J+j]*B[j*K+k];
			{
				A_B[i * J + j] += *ptr * B[j * K + k];
				B_B[j * K + k] += *ptr * A[i * J + j];
			}
		}
}

inline void mul_matrix3x3_vect(double R[3], const double **M, const double V[3])
// M is given one column after antother column
{
	for (int i = 0; i < 3; i++) //v:vertex , d:dimension
	{
		R[i] = 0;
		for (int j = 0; j < 3; j++)
			R[i] += M[j][i] * V[j];
	}
}

inline void mul_matrix_3x3(double AB[9], const double **A, const double B[3])
// A is given one column after antother column
{
	for (int i = 0; i < 3; i++) //v:vertex , d:dimension
		for (int j = 0; j < 3; j++)
		{
			double *ptr = &AB[3 * i + j];
			*ptr = 0;
			for (int k = 0; k < 3; k++)
				*ptr += A[k][i] * B[3 * k + j];
		}
}

inline double dot_prod(const double V1[3], const double V2[3])
{
	double R = 0;
	for (int i = 0; i < 3; i++)
		R += V1[i] * V2[i];
	return R;
}

inline void dot_prod_B(const double R_B, double V1_B[3], const double V2[3])
{
	for (int i = 0; i < 3; i++)
		V1_B[i] += R_B * V2[i];
}

inline void Edge_equ3(double e[3], const double v1[2], const double v2[2], bool clockwise)
{
	// compute edges equations of type ax+by+c = 0

	if (clockwise)
	{
		e[0] = (v1[1] - v2[1]);
		e[1] = (v2[0] - v1[0]);
	}
	else
	{
		e[0] = (v2[1] - v1[1]);
		e[1] = (v1[0] - v2[0]);
	}

	e[2] = -0.5 * (e[0] * (v1[0] + v2[0]) + e[1] * (v1[1] + v2[1]));
}

double signedArea(double ij[3][2], bool clockwise)
{
	double ux = ij[1][0] - ij[0][0];
	double uy = ij[1][1] - ij[0][1];
	double vx = ij[2][0] - ij[0][0];
	double vy = ij[2][1] - ij[0][1];
	return 0.5 * (ux * vy - vx * uy) * (clockwise ? 1 : -1);
}

inline void sort3(const double v[3], double sv[3], short int i[3])
{ // v  : tree unsorted values
	// sv : the three values in v but sorted
	// c  : indices of sorted values

	double tmp1;
	short int tmp2;
	for (int k = 0; k < 3; k++)
		sv[k] = v[k];
	for (int k = 0; k < 3; k++)
		i[k] = k;
	if (sv[0] > sv[1])
	{
		SWAP(sv[0], sv[1], tmp1);
		SWAP(i[0], i[1], tmp2);
	}
	if (sv[0] > sv[2])
	{
		SWAP(sv[0], sv[2], tmp1);
		SWAP(i[0], i[2], tmp2);
	}
	if (sv[1] > sv[2])
	{
		SWAP(sv[1], sv[2], tmp1);
		SWAP(i[1], i[2], tmp2);
	}
}

inline void elementwise_inverse_vec3(const double input[3], double ouput[3])
{
	for (int i = 0; i < 3; i++)
		ouput[i] = 1 / input[i];
}

inline void elementwise_prod_vec3(const double input1[3], const double input2[3], double ouput[3])
{
	for (int i = 0; i < 3; i++)
		ouput[i] = input1[i] * input2[i];
}

short int floor_div(double a, double b, int x_min, int x_max)
{

	// robust implementation of min ( x_max, max (x_min, floor(a / b)))

	short int x;

	if (abs(b) * SHRT_MAX > abs(a) + abs(b))
	{
		x = (short int)floor(a / b);
		if (x < x_min)
		{
			x = x_min;
		}
		if (x > x_max)
		{
			x = x_max;
		}
	}
	else
	{
		if (b > 0)
		{
			x = x_min;
			while (((x + 1) * b <= a) && (x < x_max))
			{
				x++;
			}
		}
		else
		{
			x = x_min;
			while (((x + 1) * b >= a) && (x < x_max))
			{
				x++;
			}
		}
	}
	return x;
}

short int ceil_div(double a, double b, int x_min, int x_max)
{
	// robust implementation of min ( x_max, max (x_min, ceil(a / b)))

	short int x;

	if (abs(b) * SHRT_MAX > abs(a) + abs(b))
	{
		x = (short int)ceil(a / b);
		if (x < x_min)
		{
			x = x_min;
		}
		if (x > x_max)
		{
			x = x_max;
		}
	}
	else
	{
		if (b > 0)
		{
			x = x_min;
			while (((x + 1) * b < a) && (x < x_max))
			{
				x++;
			}
		}
		else
		{
			x = x_min;
			while (((x + 1) * b > a) && (x < x_max))
			{
				x++;
			}
		}
	}
	return x;
}

template <class T>
void bilinear_sample(T *A, T I[], int *I_size, double p[2], int sizeA)
{

	// compute integer part and fractional part

	int fp[2];
	double e[2];

	for (int k = 0; k < 2; k++)
	{
		fp[k] = (int)floor(p[k]);
		e[k] = p[k] - fp[k];
	}

	//  if outside the texture domain then project inside. we could periodize texture

	for (int k = 0; k < 2; k++)
	{
		if (fp[k] < 0)
		{
			fp[k] = 0;
			e[k] = 0;
		}
		if (fp[k] > I_size[k] - 2)
		{
			fp[k] = I_size[k] - 2;
			e[k] = 1;
		}
	}
	// bilinear interpolation

	int indx00 = sizeA * (fp[0] + I_size[0] * fp[1]);
	int indx10 = sizeA * (fp[0] + 1 + I_size[0] * fp[1]);
	int indx01 = sizeA * (fp[0] + I_size[0] * (fp[1] + 1));
	int indx11 = sizeA * (fp[0] + 1 + I_size[0] * (fp[1] + 1));

	for (int k = 0; k < sizeA; k++)
		A[k] = ((1 - e[0]) * I[indx00 + k] + e[0] * I[indx10 + k]) * (1 - e[1]) + ((1 - e[0]) * I[indx01 + k] + e[0] * I[indx11 + k]) * e[1];
}

template <class T>
void bilinear_sample_B(T *A, T *A_B, T I[], T I_B[], int *I_size, double p[2], double p_B[2], int sizeA)
{

	// compute integer part and fractional part

	int fp[2];
	double e[2];
	double e_B[2] = {0};
	int out[2] = {0};

	for (int k = 0; k < 2; k++)
	{
		fp[k] = (int)floor(p[k]);
		e[k] = p[k] - fp[k];
	}

	//  if outside the texture domain then project inside. we could periodize texture

	for (int k = 0; k < 2; k++)
	{
		if (fp[k] < 0)
		{
			out[k] = true;
			fp[k] = 0;
			e[k] = 0;
		}
		if (fp[k] > I_size[k] - 2)
		{
			out[k] = true;
			fp[k] = I_size[k] - 2;
			e[k] = 1;
		}
	}

	// bilinear interpolation

	int indx00 = sizeA * (fp[0] + I_size[0] * fp[1]);
	int indx10 = sizeA * (fp[0] + 1 + I_size[0] * fp[1]);
	int indx01 = sizeA * (fp[0] + I_size[0] * (fp[1] + 1));
	int indx11 = sizeA * (fp[0] + 1 + I_size[0] * (fp[1] + 1));

	//for(int k=0;k<sizeA;k++)
	//	A[k]=( (1-e[0])*I[indx00+k] + e[0]*I[indx10+k] )*(1-e[1])+( (1-e[0])*I[indx01+k] + e[0]*I[indx11+k] )*e[1];

	for (int k = 0; k < sizeA; k++)
	{
		double t1 = ((1 - e[0]) * I[indx00 + k] + e[0] * I[indx10 + k]);
		double t2 = ((1 - e[0]) * I[indx01 + k] + e[0] * I[indx11 + k]);
		//A[k]=t1*(1-e[1])+t2*e[1];
		e_B[1] += -A_B[k] * t1;
		e_B[1] += A_B[k] * t2;

		double t1_B = A_B[k] * (1 - e[1]);
		double t2_B = A_B[k] * e[1];

		e_B[0] += t1_B * (I[indx10 + k] - I[indx00 + k]);
		e_B[0] += t2_B * (I[indx11 + k] - I[indx01 + k]);

		I_B[indx00 + k] = (1 - e[0]) * (1 - e[1]) * A_B[k];
		I_B[indx10 + k] = e[0] * (1 - e[1]) * A_B[k];
		I_B[indx01 + k] = (1 - e[0]) * e[1] * A_B[k];
		I_B[indx11 + k] = e[0] * e[1] * A_B[k];
	}
	for (int k = 0; k < 2; k++)
	{
		if (!out[k])
			p_B[k] += e_B[k];
	}
}

void get_triangle_stencil_equations(double Vxy[][2], double bary_to_xy1[9], double xy1_to_bary[9], double edge_eq[][3], bool strict_edge, int &x_min, int &x_max, int *y_begin, int *y_end, int *left_edge_id, int *right_edge_id)
{

	// create a matrix that map barycentric coordinates to homogeneous image coordinates
	// compute the linear transformation from image coordinates to triangle
	// coordinates
	// the matrix V2D can be seen as the matrix defining the affine
	// transformation from barycentric coordinates to image coordinates
	// because barycentric coordinates sum to one so,
	// the affine transformation mapping from homogeneous image coordinates into triangle
	// coordinates writes :

	for (int v = 0; v < 3; v++) //v:vertex , d:dimension
		for (int d = 0; d < 2; d++)
			bary_to_xy1[3 * d + v] = Vxy[v][d];
	for (int v = 0; v < 3; v++)
		bary_to_xy1[3 * 2 + v] = 1;

	inv_matrix_3x3(bary_to_xy1, xy1_to_bary);

	// compute edges equations of type x=ay+b

	bool clockwise = signedArea(Vxy, true) > 0;

	Edge_equ3(edge_eq[0], Vxy[0], Vxy[1], clockwise);
	Edge_equ3(edge_eq[1], Vxy[1], Vxy[2], clockwise);
	Edge_equ3(edge_eq[2], Vxy[2], Vxy[0], clockwise);

	// sort vertices w.r.t y direction
	double x_unsorted[3];
	double y_unsorted[3];
	short int x_order[3];
	short int y_order[3];
	double x_sorted[3];
	double y_sorted[3];

	for (int k = 0; k < 3; k++)
		x_unsorted[k] = Vxy[k][0];
	sort3(x_unsorted, x_sorted, x_order);
	for (int k = 0; k < 3; k++)
		y_unsorted[k] = Vxy[k][1];
	sort3(y_unsorted, y_sorted, y_order);

	// computing x bounds
	if (strict_edge)
	{
		x_min = (short)floor(x_sorted[0]);
	}
	else
	{
		x_min = (short)ceil(x_sorted[0]);
	}

	x_max = (short)floor(x_sorted[2]);

	// limit upper part
	if (strict_edge)
	{

		y_begin[0] = (short)floor(y_sorted[0]) + 1;
	}
	else
	{

		y_begin[0] = (short)ceil(y_sorted[0]);
	}

	y_end[0] = (short)floor(y_sorted[1]);

	// limit lower part
	if (strict_edge)
	{
		y_begin[1] = (short)floor(y_sorted[1]) + 1;
	}
	else
	{
		y_begin[1] = (short)ceil(y_sorted[1]);
	}
	y_end[1] = (short)floor(y_sorted[2]);

	// set left_edge_id and right_edge_id

	int id;
	id = y_order[0];
	if (edge_eq[(id) % 3][0] > 0)
	{
		right_edge_id[0] = (id + 2) % 3;
		left_edge_id[0] = (id) % 3;
	}
	else
	{
		right_edge_id[0] = (id) % 3;
		left_edge_id[0] = (id + 2) % 3;
	}

	id = y_order[2];
	if (edge_eq[(id) % 3][0] < 0)
	{
		right_edge_id[1] = (id) % 3;
		left_edge_id[1] = (id + 2) % 3;
	}
	else
	{
		right_edge_id[1] = (id + 2) % 3;
		left_edge_id[1] = (id) % 3;
	}
}

template <class T>
void rasterize_triangle_interpolated(double Vxy[][2], double Zvertex[3], T *Avertex[], double z_buffer[], T image[], int height, int width, int sizeA, bool strict_edge, bool perspective_correct)
{
	int y_begin[2], y_end[2];

	double edge_eq[3][3];
	double bary_to_xy1[9];
	double xy1_to_bary[9];
	double *xy1_to_A;
	double xy1_to_Z[3];
	int left_edge_id[2], right_edge_id[2];
	int x_min, x_max;

	//   compute triangle borders equations
	//double Vxy[][2],double  bary_to_xy1[9],double  xy1_to_bary[9],double* edge_eq[3],y_begin,y_end,left_edge_id,right_edge_id)

	get_triangle_stencil_equations(Vxy, bary_to_xy1, xy1_to_bary, edge_eq, strict_edge, x_min, x_max, y_begin, y_end, left_edge_id, right_edge_id);

	// create matrices that map image coordinates to attributes A and depth z
	xy1_to_A = new double[3 * sizeA];

	if (perspective_correct)
	{
		double inv_Zvertex[3];
		elementwise_inverse_vec3(Zvertex, inv_Zvertex);

		for (short int i = 0; i < sizeA; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_A[3 * i + j] = 0;
				for (short int k = 0; k < 3; k++)
					xy1_to_A[3 * i + j] += (Avertex[k][i] * inv_Zvertex[k]) * xy1_to_bary[k * 3 + j];
			}

		mul_vect_matrix3x3(xy1_to_Z, inv_Zvertex, xy1_to_bary);
	}
	else
	{
		for (short int i = 0; i < sizeA; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_A[3 * i + j] = 0;
				for (short int k = 0; k < 3; k++)
					xy1_to_A[3 * i + j] += Avertex[k][i] * xy1_to_bary[k * 3 + j];
			}

		mul_vect_matrix3x3(xy1_to_Z, Zvertex, xy1_to_bary);
	}
	for (int k = 0; k < 2; k++)
	{
		render_part_interpolated(image, z_buffer, x_min, x_max, y_begin[k], y_end[k], strict_edge, xy1_to_A, xy1_to_Z, edge_eq[left_edge_id[k]], edge_eq[right_edge_id[k]], width, height, sizeA, perspective_correct);
	}
	delete[] xy1_to_A;
}

template <class T>
void rasterize_triangle_interpolated_B(double Vxy[][2], double Vxy_B[][2], double Zvertex[3], T *Avertex[], T *Avertex_B[], double z_buffer[], T image[], T image_B[], int height, int width, int sizeA, bool strict_edge, bool perspective_correct)
{
	int y_begin[2], y_end[2];
	double edge_eq[3][3];
	double bary_to_xy1[9];
	double xy1_to_bary[9];
	double *xy1_to_A;
	double xy1_to_Z[3];
	int left_edge_id[2], right_edge_id[2];
	int x_min, x_max;

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	//   compute triangle borders equations
	//double Vxy[][2],double  bary_to_xy1[9],double  xy1_to_bary[9],double* edge_eq[3],y_begin,y_end,left_edge_id,right_edge_id)

	get_triangle_stencil_equations(Vxy, bary_to_xy1, xy1_to_bary, edge_eq, strict_edge, x_min, x_max, y_begin, y_end, left_edge_id, right_edge_id);

	// create matrices that map image coordinates to attributes A and depth z

	xy1_to_A = new double[3 * sizeA];
	for (short int i = 0; i < sizeA; i++)
		for (short int j = 0; j < 3; j++)
		{
			xy1_to_A[3 * i + j] = 0;
			for (short int k = 0; k < 3; k++)
				xy1_to_A[3 * i + j] += Avertex[k][i] * xy1_to_bary[k * 3 + j];
		}

	mul_vect_matrix3x3(xy1_to_Z, Zvertex, xy1_to_bary);
	double *xy1_to_A_B;
	xy1_to_A_B = new double[3 * sizeA];
	for (short int i = 0; i < 3 * sizeA; i++)
		xy1_to_A_B[i] = 0;

	for (int k = 0; k < 2; k++)
		render_part_interpolated_B(image, image_B, z_buffer, x_min, x_max, y_begin[k], y_end[k], strict_edge, xy1_to_A, xy1_to_A_B, xy1_to_Z, edge_eq[left_edge_id[k]], edge_eq[right_edge_id[k]], width, height, sizeA, perspective_correct);

	double xy1_to_bary_B[9] = {0};
	//for(short int i=0;i<9;i++) xy1_to_bary_B[i]=0

	for (short int i = 0; i < sizeA; i++)
		for (short int j = 0; j < 3; j++)
		{

			for (short int k = 0; k < 3; k++)
			//xy1_to_A[3*i+j]+=Avertex[k][i]*xy1_to_bary[k*3+j];
			{
				Avertex_B[k][i] += xy1_to_A_B[3 * i + j] * xy1_to_bary[k * 3 + j];
				xy1_to_bary_B[k * 3 + j] += Avertex[k][i] * xy1_to_A_B[3 * i + j];
			}
		}
	double bary_to_xy1_B[9] = {0};

	inv_matrix_3x3_B(bary_to_xy1, bary_to_xy1_B, xy1_to_bary, xy1_to_bary_B);

	for (int v = 0; v < 3; v++) //v:vertex , d:dimension
		for (int d = 0; d < 2; d++)
			Vxy_B[v][d] += bary_to_xy1_B[3 * d + v];

	delete[] xy1_to_A;
	delete[] xy1_to_A_B;
}

inline void get_xrange(int width, const double *left_eq, const double *right_eq, short int y, bool strict_edge, short int x_min, short int x_max, short int &x_begin, short int &x_end)
{
	// compute beginning and ending of the rasterized line

	short int temp_x;

	double numerator;

	if (x_min < 0)
	{
		x_min = 0;
	}
	if (x_max > width - 1)
	{
		x_max = width - 1;
	}

	x_begin = x_min;
	x_end = x_max;

	numerator = -(left_eq[1] * y + left_eq[2]);

	if (strict_edge)
	{
		// pixels falling exactly on an edge shared by two triangle will be drawn only once
		// when rasterizing the the triangle on the left of the edge.
		temp_x = 1 + floor_div(numerator, left_eq[0], x_min - 1, x_max);
	}
	else
	{
		// pixels falling exactly on an edge shared by two triangle will be drawn twice
		temp_x = ceil_div(numerator, left_eq[0], x_min - 1, x_max);
	}
	if (temp_x > x_begin)
		x_begin = temp_x;

	numerator = -(right_eq[1] * y + right_eq[2]);
	temp_x = floor_div(numerator, right_eq[0], x_min - 1, x_max);
	if (temp_x < x_end)
	{
		x_end = temp_x;
	}
}

inline void render_part_interpolated(double *image, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_A, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, bool perspective_correct)
{
	double t[3];
	double *A0y;
	double Z0y;
	short int x_begin, x_end;
	double Z;
	A0y = new double[sizeA];

	if (y_begin < 0)
	{
		y_begin = 0;
	}
	if (y_end > height - 1)
	{
		y_end = height - 1;
	}
	for (short int y = y_begin; y <= y_end; y++)
	{
		// Line rasterization setup for interpolated values

		t[0] = 0;
		t[1] = y;
		t[2] = 1;

		mul_matrixNx3_vect(sizeA, A0y, xy1_to_A, t);
		Z0y = dot_prod(xy1_to_Z, t);

		// compute beginning and ending of the rasterized line

		get_xrange(width, left_eq, right_eq, y, strict_edge, x_min, x_max, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		if (perspective_correct)
			for (short int x = x_begin; x <= x_end; x++)
			{
				double inv_Z = Z0y + xy1_to_Z[0] * x;
				Z = 1 / (inv_Z);
				if (Z < z_buffer[indx])
				{
					z_buffer[indx] = Z;
					for (short int k = 0; k < sizeA; k++)
						image[sizeA * indx + k] = (A0y[k] + xy1_to_A[3 * k] * x) * Z;
				}
				indx++;
			}
		else
		{
			for (short int x = x_begin; x <= x_end; x++)
			{
				Z = Z0y + xy1_to_Z[0] * x;
				if (Z < z_buffer[indx])
				{
					z_buffer[indx] = Z;
					for (short int k = 0; k < sizeA; k++)
						image[sizeA * indx + k] = A0y[k] + xy1_to_A[3 * k] * x;
				}
				indx++;
			}
		}
	}
	delete[] A0y;
}

inline void render_part_interpolated_B(double *image, double *image_B, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_A, double *xy1_to_A_B, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, bool perspective_correct)
{
	double t[3];
	//double *A0y;
	double *A0y_B;
	double Z0y;
	short int x_begin, x_end;
	double Z;

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	//A0y  =new double[sizeA];
	A0y_B = new double[sizeA];

	if (y_begin < 0)
	{
		y_begin = 0;
	}
	if (y_end > height - 1)
	{
		y_end = height - 1;
	}

	for (short int y = y_begin; y <= y_end; y++)
	{
		// Line rasterization setup for interpolated values

		t[0] = 0;
		t[1] = y;
		t[2] = 1;

		//mul_matrix3x3_vect(A0y,xy1_to_A,t);
		for (short int k = 0; k < sizeA; k++)
			A0y_B[k] = 0;

		Z0y = dot_prod(xy1_to_Z, t);

		// compute beginning and ending of the rasterized line

		get_xrange(width, left_eq, right_eq, y, strict_edge, x_min, x_max, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (short int x = x_begin; x <= x_end; x++)
		{
			Z = Z0y + xy1_to_Z[0] * x;
			if (Z == z_buffer[indx])
			{ //z_buffer[indx]=Z;
				for (short int k = 0; k < sizeA; k++)
				{
					//image[sizeA*indx+k]=A0y[k]+xy1_to_A[3*k]*x;
					A0y_B[k] += image_B[sizeA * indx + k];
					xy1_to_A_B[3 * k] += image_B[sizeA * indx + k] * x;
					image_B[sizeA * indx + k] = 0; // should not be necessary
				}
			}
			indx++;
		}

		mul_matrixNx3_vect_B(sizeA, A0y_B, xy1_to_A_B, t);
	}
	delete[] A0y_B;
}

template <class T>
void rasterize_triangle_textured_gouraud(double Vxy[][2], double Zvertex[3], double UVvertex[][2], double ShadeVertex[], double z_buffer[], T image[], int height, int width, int sizeA, T *Texture, int *Texture_size, bool strict_edge, bool perspective_correct)
{
	int y_begin[2], y_end[2];

	double edge_eq[3][3];
	double bary_to_xy1[9];
	double xy1_to_bary[9];
	double xy1_to_UV[6];
	double xy1_to_L[3];
	double xy1_to_Z[3];
	int left_edge_id[2], right_edge_id[2];
	int x_min, x_max;

	//   compute triangle borders equations

	get_triangle_stencil_equations(Vxy, bary_to_xy1, xy1_to_bary, edge_eq, strict_edge, x_min, x_max, y_begin, y_end, left_edge_id, right_edge_id);

	// create matrices that map image coordinates to attributes A and depth z
	if (perspective_correct)
	{
		double inv_Zvertex[3];
		double ShadeVertex_div_z[3];
		elementwise_inverse_vec3(Zvertex, inv_Zvertex);
		elementwise_prod_vec3(inv_Zvertex, ShadeVertex, ShadeVertex_div_z);
		mul_vect_matrix3x3(xy1_to_Z, inv_Zvertex, xy1_to_bary);
		mul_vect_matrix3x3(xy1_to_L, ShadeVertex_div_z, xy1_to_bary);
		for (short int i = 0; i < 2; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_UV[3 * i + j] = 0;
				for (short int k = 0; k < 3; k++)
					xy1_to_UV[3 * i + j] += (UVvertex[k][i] * inv_Zvertex[k]) * xy1_to_bary[k * 3 + j];
			}
	}
	else
	{
		mul_vect_matrix3x3(xy1_to_Z, Zvertex, xy1_to_bary);
		mul_vect_matrix3x3(xy1_to_L, ShadeVertex, xy1_to_bary);
		for (short int i = 0; i < 2; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_UV[3 * i + j] = 0;
				for (short int k = 0; k < 3; k++)
					xy1_to_UV[3 * i + j] += UVvertex[k][i] * xy1_to_bary[k * 3 + j];
			}
	}

	for (int k = 0; k < 2; k++)
		render_part_textured_gouraud(image, z_buffer, x_min, x_max, y_begin[k], y_end[k], strict_edge, xy1_to_UV, xy1_to_L, xy1_to_Z, edge_eq[left_edge_id[k]], edge_eq[right_edge_id[k]], width, height, sizeA, Texture, Texture_size, perspective_correct);
}

template <class T>
void rasterize_triangle_textured_gouraud_B(double Vxy[][2], double Vxy_B[][2], double Zvertex[3], double UVvertex[][2], double UVvertex_B[][2], double ShadeVertex[], double ShadeVertex_B[], double z_buffer[], T image[], T image_B[], int height, int width, int sizeA, T *Texture, T *Texture_B, int *Texture_size, bool strict_edge, bool perspective_correct)
{
	int y_begin[2], y_end[2];
	double edge_eq[3][3];
	double bary_to_xy1[9];
	double bary_to_xy1_B[9] = {0};
	double xy1_to_bary[9];
	double xy1_to_bary_B[9] = {0};
	double xy1_to_UV[6];
	double xy1_to_L[3];
	double xy1_to_Z[3];
	double xy1_to_UV_B[6] = {0};
	double xy1_to_L_B[3] = {0};
	double xy1_to_Z_B[3] = {0};
	int x_min, x_max;

	int left_edge_id[2], right_edge_id[2];

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	//   compute triangle borders equations

	get_triangle_stencil_equations(Vxy, bary_to_xy1, xy1_to_bary, edge_eq, strict_edge, x_min, x_max, y_begin, y_end, left_edge_id, right_edge_id);

	// create matrices that map image coordinates to attributes A and depth z

	mul_vect_matrix3x3(xy1_to_Z, Zvertex, xy1_to_bary);
	mul_vect_matrix3x3(xy1_to_L, ShadeVertex, xy1_to_bary);

	for (short int i = 0; i < 2; i++)
		for (short int j = 0; j < 3; j++)
		{
			xy1_to_UV[3 * i + j] = 0;
			for (short int k = 0; k < 3; k++)
				xy1_to_UV[3 * i + j] += UVvertex[k][i] * xy1_to_bary[k * 3 + j];
		}

	for (int k = 0; k < 2; k++)
		render_part_textured_gouraud_B(image, image_B, z_buffer, x_min, x_max, y_begin[k], y_end[k], strict_edge, xy1_to_UV, xy1_to_UV_B, xy1_to_L, xy1_to_L_B, xy1_to_Z, edge_eq[left_edge_id[k]], edge_eq[right_edge_id[k]], width, height, sizeA, Texture, Texture_B, Texture_size, perspective_correct);

	for (short int i = 0; i < 2; i++)
		for (short int j = 0; j < 3; j++)
		{
			for (short int k = 0; k < 3; k++)
			{ //xy1_to_UV[3*i+j]+=UVvertex[k][i]*xy1_to_bary[k*3+j];
				UVvertex_B[k][i] += xy1_to_UV_B[3 * i + j] * xy1_to_bary[k * 3 + j];
				xy1_to_bary_B[k * 3 + j] += xy1_to_UV_B[3 * i + j] * UVvertex[k][i];
			}
		}

	mul_vect_matrix3x3_B(xy1_to_L, xy1_to_L_B, ShadeVertex, ShadeVertex_B, xy1_to_bary, xy1_to_bary_B);

	//get_triangle_stencil_equations_B(Vxy,Vxy_B,bary_to_xy1,xy1_to_bary,edge_eq,y_begin,y_end,left_edge_id,right_edge_id);

	inv_matrix_3x3_B(bary_to_xy1, bary_to_xy1_B, xy1_to_bary, xy1_to_bary_B);

	for (int v = 0; v < 3; v++) //v:vertex , d:dimension
		for (int d = 0; d < 2; d++)
			Vxy_B[v][d] += bary_to_xy1_B[3 * d + v];
}

inline void render_part_textured_gouraud(double *image, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_UV, double *xy1_to_L, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, double *Texture, int *Texture_size, bool perspective_correct)
{
	double t[3];
	double L0y;
	double Z0y;
	short int x_begin, x_end;
	double Z;
	double *A;
	double UV0y[2];

	A = new double[sizeA];

	if (y_begin < 0)
	{
		y_begin = 0;
	}
	if (y_end > height - 1)
	{
		y_end = height - 1;
	}

	for (short int y = y_begin; y <= y_end; y++)
	{

		// Line rasterization setup for interpolated values

		t[0] = 0;
		t[1] = y;
		t[2] = 1;

		for (short int i = 0; i < 2; i++)
		{
			UV0y[i] = 0;
			for (short int k = 0; k < 3; k++)
				UV0y[i] += xy1_to_UV[k + 3 * i] * t[k];
		}

		L0y = dot_prod(xy1_to_L, t);
		Z0y = dot_prod(xy1_to_Z, t);

		// compute beginning and ending of the rasterized line

		get_xrange(width, left_eq, right_eq, y, strict_edge, x_min, x_max, x_begin, x_end);

		// line rasterization

		int indx = y * width + x_begin;
		if (perspective_correct)
		{
			for (short int x = x_begin; x <= x_end; x++)
			{
				double inv_Z = Z0y + xy1_to_Z[0] * x;
				Z = 1 / inv_Z;
				if (Z < z_buffer[indx])
				{
					double L;
					double UV[2];

					z_buffer[indx] = Z;
					L = (L0y + xy1_to_L[0] * x) * Z;

					for (int k = 0; k < 2; k++)
						UV[k] = (UV0y[k] + xy1_to_UV[3 * k] * x) * Z;

					bilinear_sample(A, Texture, Texture_size, UV, sizeA);

					for (int k = 0; k < sizeA; k++)
						image[sizeA * indx + k] = A[k] * L;
				}
				indx++;
			}
		}
		else
		{
			for (short int x = x_begin; x <= x_end; x++)
			{

				Z = Z0y + xy1_to_Z[0] * x;
				if (Z < z_buffer[indx])
				{
					double L;
					double UV[2];

					z_buffer[indx] = Z;
					L = L0y + xy1_to_L[0] * x;

					for (int k = 0; k < 2; k++)
						UV[k] = UV0y[k] + xy1_to_UV[3 * k] * x;

					bilinear_sample(A, Texture, Texture_size, UV, sizeA);

					for (int k = 0; k < sizeA; k++)
						image[sizeA * indx + k] = A[k] * L;
				}
				indx++;
			}
		}
	}
	delete[] A;
}

inline void render_part_textured_gouraud_B(double *image, double *image_B, double *z_buffer, int x_min, int x_max, int y_begin, int y_end, bool strict_edge, double *xy1_to_UV, double *xy1_to_UV_B, double *xy1_to_L, double *xy1_to_L_B, double *xy1_to_Z, double *left_eq, double *right_eq, int width, int height, int sizeA, double *Texture, double *Texture_B, int *Texture_size, bool perspective_correct)
{
	double t[3];
	double L0y;
	double Z0y;
	short int x_begin, x_end;
	double Z;
	double *A;
	double *A_B;
	double UV0y[2];
	double UV0y_B[2];

	A = new double[sizeA];
	A_B = new double[sizeA];

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	if (y_begin < 0)
	{
		y_begin = 0;
	}
	if (y_end > height - 1)
	{
		y_end = height - 1;
	}

	for (short int y = y_begin; y <= y_end; y++)
	{

		// Line rasterization setup for interpolated values
		t[0] = 0;
		t[1] = y;
		t[2] = 1;

		for (short int i = 0; i < 2; i++)
		{
			UV0y[i] = 0;
			for (short int k = 0; k < 3; k++)
				UV0y[i] += xy1_to_UV[k + 3 * i] * t[k];
		}
		for (short int i = 0; i < 2; i++)
			UV0y_B[i] = 0;

		L0y = dot_prod(xy1_to_L, t);
		double L0y_B = 0;
		Z0y = dot_prod(xy1_to_Z, t);

		// compute beginning and ending of the rasterized line

		get_xrange(width, left_eq, right_eq, y, strict_edge, x_min, x_max, x_begin, x_end);

		// line rasterization

		int indx = y * width + x_begin;
		for (short int x = x_begin; x <= x_end; x++)
		{
			Z = Z0y + xy1_to_Z[0] * x;
			if (Z == z_buffer[indx])
			{
				double L;
				double UV[2];

				//z_buffer[indx]=Z;
				L = L0y + xy1_to_L[0] * x;
				double L_B = 0;

				for (int k = 0; k < 2; k++)
					UV[k] = UV0y[k] + xy1_to_UV[3 * k] * x;
				double UV_B[2] = {0};

				bilinear_sample(A, Texture, Texture_size, UV, sizeA);
				for (int k = 0; k < sizeA; k++)
					A_B[k] = 0;
				//for(int k=0;k<sizeA;k++)
				//	image[sizeA*indx+k]=A[k]*L;
				//}
				for (int k = 0; k < sizeA; k++)
				{
					A_B[k] += image_B[sizeA * indx + k] * L;
					L_B += image_B[sizeA * indx + k] * A[k];
				}
				bilinear_sample_B(A, A_B, Texture, Texture_B, Texture_size, UV, UV_B, sizeA);
				for (int k = 0; k < 2; k++)
				{ //UV[k]=UV0y[k]+xy1_to_UV[3*k]*x;
					UV0y_B[k] += UV_B[k];
					xy1_to_UV_B[3 * k] += UV_B[k] * x;
				}
				//L=L0y+xy1_to_L[0]*x;
				L0y_B += L_B;
				xy1_to_L_B[0] += x * L_B;
			}
			indx++;
		}
		for (short int i = 0; i < 2; i++)
			for (short int k = 0; k < 3; k++)
				xy1_to_UV_B[k + 3 * i] += UV0y_B[i] * t[k];

		dot_prod_B(L0y_B, xy1_to_L_B, t);
	}
	delete[] A;
	delete[] A_B;
}

void get_edge_stencil_equations(double Vxy[][2], int height, int width, double sigma, double xy1_to_bary[6], double xy1_to_transp[3], double ineq[12], int &y_begin, int &y_end, bool clockwise)
{
	double edge_to_xy1[9];
	double xy1_to_edge[9];
	double n[2];

	// compute the linear transformation from image coordinates to triangle
	// coordinates
	// the matrix V2D can be seen as the matrix defining the affine
	// transformation from barycentric coordinates to image coordinates
	// because barycentric coordinates sum to one so,
	// the affine transfomation mapping from homogeneous image coordinates into triangle
	// coordinates writes :

	// create a matrix that map edge coordinates to homogeneous image coordinates

	// outward normal of the edge
	if (clockwise)
	{
		n[0] = Vxy[0][1] - Vxy[1][1];
		n[1] = Vxy[1][0] - Vxy[0][0];
	}
	else
	{
		n[0] = Vxy[1][1] - Vxy[0][1];
		n[1] = Vxy[0][0] - Vxy[1][0];
	}
	double inv_norm = 1 / sqrt(n[0] * n[0] + n[1] * n[1]);
	n[0] *= inv_norm;
	n[1] *= inv_norm;

	for (int v = 0; v < 2; v++) //v:vertex , d:dimension
		for (int d = 0; d < 2; d++)
			edge_to_xy1[3 * d + v] = Vxy[v][d];
	for (int d = 0; d < 2; d++)
		edge_to_xy1[3 * d + 2] = n[d];
	for (int v = 0; v < 2; v++)
		edge_to_xy1[3 * 2 + v] = 1;
	edge_to_xy1[3 * 2 + 2] = 0;

	inv_matrix_3x3(edge_to_xy1, xy1_to_edge);

	for (int k = 0; k < 6; k++)
	{
		xy1_to_bary[k] = xy1_to_edge[k];
	}

	for (int k = 0; k < 3; k++)
	{
		xy1_to_transp[k] = (1 / sigma) * xy1_to_edge[6 + k];
	}

	// setup border inequalities e*[i,j,1]>0 -> inside the paralellogram

	for (short int k = 0; k < 2; k++)
		for (short int j = 0; j < 3; j++)
		{
			ineq[3 * k + j] = xy1_to_bary[3 * k + j];
		}

	for (short int j = 0; j < 3; j++)
	{
		ineq[3 * 2 + j] = xy1_to_transp[j];
	}

	for (short int j = 0; j < 2; j++)
	{
		ineq[3 * 3 + j] = -xy1_to_transp[j];
	}
	ineq[3 * 3 + 2] = (1 - xy1_to_transp[2]);

	// y limits

	y_begin = height - 1;
	for (short int k = 0; k < 2; k++)
	{
		if (Vxy[k][1] - sigma < y_begin)
			y_begin = (int)floor(Vxy[k][1] - sigma) + 1;
	}
	if (y_begin < 0)
	{
		y_begin = 0;
	}

	y_end = 0;
	for (short int k = 0; k < 2; k++)
	{
		if (Vxy[k][1] + sigma > y_end)
			y_end = (int)floor(Vxy[k][1] + sigma);
	}
	if (y_end > height - 1)
	{
		y_end = height - 1;
	}
}

void get_edge_stencil_equations_B(double Vxy[][2], double Vxy_B[][2], double sigma, double xy1_to_bary_B[6], double xy1_to_transp_B[3], bool clockwise)
{
	double edge_to_xy1[9];
	double xy1_to_edge[9];
	double n[2];
	double nt[2];
	if (clockwise)
	{
		nt[0] = Vxy[0][1] - Vxy[1][1];
		nt[1] = Vxy[1][0] - Vxy[0][0];
	}
	else
	{
		nt[0] = Vxy[1][1] - Vxy[0][1];
		nt[1] = Vxy[0][0] - Vxy[1][0];
	}
	double inv_norm = 1 / sqrt(nt[0] * nt[0] + nt[1] * nt[1]);
	n[0] = nt[0] * inv_norm;
	n[1] = nt[1] * inv_norm;

	for (int v = 0; v < 2; v++) //v:vertex , d:dimension
		for (int d = 0; d < 2; d++)
			edge_to_xy1[3 * d + v] = Vxy[v][d];
	for (int d = 0; d < 2; d++)
		edge_to_xy1[3 * d + 2] = n[d];
	for (int v = 0; v < 2; v++)
		edge_to_xy1[3 * 2 + v] = 1;
	edge_to_xy1[3 * 2 + 2] = 0;

	double edge_to_xy1_B[9] = {0};
	double xy1_to_edge_B[9] = {0};

	for (int k = 0; k < 3; k++)
	{
		xy1_to_edge_B[6 + k] += xy1_to_transp_B[k] * (1 / sigma);
	}
	for (int k = 0; k < 6; k++)
	{
		xy1_to_edge_B[k] += xy1_to_bary_B[k];
	}

	inv_matrix_3x3_B(edge_to_xy1, edge_to_xy1_B, xy1_to_edge, xy1_to_edge_B);

	for (int v = 0; v < 2; v++) //v:vertex , d:dimension
		for (int d = 0; d < 2; d++)
			Vxy_B[v][d] += edge_to_xy1_B[3 * d + v];
	double n_B[3] = {0};
	for (int d = 0; d < 2; d++)
		n_B[d] += edge_to_xy1_B[3 * d + 2];

	double nt_B[2] = {0};
	double inv_norm_B = 0;

	for (int k = 0; k < 2; k++)
	{
		nt_B[k] += n_B[k] * inv_norm;
		inv_norm_B += n_B[k] * nt[k];
	}
	double nor_B = -inv_norm_B * (inv_norm * inv_norm);
	double nor_s_B = nor_B * 0.5 * inv_norm;

	nt_B[0] += 2 * nt[0] * nor_s_B;
	nt_B[1] += 2 * nt[1] * nor_s_B;
	if (clockwise)
	{
		Vxy_B[0][1] += nt_B[0];
		Vxy_B[1][1] += -nt_B[0];
		Vxy_B[1][0] += nt_B[1];
		Vxy_B[0][0] += -nt_B[1];
	}
	else
	{
		Vxy_B[0][1] += -nt_B[0];
		Vxy_B[1][1] += nt_B[0];
		Vxy_B[1][0] += -nt_B[1];
		Vxy_B[0][0] += nt_B[1];
	}
}

template <class Te>
void rasterize_edge_interpolated(double Vxy[][2], Te image[], Te *Avertex[], double z_buffer[], double Zvertex[], int height, int width, int sizeA, double sigma, bool clockwise, bool perspective_correct)
{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double *xy1_to_A;
	double *A0y = new double[sizeA];

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double B_inc[2];
	for (short int k = 0; k < 2; k++)
		B_inc[k] = xy1_to_bary[3 * k];

	double T_inc = xy1_to_transp[0];

	if (perspective_correct)
	{
		double inv_Zvertex[3];
		elementwise_inverse_vec3(Zvertex, inv_Zvertex);
		mul_matrix(1, 2, 3, xy1_to_Z, inv_Zvertex, xy1_to_bary);

		xy1_to_A = new double[3 * sizeA];
		for (short int i = 0; i < sizeA; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_A[3 * i + j] = 0;
				for (short int k = 0; k < 2; k++)
					xy1_to_A[3 * i + j] += (Avertex[k][i] * inv_Zvertex[k]) * xy1_to_bary[k * 3 + j];
			}
	}
	else
	{
		mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);
		xy1_to_A = new double[3 * sizeA];
		for (short int i = 0; i < sizeA; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_A[3 * i + j] = 0;
				for (short int k = 0; k < 2; k++)
					xy1_to_A[3 * i + j] += Avertex[k][i] * xy1_to_bary[k * 3 + j];
			}
	}

	for (short int y = y_begin; y <= y_end; y++)
	{ // Line rasterization setup for interpolated values
		double t[3];
		double T0y, Z0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		mul_matrixNx3_vect(sizeA, A0y, xy1_to_A, t);
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);

		// get x range of the line
		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		if (perspective_correct)
		{
			for (short int x = x_begin; x <= x_end; x++)
			{
				double inv_Z = Z0y + xy1_to_Z[0] * x;
				double Z = 1 / inv_Z;
				if (Z < z_buffer[indx])
				{
					double T = T0y + T_inc * x;

					for (short int k = 0; k < sizeA; k++)
					{
						image[sizeA * indx + k] *= T;
						double A = (A0y[k] + xy1_to_A[3 * k] * x) * Z;
						image[sizeA * indx + k] += (1 - T) * A;
					}
				}
				indx++;
			}
		}
		else
		{
			for (short int x = x_begin; x <= x_end; x++)
			{
				double Z = Z0y + xy1_to_Z[0] * x;
				if (Z < z_buffer[indx])
				{
					double T = T0y + T_inc * x;

					for (short int k = 0; k < sizeA; k++)
					{
						image[sizeA * indx + k] *= T;
						double A = (A0y[k] + xy1_to_A[3 * k] * x);
						image[sizeA * indx + k] += (1 - T) * A;
					}
				}
				indx++;
			}
		}
	}
	delete[] A0y;
	delete[] xy1_to_A;
}

template <class Te>
void rasterize_edge_interpolated_B(double Vxy[][2], double Vxy_B[][2], Te image[], Te image_B[], Te *Avertex[], Te *Avertex_B[], double z_buffer[], double Zvertex[], int height, int width, int sizeA, double sigma, bool clockwise, bool perspective_correct)
{
	double xy1_to_bary[6];
	double xy1_to_bary_B[6] = {0};
	double xy1_to_transp[3];
	double xy1_to_transp_B[3] = {0};
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double *xy1_to_A;
	double *A0y = new double[sizeA];
	double *A0y_B = new double[sizeA];

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	//double B_inc[2];
	//for(short int k=0;k<2;k++)
	//	B_inc[k] = xy1_to_bary  [3*k];

	double T_inc = xy1_to_transp[0];
	double T_inc_B = 0;

	mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);

	xy1_to_A = new double[3 * sizeA];
	double *xy1_to_A_B;
	xy1_to_A_B = new double[3 * sizeA];
	for (short int i = 0; i < 3 * sizeA; i++)
		xy1_to_A_B[i] = 0;

	for (short int i = 0; i < sizeA; i++)
		for (short int j = 0; j < 3; j++)
		{
			xy1_to_A[3 * i + j] = 0;
			for (short int k = 0; k < 2; k++)
				xy1_to_A[3 * i + j] += Avertex[k][i] * xy1_to_bary[k * 3 + j];
		}

	for (short int y = y_begin; y <= y_end; y++)
	{ // Line rasterization setup for interpolated values
		double t[3];
		double T0y, Z0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		mul_matrixNx3_vect(sizeA, A0y, xy1_to_A, t);
		for (short int k = 0; k < sizeA; k++)
			A0y_B[k] = 0;

		T0y = dot_prod(xy1_to_transp, t);
		double T0y_B = 0;
		Z0y = dot_prod(xy1_to_Z, t);

		// get x range of the line
		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (short int x = x_begin; x <= x_end; x++)
		{
			double Z = Z0y + xy1_to_Z[0] * x;
			if (Z < z_buffer[indx])
			{
				double T = T0y + T_inc * x;
				double T_B = 0;

				for (short int k = 0; k < sizeA; k++)
				{

					double A = A0y[k] + xy1_to_A[3 * k] * x;
					//image[sizeA*indx+k]*= T;

					//image[sizeA*indx+k]+= (1-T)*A;
					T_B += -image_B[sizeA * indx + k] * A;
					double A_B = (1 - T) * image_B[sizeA * indx + k];

					// restoring the color before edge drawned

					image[sizeA * indx + k] = (image[sizeA * indx + k] - (1 - T) * A) / T;

					T_B += image_B[sizeA * indx + k] * image[sizeA * indx + k];

					image_B[sizeA * indx + k] *= T;

					A0y_B[k] += A_B;
					xy1_to_A_B[3 * k] += x * A_B;
				}
				T0y_B += T_B;
				T_inc_B += x * T_B;
			}
			indx++;
		}
		mul_matrixNx3_vect_B(sizeA, A0y_B, xy1_to_A_B, t);
		//T0y=dot_prod_B(xy1_to_transp,t);
		for (int k = 0; k < 3; k++)
			xy1_to_transp_B[k] += T0y_B * t[k];
	}

	for (short int i = 0; i < sizeA; i++)
		for (short int j = 0; j < 3; j++)
		{
			for (short int k = 0; k < 2; k++)
			//xy1_to_A[3*i+j]+=Avertex[k][i]*xy1_to_bary[k*3+j];
			{
				Avertex_B[k][i] += xy1_to_A_B[3 * i + j] * xy1_to_bary[k * 3 + j];
				xy1_to_bary_B[k * 3 + j] += Avertex[k][i] * xy1_to_A_B[3 * i + j];
			}
		}

	//for(short int k=0;k<2;k++)
	//	xy1_to_bary_B  [3*k]+=B_inc_B[k];
	xy1_to_transp_B[0] += T_inc_B;

	get_edge_stencil_equations_B(Vxy, Vxy_B, sigma, xy1_to_bary_B, xy1_to_transp_B, clockwise);

	delete[] A0y;
	delete[] A0y_B;
	delete[] xy1_to_A;
	delete[] xy1_to_A_B;
}

template <class Te>
void rasterize_edge_textured_gouraud(double Vxy[][2], double Zvertex[2], double UVvertex[][2], double ShadeVertex[2], double z_buffer[], Te image[], int height, int width, int sizeA, Te *Texture, int *Texture_size, double sigma, bool clockwise, bool perspective_correct)
{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double xy1_to_UV[6];
	double xy1_to_L[3];
	double *A;
	double UV0y[2];

	A = new double[sizeA];

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double B_inc[2];
	for (short int k = 0; k < 2; k++)
		B_inc[k] = xy1_to_bary[3 * k];

	double T_inc = xy1_to_transp[0];
	if (perspective_correct)
	{
		double inv_Zvertex[3];
		double ShadeVertex_div_z[3];

		elementwise_inverse_vec3(Zvertex, inv_Zvertex);
		elementwise_prod_vec3(inv_Zvertex, ShadeVertex, ShadeVertex_div_z);
		mul_matrix(1, 2, 3, xy1_to_Z, inv_Zvertex, xy1_to_bary);
		mul_matrix(1, 2, 3, xy1_to_L, ShadeVertex_div_z, xy1_to_bary);

		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				xy1_to_UV[3 * i + j] = 0;
				for (int k = 0; k < 2; k++)
					xy1_to_UV[3 * i + j] += UVvertex[k][i] * inv_Zvertex[k] * xy1_to_bary[k * 3 + j];
			}
		}
	}
	else
	{
		mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);
		mul_matrix(1, 2, 3, xy1_to_L, ShadeVertex, xy1_to_bary);

		for (short int i = 0; i < 2; i++)
			for (short int j = 0; j < 3; j++)
			{
				xy1_to_UV[3 * i + j] = 0;
				for (short int k = 0; k < 2; k++)
					xy1_to_UV[3 * i + j] += UVvertex[k][i] * xy1_to_bary[k * 3 + j];
			}
	}

	for (short int y = y_begin; y <= y_end; y++)
	{ // Line rasterization setup for interpolated values

		double t[3];
		double T0y, Z0y, L0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);
		L0y = dot_prod(xy1_to_L, t);

		for (short int i = 0; i < 2; i++)
		{
			UV0y[i] = 0;
			for (short int k = 0; k < 3; k++)
				UV0y[i] += xy1_to_UV[k + 3 * i] * t[k];
		}

		// get x range of the line

		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (short int x = x_begin; x <= x_end; x++)
		{
			double Z;
			if (perspective_correct)
			{
				double inv_Z = Z0y + xy1_to_Z[0] * x;
				Z = 1 / inv_Z;
			}
			else
			{
				Z = Z0y + xy1_to_Z[0] * x;
			}
			if (Z < z_buffer[indx])
			{
				double L = L0y + xy1_to_L[0] * x;
				;
				double T = T0y + T_inc * x;
				double UV[2];
				for (int k = 0; k < 2; k++)
				{
					UV[k] = UV0y[k] + xy1_to_UV[3 * k] * x;
				}
				if (perspective_correct)
				{
					L *= Z;
					for (int k = 0; k < 2; k++)
					{
						UV[k] *= Z;
					}
				}
				bilinear_sample(A, Texture, Texture_size, UV, sizeA);

				for (short int k = 0; k < sizeA; k++)
				{
					image[sizeA * indx + k] *= T;
					image[sizeA * indx + k] += (1 - T) * A[k] * L;
				}
			}
			indx++;
		}
	}
	delete[] A;
}

template <class Te>
void rasterize_edge_textured_gouraud_B(double Vxy[][2], double Vxy_B[][2], double Zvertex[2], double UVvertex[][2], double UVvertex_B[][2], double ShadeVertex[2], double ShadeVertex_B[2], double z_buffer[], Te image[], Te image_B[], int height, int width, int sizeA, Te *Texture, Te *Texture_B, int *Texture_size, double sigma, bool clockwise, bool perspective_correct)

{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double xy1_to_UV[6];
	double xy1_to_UV_B[6];
	double xy1_to_L[3];

	double *A;
	double *A_B;
	double UV0y[2];

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}
	A = new double[sizeA];
	A_B = new double[sizeA];

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double xy1_to_bary_B[6] = {0};
	double xy1_to_transp_B[3] = {0};

	//double B_inc[2];
	//for(short int k=0;k<2;k++)
	//	B_inc[k] = xy1_to_bary  [3*k];

	double T_inc = xy1_to_transp[0];
	double T_inc_B = 0;

	mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);
	mul_matrix(1, 2, 3, xy1_to_L, ShadeVertex, xy1_to_bary);
	double xy1_to_L_B[3] = {0};

	for (short int i = 0; i < 2; i++)
		for (short int j = 0; j < 3; j++)
		{
			xy1_to_UV[3 * i + j] = 0;
			xy1_to_UV_B[3 * i + j] = 0;
			for (short int k = 0; k < 2; k++)
				xy1_to_UV[3 * i + j] += UVvertex[k][i] * xy1_to_bary[k * 3 + j];
		}

	for (short int y = y_begin; y <= y_end; y++)
	{ // Line rasterization setup for interpolated values

		double t[3];
		double T0y, Z0y, L0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);
		L0y = dot_prod(xy1_to_L, t);
		double L0y_B = 0;
		double T0y_B = 0;

		for (short int i = 0; i < 2; i++)
		{
			UV0y[i] = 0;
			for (short int k = 0; k < 3; k++)
				UV0y[i] += xy1_to_UV[k + 3 * i] * t[k];
		}
		double UV0y_B[2] = {0};

		// get x range of the line

		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (short int x = x_begin; x <= x_end; x++)
		{
			double Z = Z0y + xy1_to_Z[0] * x;
			if (Z < z_buffer[indx])
			{

				double L = L0y + xy1_to_L[0] * x;
				double L_B = 0;
				double T = T0y + T_inc * x;
				double T_B = 0;

				double UV[2];
				for (int k = 0; k < 2; k++)
					UV[k] = UV0y[k] + xy1_to_UV[3 * k] * x;

				bilinear_sample(A, Texture, Texture_size, UV, sizeA);

				for (short int k = 0; k < sizeA; k++)
					A_B[k] = 0;
				for (short int k = 0; k < sizeA; k++)
				{
					//image[sizeA*indx+k]*= T;
					//image[sizeA*indx+k]+= (1-T)*A[k]*L;
					T_B += -image_B[sizeA * indx + k] * A[k] * L;
					A_B[k] += L * (1 - T) * image_B[sizeA * indx + k];
					L_B += image_B[sizeA * indx + k] * (1 - T) * A[k];
					image[sizeA * indx + k] = (image[sizeA * indx + k] - (1 - T) * A[k] * L) / T;
					T_B += image_B[sizeA * indx + k] * image[sizeA * indx + k];
					image_B[sizeA * indx + k] *= T;
				}

				double UV_B[2] = {0};
				bilinear_sample_B(A, A_B, Texture, Texture_B, Texture_size, UV, UV_B, sizeA);

				for (int k = 0; k < 2; k++)
				{ //UV[k]=UV0y[k]+xy1_to_UV[3*k]*x;
					UV0y_B[k] += UV_B[k];
					xy1_to_UV_B[3 * k] += UV_B[k] * x;
				}
				//L=L0y+xy1_to_L[0]*x;
				L0y_B += L_B;
				xy1_to_L_B[0] += x * L_B;
				T0y_B += T_B;
				T_inc_B += x * T_B;
			}
			indx++;
		}

		for (int k = 0; k < 3; k++)
			xy1_to_transp_B[k] += T0y_B * t[k];
		for (short int i = 0; i < 2; i++)
			for (short int k = 0; k < 3; k++)
				xy1_to_UV_B[k + 3 * i] += UV0y_B[i] * t[k];
		dot_prod_B(L0y_B, xy1_to_L_B, t);
	}

	for (short int i = 0; i < 2; i++)
		for (short int j = 0; j < 3; j++)
		{
			for (short int k = 0; k < 2; k++)
			//xy1_to_UV[3*i+j]+=UVvertex[k][i]*xy1_to_bary[k*3+j];
			{
				UVvertex_B[k][i] += xy1_to_UV_B[3 * i + j] * xy1_to_bary[k * 3 + j];
				xy1_to_bary_B[k * 3 + j] += xy1_to_UV_B[3 * i + j] * UVvertex[k][i];
			}
		}

	mul_matrix_B(1, 2, 3, xy1_to_L, xy1_to_L_B, ShadeVertex, ShadeVertex_B, xy1_to_bary, xy1_to_bary_B);

	xy1_to_transp_B[0] += T_inc_B;

	get_edge_stencil_equations_B(Vxy, Vxy_B, sigma, xy1_to_bary_B, xy1_to_transp_B, clockwise);

	delete[] A;
	delete[] A_B;
}

template <class T>
void rasterize_edge_textured_gouraud_error(double Vxy[][2], double Zvertex[2], double UVvertex[][2], double ShadeVertex[2], double z_buffer[], T image[], double *err_buffer, int height, int width, int sizeA, T *Texture, int *Texture_size, double sigma, bool clockwise, bool perspective_correct)

{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double xy1_to_UV[6];
	double xy1_to_L[3];
	double *A;
	double UV0y[2];

	A = new double[sizeA];

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double B_inc[2];
	for (int k = 0; k < 2; k++)
		B_inc[k] = xy1_to_bary[3 * k];

	double T_inc = xy1_to_transp[0];

	if (perspective_correct)
	{
		double inv_Zvertex[3];
		double ShadeVertex_div_z[3];
		elementwise_inverse_vec3(Zvertex, inv_Zvertex);
		elementwise_prod_vec3(inv_Zvertex, ShadeVertex, ShadeVertex_div_z);
		mul_matrix(1, 2, 3, xy1_to_Z, inv_Zvertex, xy1_to_bary);
		mul_matrix(1, 2, 3, xy1_to_L, ShadeVertex_div_z, xy1_to_bary);

		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				xy1_to_UV[3 * i + j] = 0;
				for (int k = 0; k < 2; k++)
					xy1_to_UV[3 * i + j] += UVvertex[k][i] * inv_Zvertex[k] * xy1_to_bary[k * 3 + j];
			}
		}
	}
	else
	{
		mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);
		mul_matrix(1, 2, 3, xy1_to_L, ShadeVertex, xy1_to_bary);

		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				xy1_to_UV[3 * i + j] = 0;
				for (int k = 0; k < 2; k++)
					xy1_to_UV[3 * i + j] += UVvertex[k][i] * xy1_to_bary[k * 3 + j];
			}
		}
	}
	for (int y = y_begin; y <= y_end; y++)
	{

		// Line rasterization setup for interpolated values

		double t[3];
		double T0y, Z0y, L0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);
		L0y = dot_prod(xy1_to_L, t);

		for (int i = 0; i < 2; i++)
		{
			UV0y[i] = 0;
			for (int k = 0; k < 3; k++)
				UV0y[i] += xy1_to_UV[k + 3 * i] * t[k];
		}

		// get x range of the line

		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (int x = x_begin; x <= x_end; x++)
		{
			double Z;

			if (perspective_correct)
			{
				double inv_Z = Z0y + xy1_to_Z[0] * x;
				Z = 1 / inv_Z;
			}
			else
			{
				Z = Z0y + xy1_to_Z[0] * x;
			}

			if (Z < z_buffer[indx])
			{
				double L = L0y + xy1_to_L[0] * x;
				double Tr = T0y + T_inc * x;
				double UV[2];
				for (int k = 0; k < 2; k++)
					UV[k] = UV0y[k] + xy1_to_UV[3 * k] * x;

				if (perspective_correct)
				{
					L *= Z;
					for (int k = 0; k < 2; k++)
						UV[k] *= Z;
				}

				bilinear_sample(A, Texture, Texture_size, UV, sizeA);
				double Err = 0;
				for (int k = 0; k < sizeA; k++)
				{
					double diff = A[k] * L - image[sizeA * indx + k];
					Err += diff * diff;
				}
				err_buffer[indx] *= Tr;
				err_buffer[indx] += (1 - Tr) * Err;
			}
			indx++;
		}
	}
	delete[] A;
}

template <class T>
void rasterize_edge_textured_gouraud_error_B(double Vxy[][2], double Vxy_B[][2], double Zvertex[2], double UVvertex[][2], double UVvertex_B[][2], double ShadeVertex[2], double ShadeVertex_B[2], double z_buffer[], T image[], double err_buffer[], double err_buffer_B[], int height, int width, int sizeA, T *Texture, T *Texture_B, int *Texture_size, double sigma, bool clockwise, bool perspective_correct)
{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double xy1_to_UV[6];
	double xy1_to_UV_B[6];
	double xy1_to_L[3];

	double *A;
	double *A_B;
	double UV0y[2];

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	A = new double[sizeA];
	A_B = new double[sizeA];

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double xy1_to_bary_B[6] = {0};
	double xy1_to_transp_B[3] = {0};

	//double B_inc[2];
	//for(int k=0;k<2;k++)
	//	B_inc[k] = xy1_to_bary  [3*k];

	double T_inc = xy1_to_transp[0];
	double T_inc_B = 0;

	mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);
	mul_matrix(1, 2, 3, xy1_to_L, ShadeVertex, xy1_to_bary);
	double xy1_to_L_B[3] = {0};

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			xy1_to_UV[3 * i + j] = 0;
			xy1_to_UV_B[3 * i + j] = 0;
			for (int k = 0; k < 2; k++)
				xy1_to_UV[3 * i + j] += UVvertex[k][i] * xy1_to_bary[k * 3 + j];
		}
	}

	for (int y = y_begin; y <= y_end; y++)
	{ // Line rasterization setup for interpolated values

		double t[3];
		double T0y, Z0y, L0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);
		L0y = dot_prod(xy1_to_L, t);
		double L0y_B = 0;
		double T0y_B = 0;

		for (int i = 0; i < 2; i++)
		{
			UV0y[i] = 0;
			for (int k = 0; k < 3; k++)
				UV0y[i] += xy1_to_UV[k + 3 * i] * t[k];
		}
		double UV0y_B[2] = {0};

		// get x range of the line

		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (int x = x_begin; x <= x_end; x++)
		{
			double Z = Z0y + xy1_to_Z[0] * x;
			if (Z < z_buffer[indx])
			{
				double L = L0y + xy1_to_L[0] * x;
				double L_B = 0;
				double Tr = T0y + T_inc * x;
				double Tr_B = 0;

				double UV[2];
				for (int k = 0; k < 2; k++)
					UV[k] = UV0y[k] + xy1_to_UV[3 * k] * x;

				bilinear_sample(A, Texture, Texture_size, UV, sizeA);
				for (int k = 0; k < sizeA; k++)
					A_B[k] = 0;
				double Err = 0;
				for (int k = 0; k < sizeA; k++)
				{
					double diff = A[k] * L - image[sizeA * indx + k];
					Err += diff * diff;
				}

				double Err_B = 0;
				//	err_buffer[indx]*= Tr;
				//	err_buffer[indx]+= (1-Tr)*Err;
				Tr_B += -Err * err_buffer_B[indx];
				Err_B += (1 - Tr) * err_buffer_B[indx];
				err_buffer[indx] -= (1 - Tr) * Err;
				err_buffer[indx] /= Tr;
				Tr_B += err_buffer_B[indx] * err_buffer[indx];
				err_buffer_B[indx] *= Tr;

				for (int k = 0; k < sizeA; k++)
				{
					double diff = A[k] * L - image[sizeA * indx + k];
					//Err+=diff*diff;
					double diff_B = 2 * diff * Err_B;
					A_B[k] += diff_B * L;
					L_B += diff_B * A[k];
				}

				double UV_B[2] = {0};

				bilinear_sample_B(A, A_B, Texture, Texture_B, Texture_size, UV, UV_B, sizeA);
				for (int k = 0; k < 2; k++)
				{ //UV[k]=UV0y[k]+xy1_to_UV[3*k]*x;
					UV0y_B[k] += UV_B[k];
					xy1_to_UV_B[3 * k] += UV_B[k] * x;
				}
				//L=L0y+xy1_to_L[0]*x;
				L0y_B += L_B;
				xy1_to_L_B[0] += x * L_B;
				T0y_B += Tr_B;
				T_inc_B += x * Tr_B;
			}
			indx++;
		}
		for (int k = 0; k < 3; k++)
			xy1_to_transp_B[k] += T0y_B * t[k];
		for (int i = 0; i < 2; i++)
			for (int k = 0; k < 3; k++)
				xy1_to_UV_B[k + 3 * i] += UV0y_B[i] * t[k];
		dot_prod_B(L0y_B, xy1_to_L_B, t);
	}

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 2; k++)
			//xy1_to_UV[3*i+j]+=UVvertex[k][i]*xy1_to_bary[k*3+j];
			{
				UVvertex_B[k][i] += xy1_to_UV_B[3 * i + j] * xy1_to_bary[k * 3 + j];
				xy1_to_bary_B[k * 3 + j] += xy1_to_UV_B[3 * i + j] * UVvertex[k][i];
			}
		}
	}
	mul_matrix_B(1, 2, 3, xy1_to_L, xy1_to_L_B, ShadeVertex, ShadeVertex_B, xy1_to_bary, xy1_to_bary_B);

	xy1_to_transp_B[0] += T_inc_B;

	get_edge_stencil_equations_B(Vxy, Vxy_B, sigma, xy1_to_bary_B, xy1_to_transp_B, clockwise);

	delete[] A;
	delete[] A_B;
}

template <class T>
void rasterize_edge_interpolated_error(double Vxy[][2], double Zvertex[2], T *Avertex[], double z_buffer[], T image[], double *err_buffer, int height, int width, int sizeA, double sigma, bool clockwise, bool perspective_correct)

{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];
	double *xy1_to_A = new double[3 * sizeA];
	double *A0y = new double[sizeA];

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double B_inc[2];
	for (int k = 0; k < 2; k++)
		B_inc[k] = xy1_to_bary[3 * k];

	double T_inc = xy1_to_transp[0];

	if (perspective_correct)
	{
		double inv_Zvertex[3];
		elementwise_inverse_vec3(Zvertex, inv_Zvertex);
		mul_matrix(1, 2, 3, xy1_to_Z, inv_Zvertex, xy1_to_bary);
		for (int i = 0; i < sizeA; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				xy1_to_A[3 * i + j] = 0;
				for (int k = 0; k < 2; k++)
					xy1_to_A[3 * i + j] += Avertex[k][i] * inv_Zvertex[k] * xy1_to_bary[k * 3 + j];
			}
		}
	}
	else
	{
		mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);

		for (int i = 0; i < sizeA; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				xy1_to_A[3 * i + j] = 0;
				for (int k = 0; k < 2; k++)
					xy1_to_A[3 * i + j] += Avertex[k][i] * xy1_to_bary[k * 3 + j];
			}
		}
	}

	for (int y = y_begin; y <= y_end; y++)
	{ // Line rasterization setup for interpolated values

		double t[3];
		double T0y, Z0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		mul_matrixNx3_vect(sizeA, A0y, xy1_to_A, t);
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);

		// get x range of the line

		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (int x = x_begin; x <= x_end; x++)
		{
			double Z;

			if (perspective_correct)
			{
				double inv_Z = Z0y + xy1_to_Z[0] * x;
				Z = 1 / inv_Z;
			}
			else
			{
				Z = Z0y + xy1_to_Z[0] * x;
			}
			if (Z < z_buffer[indx])
			{

				double Tr = T0y + T_inc * x;
				double Err = 0;
				for (int k = 0; k < sizeA; k++)
				{
					double A = (A0y[k] + xy1_to_A[3 * k] * x);
					if (perspective_correct)
					{
						A *= Z;
					}
					double diff = A - image[sizeA * indx + k];
					Err += diff * diff;
				}
				err_buffer[indx] *= Tr;
				err_buffer[indx] += (1 - Tr) * Err;
			}
			indx++;
		}
	}

	delete[] A0y;
	delete[] xy1_to_A;
}

template <class T>
void rasterize_edge_interpolated_error_B(double Vxy[][2], double Vxy_B[][2], double Zvertex[2], T *Avertex[], T *Avertex_B[], double z_buffer[], T image[], double err_buffer[], double err_buffer_B[], int height, int width, int sizeA, double sigma, bool clockwise, bool perspective_correct)

{
	double xy1_to_bary[6];
	double xy1_to_transp[3];
	double ineq[12];
	int y_begin, y_end;
	double xy1_to_Z[3];

	double *A0y = new double[sizeA];
	double *A0y_B = new double[sizeA];
	double *xy1_to_A = new double[3 * sizeA];
	double *xy1_to_A_B = new double[3 * sizeA];

	if (perspective_correct)
	{
		throw "backward gradient propagation not supported yet with perspective_correct=True";
	}

	get_edge_stencil_equations(Vxy, height, width, sigma, xy1_to_bary, xy1_to_transp, ineq, y_begin, y_end, clockwise);

	double xy1_to_bary_B[6] = {0};
	double xy1_to_transp_B[3] = {0};

	//double B_inc[2];
	//for(int k=0;k<2;k++)
	//	B_inc[k] = xy1_to_bary  [3*k];

	double T_inc = xy1_to_transp[0];
	double T_inc_B = 0;

	mul_matrix(1, 2, 3, xy1_to_Z, Zvertex, xy1_to_bary);

	for (short int i = 0; i < 3 * sizeA; i++)
		xy1_to_A_B[i] = 0;

	for (int i = 0; i < sizeA; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			xy1_to_A[3 * i + j] = 0;
			for (int k = 0; k < 2; k++)
				xy1_to_A[3 * i + j] += Avertex[k][i] * xy1_to_bary[k * 3 + j];
		}
	}

	for (short int y = y_begin; y <= y_end; y++)
	{
		// Line rasterization setup for interpolated values

		double t[3];
		double T0y, Z0y;

		t[0] = 0;
		t[1] = y;
		t[2] = 1;
		mul_matrixNx3_vect(sizeA, A0y, xy1_to_A, t);
		for (short int k = 0; k < sizeA; k++)
			A0y_B[k] = 0;
		T0y = dot_prod(xy1_to_transp, t);
		Z0y = dot_prod(xy1_to_Z, t);
		double T0y_B = 0;

		// get x range of the line

		int x_begin, x_end;
		get_edge_xrange_from_ineq(ineq, width, y, x_begin, x_end);

		//rasterize line

		int indx = y * width + x_begin;
		for (int x = x_begin; x <= x_end; x++)
		{
			double Z = Z0y + xy1_to_Z[0] * x;
			if (Z < z_buffer[indx])
			{

				double Tr = T0y + T_inc * x;
				double Tr_B = 0;

				double Err = 0;
				for (int k = 0; k < sizeA; k++)
				{
					double A = A0y[k] + xy1_to_A[3 * k] * x;
					double diff = A - image[sizeA * indx + k];
					Err += diff * diff;
				}

				double Err_B = 0;
				//	err_buffer[indx]*= Tr;
				//	err_buffer[indx]+= (1-Tr)*Err;
				Tr_B += -Err * err_buffer_B[indx];
				Err_B += (1 - Tr) * err_buffer_B[indx];
				err_buffer[indx] -= (1 - Tr) * Err;
				err_buffer[indx] /= Tr;
				Tr_B += err_buffer_B[indx] * err_buffer[indx];
				err_buffer_B[indx] *= Tr;

				for (int k = 0; k < sizeA; k++)
				{
					double A = A0y[k] + xy1_to_A[3 * k] * x;
					double diff = A - image[sizeA * indx + k];
					//Err+=diff*diff;
					double diff_B = 2 * diff * Err_B;
					double A_B = diff_B;
					A0y_B[k] += A_B;
					xy1_to_A_B[3 * k] += x * A_B;
				}

				T0y_B += Tr_B;
				T_inc_B += x * Tr_B;
			}
			indx++;
		}
		for (int k = 0; k < 3; k++)
			xy1_to_transp_B[k] += T0y_B * t[k];
	}

	for (short int i = 0; i < sizeA; i++)
		for (short int j = 0; j < 3; j++)
		{
			for (short int k = 0; k < 2; k++)
			//xy1_to_A[3*i+j]+=Avertex[k][i]*xy1_to_bary[k*3+j];
			{
				Avertex_B[k][i] += xy1_to_A_B[3 * i + j] * xy1_to_bary[k * 3 + j];
				xy1_to_bary_B[k * 3 + j] += Avertex[k][i] * xy1_to_A_B[3 * i + j];
			}
		}

	xy1_to_transp_B[0] += T_inc_B;

	get_edge_stencil_equations_B(Vxy, Vxy_B, sigma, xy1_to_bary_B, xy1_to_transp_B, clockwise);

	delete[] A0y;
	delete[] A0y_B;
	delete[] xy1_to_A;
	delete[] xy1_to_A_B;
}

void get_edge_xrange_from_ineq(double ineq[12], int width, int y, int &x_begin, int &x_end)
{
	// compute beginning and ending of the rasterized line while doing edge antialiasing

	x_begin = 0;
	x_end = width - 1;
	double numerator;

	for (short int k = 0; k < 4; k++)
	{
		numerator = -(ineq[3 * k + 1] * y + ineq[3 * k + 2]);
		if (ineq[3 * k] < 0)
		{
			short int temp_x = floor_div(numerator, ineq[3 * k], x_begin - 1, x_end + 1);
			if (temp_x < x_end)
			{
				x_end = temp_x;
			}
		}
		else
		{
			short int temp_x = 1 + floor_div(numerator, ineq[3 * k], x_begin - 1, x_end + 1);
			if (temp_x > x_begin)
			{
				x_begin = temp_x;
			}
		}
	}
}

struct sortdata
{
	double value;
	size_t index;
};

struct sortcompare
{
	bool operator()(sortdata const &left, sortdata const &right)
	{
		return left.value > right.value;
	}
};

void checkSceneValid(Scene scene, bool has_derivatives)
{
	if (scene.faces == NULL)
		throw "scene.texture == NULL";
	if (scene.faces_uv == NULL)
		throw "scene.faces_uv == NULL";
	if (scene.depths == NULL)
		throw "scene.depths == NULL";
	if (scene.uv == NULL)
		throw "scene.uv == NULL";
	if (scene.ij == NULL)
		throw "scene.ij == NULL";
	if (scene.shade == NULL)
		throw "scene.shade == NULL";
	if (scene.colors == NULL)
		throw "scene.colors == NULL";
	if (scene.edgeflags == NULL)
		throw "scene.edgeflags == NULL";
	if (scene.textured == NULL)
		throw "scene.textured == NULL";
	if (scene.shaded == NULL)
		throw "scene.shaded == NULL";
	if (scene.texture == NULL)
		throw "scene.texture == NULL";
	if ((scene.background_image == NULL) && (scene.background_color == NULL))
		throw "scene.background == NULL and scene.background_color == NULL";
	if (has_derivatives)
	{
		if (scene.uv_b == NULL)
			throw "scene.uv_b == NULL";
		if (scene.ij_b == NULL)
			throw "scene.ij_b == NULL";
		if (scene.shade_b == NULL)
			throw "scene.shade_b == NULL";
		if (scene.colors_b == NULL)
			throw "scene.colors_b == NULL";
		if (scene.texture_b == NULL)
			throw "scene.texture_b == NULL";
	}
	for (int k = 0; k < scene.nb_triangles * 3; k++)
	{
		if (scene.faces[k] >= (unsigned int)scene.nb_vertices)
		{
			throw "scene.faces value greater than scene.nb_vertices";
		}
		if (scene.faces_uv[k] >= (unsigned int)scene.nb_uv)
		{
			std::cout << "scene.faces_uv value " << scene.faces_uv[k] << " greater than  scene.nb_uv (" << (unsigned int)scene.nb_uv << ")" << std::endl;
			throw "scene.faces_uv value greater than scene.nb_uv";
		}
	}
}

void renderScene(Scene scene, double *image, double *z_buffer, double sigma, bool antialiaseError = 0, double *obs = NULL, double *err_buffer = NULL)
{

	checkSceneValid(scene, false);
	// first pass : render triangle without edge antialiasing

	int Texture_size[2];

	Texture_size[1] = scene.texture_height;
	Texture_size[0] = scene.texture_width;

	if (scene.background_image != NULL)
	{
		memcpy(image, scene.background_image, scene.height * scene.width * scene.nb_colors * sizeof(double));
	}
	else
	{
		for (int i = 0; i < scene.height * scene.width; i++)
		{
			for (int k = 0; k < scene.nb_colors; k++)
			{
				image[i * scene.nb_colors + k] = scene.background_color[k];
			}
		}
	}
	//for (int k=0;k<scene.height*scene.width;k++)
	//z_buffer[k]=100000;
	fill(z_buffer, z_buffer + scene.height * scene.width, numeric_limits<double>::infinity());

	vector<sortdata> sum_depth;
	sum_depth.resize(scene.nb_triangles);
	vector<double> signedAreaV;
	signedAreaV.resize(scene.nb_triangles);

	for (int k = 0; k < scene.nb_triangles; k++)
	{
		sum_depth[k].value = 0;
		sum_depth[k].index = k;

		bool all_verticesInFront = true;
		unsigned int *face = &scene.faces[k * 3];

		for (int i = 0; i < 3; i++)
		{
			if (scene.depths[face[i]] < 0)
			{
				all_verticesInFront = false;
			}
			sum_depth[k].value += scene.depths[face[i]];
		}
		if (all_verticesInFront)
		{
			double ij[3][2];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 2; j++)
					ij[i][j] = scene.ij[face[i] * 2 + j];
			signedAreaV[k] = signedArea(ij, scene.clockwise);
		}
		else
		{
			signedAreaV[k] = 0;
		}
	}

	sort(sum_depth.begin(), sum_depth.end(), sortcompare());

	float pixel_center_offset = scene.integer_pixel_centers ? 0 : 0.5;

	for (int k = 0; k < scene.nb_triangles; k++)
		if ((signedAreaV[k] > 0) || (!scene.backface_culling))
		{
			unsigned int *face = &scene.faces[k * 3];
			double ij[3][2];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 2; j++)
					ij[i][j] = scene.ij[face[i] * 2 + j] - pixel_center_offset;

			double depths[3];
			for (int i = 0; i < 3; i++)
				depths[i] = scene.depths[face[i]];

			if ((scene.textured[k] && scene.shaded[k]))
			{
				unsigned int *face_uv = &scene.faces_uv[k * 3];
				double shade[3];
				for (int i = 0; i < 3; i++)
					shade[i] = scene.shade[face[i]];
				double uv[3][2];
				for (int i = 0; i < 3; i++)
					for (int j = 0; j < 2; j++)
					{
						uv[i][j] = scene.uv[face_uv[i] * 2 + j];
					}

				rasterize_triangle_textured_gouraud(ij, depths, uv, shade, z_buffer, image, scene.height, scene.width, scene.nb_colors, scene.texture, Texture_size, scene.strict_edge, scene.perspective_correct);
			}
			if (!scene.textured[k])
			{
				double *colors[3];
				for (int i = 0; i < 3; i++)
					colors[i] = scene.colors + face[i] * scene.nb_colors;
				rasterize_triangle_interpolated(ij, depths, colors, z_buffer, image, scene.height, scene.width, scene.nb_colors, scene.strict_edge, scene.perspective_correct);
			}
		}

	int list_sub[3][2] = {1, 0, 2, 1, 0, 2};

	if (antialiaseError)
	{
		for (int k = 0; k < scene.width * scene.height; k++)
		{
			double s = 0;
			double d;
			for (int i = 0; i < scene.nb_colors; i++)
			{
				d = (image[scene.nb_colors * k + i] - obs[scene.nb_colors * k + i]);
				s += d * d;
			}
			err_buffer[k] = s;
		}
	}

	if (sigma > 0)
	{
		for (int it = 0; it < scene.nb_triangles; it++)
		{
			size_t k = sum_depth[it].index; // we render the silhouette edges from the furthest from the camera to the nearest as we don't use z_buffer for discontinuity edge overdraw
			unsigned int *face = &scene.faces[k * 3];

			//k=order[i];
			if (signedAreaV[k] > 0)
			{

				for (int n = 0; n < 3; n++)
				{

					if (scene.edgeflags[n + k * 3])
					{
						double ij[2][2];
						int *sub;
						sub = list_sub[n];
						for (int i = 0; i < 2; i++)
							for (int j = 0; j < 2; j++)
								ij[i][j] = scene.ij[face[sub[i]] * 2 + j] - pixel_center_offset;

						double depths[2];
						for (int i = 0; i < 2; i++)
						{
							depths[i] = scene.depths[face[sub[i]]];
						}

						if ((scene.textured[k]) && (scene.shaded[k]))
						{
							unsigned int *face_uv = &scene.faces_uv[k * 3];

							double uv[2][2];
							for (int i = 0; i < 2; i++)
								for (int j = 0; j < 2; j++)
									uv[i][j] = scene.uv[face_uv[sub[i]] * 2 + j];
							double shade[2];
							for (int i = 0; i < 2; i++)
								shade[i] = scene.shade[face[sub[i]]];
							if (antialiaseError)
								rasterize_edge_textured_gouraud_error(ij, depths, uv, shade, z_buffer, obs, err_buffer, scene.height, scene.width, scene.nb_colors, scene.texture, Texture_size, sigma, scene.clockwise, scene.perspective_correct);
							else
								rasterize_edge_textured_gouraud(ij, depths, uv, shade, z_buffer, image, scene.height, scene.width, scene.nb_colors, scene.texture, Texture_size, sigma, scene.clockwise, scene.perspective_correct);
						}
						else
						{
							double *colors[2];
							for (int i = 0; i < 2; i++)
							{
								colors[i] = scene.colors + face[sub[i]] * scene.nb_colors;
							}
							if (antialiaseError)
								rasterize_edge_interpolated_error(ij, depths, colors, z_buffer, obs, err_buffer, scene.height, scene.width, scene.nb_colors, sigma, scene.clockwise, scene.perspective_correct);
							else
								rasterize_edge_interpolated(ij, image, colors, z_buffer, depths, scene.height, scene.width, scene.nb_colors, sigma, scene.clockwise, scene.perspective_correct);
						}
					}
				}
			}
		}
	}
}

void renderScene_B(Scene scene, double *image, double *z_buffer, double *image_b, double sigma, bool antialiaseError = 0, double *obs = NULL, double *err_buffer = NULL, double *err_buffer_b = NULL)
{

	// first pass : render triangle without edge antialiasing

	int Texture_size[2];

	Texture_size[1] = scene.texture_height;
	Texture_size[0] = scene.texture_width;

	checkSceneValid(scene, true);

	int list_sub[3][2] = {1, 0, 2, 1, 0, 2};

	vector<sortdata> sum_depth;
	sum_depth.resize(scene.nb_triangles);
	vector<double> signedAreaV;
	signedAreaV.resize(scene.nb_triangles);

	if (!scene.backface_culling)
	{
		throw "You have to use backface_culling true if you ant to compute gradients";
	}

	for (int k = 0; k < scene.nb_triangles; k++)
	{
		sum_depth[k].value = 0;
		sum_depth[k].index = k;

		bool all_verticesInFront = true;
		unsigned int *face = &scene.faces[k * 3];

		for (int i = 0; i < 3; i++)
		{
			if (scene.depths[face[i]] < 0)
			{
				all_verticesInFront = false;
			}
			sum_depth[k].value += scene.depths[face[i]];
		}
		if (all_verticesInFront)
		{
			double ij[3][2];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 2; j++)
					ij[i][j] = scene.ij[face[i] * 2 + j];
			signedAreaV[k] = signedArea(ij, scene.clockwise);
		}
		else
		{
			signedAreaV[k] = 0;
		}
	}

	sort(sum_depth.begin(), sum_depth.end(), sortcompare());

	float pixel_center_offset = scene.integer_pixel_centers ? 0 : 0.5;

	if (sigma > 0)
		for (int it = scene.nb_triangles - 1; it >= 0; it--)
		{
			size_t k = sum_depth[it].index;
			unsigned int *face = &scene.faces[k * 3];

			if (signedAreaV[k] > 0)
				for (int n = 2; n >= 0; n--)
				{
					if (scene.edgeflags[n + k * 3])
					{
						double ij[2][2];
						int *sub;
						sub = list_sub[n];
						for (int i = 0; i < 2; i++)
							for (int j = 0; j < 2; j++)
								ij[i][j] = scene.ij[face[sub[i]] * 2 + j] - pixel_center_offset;
						double ij_b[2][2];
						sub = list_sub[n];
						for (int i = 0; i < 2; i++)
							for (int j = 0; j < 2; j++)
								ij_b[i][j] = scene.ij_b[face[sub[i]] * 2 + j];
						double depths[2];
						for (int i = 0; i < 2; i++)
						{
							depths[i] = scene.depths[face[sub[i]]];
						}

						if ((scene.textured[k]) && (scene.shaded[k]))
						{

							unsigned int *face_uv = &scene.faces_uv[k * 3];
							double uv[2][2];
							double uv_b[2][2];
							for (int i = 0; i < 2; i++)
								for (int j = 0; j < 2; j++)
								{
									uv[i][j] = scene.uv[face_uv[sub[i]] * 2 + j];
									uv_b[i][j] = scene.uv_b[face_uv[sub[i]] * 2 + j];
								}

							double shade[2];
							double shade_b[2];
							for (int i = 0; i < 2; i++)
							{
								shade[i] = scene.shade[face[sub[i]]];
								shade_b[i] = scene.shade_b[face[sub[i]]];
							}

							if (antialiaseError)
							{
								rasterize_edge_textured_gouraud_error_B(ij, ij_b, depths, uv, uv_b, shade, shade_b, z_buffer, obs, err_buffer, err_buffer_b, scene.height, scene.width, scene.nb_colors, scene.texture, scene.texture_b, Texture_size, sigma, scene.clockwise, scene.perspective_correct);
							}
							else
							{
								rasterize_edge_textured_gouraud_B(ij, ij_b, depths, uv, uv_b, shade, shade_b, z_buffer, image, image_b, scene.height, scene.width, scene.nb_colors, scene.texture, scene.texture_b, Texture_size, sigma, scene.clockwise, scene.perspective_correct);
							}

							for (int i = 0; i < 2; i++)
								for (int j = 0; j < 2; j++)
								{
									scene.uv_b[face_uv[sub[i]] * 2 + j] = uv_b[i][j];
								}
							for (int i = 0; i < 2; i++)
							{
								scene.shade_b[face[sub[i]]] = shade_b[i];
							}
						}
						else
						{
							double *colors[2];
							double *colors_b[2];

							for (int i = 0; i < 2; i++)
							{
								colors[i] = scene.colors + face[sub[i]] * scene.nb_colors;
								colors_b[i] = scene.colors_b + face[sub[i]] * scene.nb_colors;
							}

							if (antialiaseError)
								rasterize_edge_interpolated_error_B(ij, ij_b, depths, colors, colors_b, z_buffer, obs, err_buffer, err_buffer_b, scene.height, scene.width, scene.nb_colors, sigma, scene.clockwise, scene.perspective_correct);
							else
								rasterize_edge_interpolated_B(ij, ij_b, image, image_b, colors, colors_b, z_buffer, depths, scene.height, scene.width, scene.nb_colors, sigma, scene.clockwise, scene.perspective_correct);
						}
						for (int i = 0; i < 2; i++)
							for (int j = 0; j < 2; j++)
							{
								scene.ij_b[face[sub[i]] * 2 + j] = ij_b[i][j];
							}
					}
				}
		}

	if (antialiaseError)
	{
		image_b = new double[scene.width * scene.height * scene.nb_colors];
		for (int k = 0; k < scene.width * scene.height; k++)
			for (int i = 0; i < scene.nb_colors; i++)
				image_b[scene.nb_colors * k + i] = -2 * (obs[scene.nb_colors * k + i] - image[scene.nb_colors * k + i]) * err_buffer_b[k];
	}

	for (int k = scene.nb_triangles - 1; k >= 0; k--)
		if (signedAreaV[k] > 0)
		{
			unsigned int *face = &scene.faces[k * 3];
			double ij[3][2];
			double ij_b[3][2];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 2; j++)
					ij[i][j] = scene.ij[face[i] * 2 + j] - pixel_center_offset;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 2; j++)
					ij_b[i][j] = scene.ij_b[face[i] * 2 + j];

			double depths[3];
			for (int i = 0; i < 3; i++)
			{
				depths[i] = scene.depths[face[i]];
			}

			if (scene.textured[k] && scene.shaded[k])
			{

				unsigned int *face_uv = &scene.faces_uv[k * 3];
				double uv[3][2];
				double uv_b[3][2];
				double shade[3];
				double shade_b[3];

				for (int i = 0; i < 3; i++)
					shade[i] = scene.shade[face[i]];

				for (int i = 0; i < 3; i++)
					shade_b[i] = scene.shade_b[face[i]];

				for (int i = 0; i < 3; i++)
					for (int j = 0; j < 2; j++)
					{
						uv[i][j] = scene.uv[face_uv[i] * 2 + j];
						uv_b[i][j] = scene.uv_b[face_uv[i] * 2 + j];
					}

				rasterize_triangle_textured_gouraud_B(ij, ij_b, depths, uv, uv_b, shade, shade_b, z_buffer, image, image_b, scene.height, scene.width, scene.nb_colors, scene.texture, scene.texture_b, Texture_size, scene.strict_edge, scene.perspective_correct);
				for (int i = 0; i < 3; i++)
					for (int j = 0; j < 2; j++)
					{
						scene.uv_b[face_uv[i] * 2 + j] = uv_b[i][j];
					}
				for (int i = 0; i < 3; i++)
					scene.shade_b[face[i]] = shade_b[i];
			}
			if (!scene.textured[k])
			{
				double *colors[3];
				double *colors_b[3];

				for (int i = 0; i < 3; i++)
				{
					colors[i] = scene.colors + face[i] * scene.nb_colors;
					colors_b[i] = scene.colors_b + face[i] * scene.nb_colors;
				}

				rasterize_triangle_interpolated_B(ij, ij_b, depths, colors, colors_b, z_buffer, image, image_b, scene.height, scene.width, scene.nb_colors, scene.strict_edge, scene.perspective_correct);
			}

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 2; j++)
					scene.ij_b[face[i] * 2 + j] = ij_b[i][j];
		}

	if (antialiaseError)
	{
		delete[] image_b;
	}
}
