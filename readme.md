# DEODR

DEODR (for Discontinuity-Edge-Overdraw based Differentiable Renderer) is a differentiable 3D mesh renderer written in C with **Python** (Pytorch) and **Matlab** bindings. It provides a rendering backward differentiation function that will provides derivatives of a loss defined on the rendered image with respect to the lightning, the 3D vertices positions and the vertices colors. 
The core triangle rasterization procedures and their adjoint are written in C for speed, while the vertices normals computation and camera projection are computed in either PyTorch or Matlab in order to gain flexibility and improve the integration with automatic differentiation libraries. Unlike most other differentiable renderers, the rendering is differentiable along the occlusion boundaries and no had-hoc approximation is needed in the backpropagation pass to deal with occlusion boundaries. This is achieved by using a differentiable antialiasing method called *Disontinuity-edge-overdraw* [2] that progressively blends the colour of the front triangle with the back triangle along occlusion boundaries. 

# Features

* linearly interpolated color triangles with arbitray number of color chanels
* textured triangles with gouraud shading
* derivatives with respect to triangles vertices positions, triangles colors and lights. 
* derivatives along occlusion boundaries
* differentiability of the rendering function 
* exact gradient of the rendering function

Some unsuported features:

* derivatives with respect to the texture
* differentiable handling of seams at visible self intersections
* GPU acceleration
* self-collision detection to prevent interpenetrations
* texture mip-mapping 
 
#Using texture triangles

Keeping the rendering differentiable everywhere when using texture is challenging: if you use textured triangles you will need to make sure there no adjacent triangles in the 3D mesh are simultaneously visibles while disconected in the UV map, i.e that there is no visible seam. Otherwise the rendering won't in general be continuous with respect to the 3D vertices positions due to the texture discontinuity along the seam. Depending on the shape of your object, you might not be able to  define continuous UV mapping over the entire mesh and will need to define the UV texture coordinates in a very specific manner described in Figure 3 in [1], with some contraints on the texture intensities so that the continuity of the rendering is still garanteed along edges between disconected triangles in the UV map after texture bilinear interpolation.
Notre that an improved version that approach is also described in [7].

# Installation
## Python

### Windows

	pip install git+https://github.com/martinResearch/DEODR.git
	

## Matlab
Simply download the zip file, decompress it, run compile.m.

For the hand fitting example you will also need to download the automatic differentiation toolbox from [here](https://github.com/martinResearch/MatlabAutoDiff) 
add add the decompressed folder in your matlab path

# Examples

## Iterative Mesh fitting in Python 
 
Example of fitting a hand mesh to a depth sensor image [*DEODR/examples/depth_image_hand_fitting.py*](PyDEODR/examples/depth_image_hand_fitting.py)
 ![animation](./images/python_depth_hand.gif)

Example of fitting a hand mesh to a RGB sensor image [*DEODR/examples/depth_image_hand_fitting.py*](PyDEODR/examples/rgb_image_hand_fitting.py)
 ![animation](./images/python_rgb_hand.gif)

## iterative mesh fitting in Matlab
You can call a simple triangle soup fitting [here](Matlab/examples/triangle_soup_fitting.m) 

![animation](./images/soup_fitting.gif)

You can call a simple hand fitting example [here](Matlab/examples/hand_fitting.m).
For this example you will also need to download the automatic differentiation toolbox from [https://github.com/martinResearch/MatlabAutoDiff](https://github.com/martinResearch/MatlabAutoDiff) 

![animation](./images/hand_fitting.gif)



 
# Details

This code implements the core of the differentiable renderer described in [1] and has been mostly written in 2008-2009. It is anterior to OpenDR and is to my knowledge the first differentiable renderer to appear in the litterature.
It renders a set of triangles with texture bilinearly interpolated and shaded or with interpolated RGB colour. In contrast with most renderers, the rendered image is differentiable with respect to the vertices positions even along occlusion boundaries. This is achieved by using a differentiable antialiasing method called *Discontinuity-Edge-Overdraw* [2] that progressively blends the colour of the front triangle with the back triangle along occlusion boundaries, using a linear combination of from and back triangles with a mixing coefficient that varies continously as the reprojected vertices move in the image (see [1] for more details). This allows us to and capture the effect of change of visibility along occlusion boundaries in the gradient of the loss in a principeld manner by simply applying the chain rule of derivatives to our differentiable rendering function. Note that this code does not provide explicitly the sparse Jacobian of the rendering function (where each row correspond to a color intensity of a pixel of the rendered image, like done in [3]) but provides the vector-Jacobian product operator, which corresponds to the backward function in PyTorch.

This can be used to do efficient analysis-by-synthesis computer vision by minimizing the function E that sums the squared difference between a rendered image and a reference observed image I<sub>o</sub> with respect of the scene parameters we aim to estimate.

![latex: \large~~~~~~~~~~~~~~ $E(V)=\sum_{ij} (I(i,j,V)-I_o(i,j))^2$](./images/error_function.svg)


Using D(i,j) as the *adjoint* i.e. derivative of the error (for example the squared residual) between observed pixel intensity and synthetized pixel intensity at location (i,j) 

![latex: \large~~~~~~~~~~~~~~ $ D(i,j)=\partial E/\partial I(i,j))$](./images/adjoint.svg)

We can use DEODR to obtain the gradient with respect to the 2D vertices locations  and their colors i.e :
 
![latex: \large ~~~~~~~~~~~~~~$\partial E/\partial V_k =\sum_{ij} D(i,j)(\partial I(i,j)/\partial V_k)$](./images/backoperator1.svg)
 
and 

![latex: \large ~~~~~~~~~~~~~~$\partial E/\partial C_k = \sum_{ij} D(i,j)(\partial I(i,j)/\partial C_k)$](./images/backoperator2.svg)


In combination with automatic differentiation tool this core function allows to obtain the gradient of 
the error function with respect to the parameters of a complex 3D scene we aim to estimate.


The rendering function implemented in C++ can draw image given a list of 2D projected triangles with associated depth (i.e 2.5D scene), where each triangle can have 

* a linearly interpolated color between its three extremities 
* a texture with linear texture mapping (no perspective-correct texture mapping yet but it will lead to noticeable bias only for large triangle that are no fronto-parallel)
* a texture combined with shading interpolated linearly (gouraud shading)

We provide a functions in Matlab and Python to obtain the 2.5D representation of the scene from a textured 3D mesh, a camera and simple lighting model. This function is used in the example of 3D model hand fitting on image. 
We kept this function as minimalistic as possible as we did not intend to rewrite an entire rendering pipeline but to focus on the part that is difficult to differentiate and that cannot be differentiated easily using automatic differentiation.

Our code provides two methods to handle discontinuities at the occlusion boundaries

* the first method consists in antialiasing the synthetic image before comparing it to the observed image. 
* the second method consists in antialising the squared residual between the observed image and the synthetized one, and corresponds to the method described in [1]. Note that antialiasing the residual instead of the squared residual is equivalent to do antialiasing on the synthetized image and then subtract the observed image.

The choice of the method is done through the Boolean parameter *antialiaseError*. Both approach lead to a differentiable error function after summation of the residuals over the pixels and both lead to similar gradients. The difference is subtle and is only noticeable at the borders after convergence on synthetic antialiased data. The first methods can potentially provide more flexibility for the design of the error function. One can for example use non-local comparison by comparing image moments instead of comparing pixel per pixel.

**Note:** In order to keep the code minimal and well documented, I decided not to provide here the code to model the articulated hand and the code to update the texture image from observation used in [1]. The hand fitting example provided here does not relies on a underlying skeleton but on a regularization term that favors rigid deformations. Some Matlab code for Linear Blend Skinning can be found [here](http://uk.mathworks.com/matlabcentral/fileexchange/43039-linear-blend-skinning/). Using a Matlab implementation of the skinning equations would allow the use of the Matlab automatic differentiation toolbox provided [here](https://github.com/martinResearch/MatlabAutoDiff) to compute the Jacobian of the vertices positions with respect to the hand pose parameters.

# License

[BSD 2-clause "Simplified" license](licence.txt).



# Citation

If you use any part of this work please cite the following:

Model-based 3D Hand Pose Estimation from Monocular Video. M. de la Gorce, N. Paragios and David Fleet. PAMI 2011 [pdf](http://www.cs.toronto.edu/~fleet/research/Papers/deLaGorcePAMI2011.pdf)

    @article{deLaGorce:2011:MHP:2006854.2007005,
     author = {de La Gorce, Martin and Fleet, David J. and Paragios, Nikos},
     title = {Model-Based 3D Hand Pose Estimation from Monocular Video},
     journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
     issue_date = {September 2011},
     volume = {33},
     number = {9},
     month = sep,
     year = {2011},
     issn = {0162-8828},
     pages = {1793--1805},
     numpages = {13},
     url = {http://dx.doi.org/10.1109/TPAMI.2011.33},
     doi = {10.1109/TPAMI.2011.33},
     acmid = {2007005},
     publisher = {IEEE Computer Society},
     address = {Washington, DC, USA},
    } 







## Other differentiable renderers 
* [**OpenDR**](https://github.com/mattloper/opendr/wiki) [3] (MIT Licence) is an open source differentiable renderer written in python and make publicly available in 2014. OpenDR calls OpenGL and relies an a python automatic differentiation toolbox by the same author called [chumpy](https://github.com/mattloper/chumpy). Like in our code OpenDR uses a intermediate 2.5D representation of the scene using a set of 2D projected triangles. In contrast to our code OpenDR does not provide a continuous loss function as there is not continuous antialiasing formulation at the occlusion boundaries and the minimised function will have jumps when a pixel at the boundary switch between the front of back object. By providing a continuous differentiable error function using edge-overdraw antialiasing and its exact gradient, our method can lead to better a convergence of continuous optimisation methods..

* [**DIRT**](https://github.com/pmh47/dirt) (MIT licence) is an open soure differentiable renderer that uses approximations in the gradient computation similar OpenDR but that is interfaced with tensorflow. It makes considerable effort to return correctly-behaving derivatives even in cases of self-occlusion, where most other differentiable renderers can fail. 

* [**Neural 3D Mesh Renderer**](https://github.com/hiroharu-kato/neural_renderer) (MIT Licence). Method published in [5]. This method consists in a differentiable render whose gradients are designed to be used in neural networks. It is claimed in the paper that the gradients computed by OpenDR are not adequate for neural network use, but there is unfortunalty no detailed explaination of why the autors came to that conclusion.
While anterior to this paper, the method in [1] can be used in conjunction with a neural network. 
* [**tf\_mesh\_renderer**](https://github.com/google/tf_mesh_renderer) (Apache License 2.0). A differentiable, 3D mesh renderer using TensorFlow. [github](https://github.com/google/tf_mesh_renderer)
Unlike other differentiable renderer it does not provides suppport for oclusion boundaries in the gradient computation and thus is inadequate for many applications.
* Code accompanying the paper [6] [github](https://github.com/ndrplz/differentiable-renderer). It renders only silhouettes. 

## References
[1] *Model-based 3D Hand Pose Estimation from Monocular Video. M. de la Gorce, N. Paragios and David Fleet.* PAMI 2011 [pdf](http://www.cs.toronto.edu/~fleet/research/Papers/deLaGorcePAMI2011.pdf)

[2] *Discontinuity edge overdraw* P.V. Sander and H. Hoppe, J.Snyder and S.J. Gortler. SI3D 2001 [pdf](http://hhoppe.com/overdraw.pdf)

[3] *OpenDR: An Approximate Differentiable Renderer* Loper, Matthew M. and Black, Michael J. ECCV 2014 [pdf](http://files.is.tue.mpg.de/black/papers/OpenDR.pdf)
[code](https://github.com/mattloper/opendr) [online documentation](https://github.com/mattloper/opendr/wiki)

[4] *A Morphable Model For The Synthesis Of 3D Faces*. Volker Blanz and Thomas Vetter. SIGGRAPH 99

[5] *Neural 3D Mesh Renderer*. Hiroharu Kato and Yoshitaka Ushiku and Tatsuya Harada.CoRR 2017 [PDF](https://arxiv.org/pdf/1711.07566.pdf)

[6] *End-to-end 6-DoF Object Pose Estimation through Differentiable Rasterization* Andrea Palazzi, Luca Bergamini, Simone Calderara, Rita Cucchiara. Second Workshop on 3D Reconstruction Meets Semantics (3DRMS) at ECCVW 2018.

[7] *Mesh Color textures* Cem Yuksel. Proceedings of High Performance Graphics 2017
