
# DEODR

DEODR (for Discontinuity-Edge-Overdraw based Differentiable Renderer) is a differentiable 3D mesh renderer written in C with **Python** and **Matlab** bindings. The python code provides interfaces with **Pytorch** and **Tensorflow**. It provides a differentiable rendering function and its associated reverse mode differentiation function (a.k.a adjoint function) that provides derivatives of a loss defined on the rendered image with respect to the lightning, the 3D vertices positions and the vertices colors. 
The core triangle rasterization procedures and their adjoint are written in C for speed, while the vertices normals computation and camera projections are computed in either Python (numpy, pytorch or tensorflow) or Matlab in order to gain flexibility and improve the integration with automatic differentiation libraries. The core C++ differentiable renderer has been implemented in 2008 and described in [1,2]. Unlike most other differentiable renderers (except the recent SoftRas [8] and to some extend the differentiable ray/path tracing methods in [10] and [13]), the rendering is differentiable along the occlusion boundaries and no had-hoc approximation is needed in the backpropagation pass to deal with discontinuities along occlusion boundaries. This is achieved by using a differentiable antialiasing method called *Discontinuity-Edge-Overdraw* [3] that progressively blends the colour of the front triangle with the back triangle along occlusion boundaries.

[![PyPI version](https://badge.fury.io/py/deodr.svg)](https://badge.fury.io/py/deodr)
[![Build status](https://github.com/martinResearch/DEODR/actions/workflows/wheels.yml/badge.svg)
![Python package](https://github.com/martinResearch/DEODR/actions/workflows/pythonpackage.yml/badge.svg)

# Table of content

1. [Features](#Features)
2. [Installation](#Installation)
3. [Examples](#Examples)
4. [Equations](#Equations) 
5. [License](#License)
5. [Alternatives](#Alternatives)
6. [References](#References)

# Features

* linearly interpolated color triangles with arbitrary number of color channels.
* textured triangles with Gouraud shading. The gradient backward pass is supported only for linear texture mapping (it is not implemented for perspective-correct texture mapping yet).
* derivatives with respect to triangles vertices positions, triangles colors and lights. 
* derivatives with respect to the texture pixel intensities
* derivatives with respect to the texture UV coordinates
* derivatives along occlusion boundaries
* differentiability of the rendering function 
* exact gradient of the rendering function
* classical camera projection representation used in computer vision 
* camera distortion with OpenCV's 5 distortion parameters described [here](https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html). It requires small triangles surface tesselations as the distortion is applied only at the vertices projection stage. 
* possibility to render images corresponding to depth, normals, albedo, shading, xyz coordinates, object/background mask and faces ids. Example [here](https://github.com/martinResearch/DEODR/blob/master/deodr/examples/render_mesh.py#L68) 

Some **unsupported** features:

* SIMD instructions acceleration
* multithreading
* GPU acceleration
* differentiable handling of seams at visible self intersections
* self-collision detection to prevent interpenetrations (that lead to aliasing and non differentiability along the visible self-intersections)
* phong shading
* gradients backward pass for perspective-correct texture mapping
* gradients backward pass for perspective-correct vertex attributes/color interpolation
* texture mip-mapping (would require [trilinear filtering](https://en.wikipedia.org/wiki/Trilinear_filtering) to make it smoother and differentiable)
* shadow casting (making it differentiable would be challenging)
 
### Using texture triangles

Keeping the rendering differentiable everywhere when using texture is challenging: if you use textured triangles you will need to make sure there are no adjacent triangles in the 3D mesh that are simultaneously visible while being disconnected in the UV map, i.e. that there is no visible seam. Otherwise the rendering will not in general be continuous with respect to the 3D vertices positions due to the texture discontinuity along the seam. Depending on the shape of your object, you might not be able to define a continuous UV mapping over the entire mesh and you will need to define the UV texture coordinates in a very specific manner described in Figure 3 in [1], with some constraints on the texture intensities so that the continuity of the rendering is still guaranteed along edges between disconnected triangles in the UV map after texture bilinear interpolation. Note that an improved version of that approach is also described in [8].

# Installation
## Python

For python 3.6,3.7 and python 3.8 under Windows and Linux you can use 

	pip install deodr

Otherwise or if that does not work you can try

	pip install git+https://github.com/martinResearch/DEODR.git
	
## Matlab
Simply download the zip file, decompress it, run compile.m.

For the hand fitting example you will also need to download my Matlab automatic differentiation toolbox from [here](https://github.com/martinResearch/MatlabAutoDiff) 
add the decompressed folder in your matlab path

# Examples

## Iterative Mesh fitting in Python 
 
Example of fitting a hand mesh to a depth sensor image [*deodr/examples/depth_image_hand_fitting.py*](deodr/examples/depth_image_hand_fitting.py)

 ![animation](./images/python_depth_hand.gif)

Example of fitting a hand mesh to a RGB sensor image [*deodr/examples/rgb_image_hand_fitting.py*](deodr/examples/rgb_image_hand_fitting.py)

 ![animation](./images/python_rgb_hand.gif)

Example of fitting a hand mesh to several RGB sensor images [*deodr/examples/rgb_multiview_hand.py*](deodr/examples/rgb_multiview_hand.py)

 ![animation](./images/multiview.gif)


## iterative mesh fitting in Matlab
Example of a simple triangle soup fitting [*Matlab/examples/triangle_soup_fitting.m*](Matlab/examples/triangle_soup_fitting.m) 

![animation](./images/soup_fitting.gif)

Example of fitting a hand mesh to a RGB sensor image [*Matlab/examples/hand_fitting.m*](Matlab/examples/hand_fitting.m).
For this example you will also need to download the automatic differentiation toolbox from [https://github.com/martinResearch/MatlabAutoDiff](https://github.com/martinResearch/MatlabAutoDiff) 

![animation](./images/hand_fitting.gif)



 
# Equations

This code implements the core of the differentiable renderer described in [1,2] and has been mostly written in 2008-2009. It is anterior to OpenDR and is to my knowledge the first differentiable renderer that deals with vertices displacements to appear in the literature.
It renders a set of triangles with either a texture that is bilinearly interpolated and shaded or with interpolated RGB colours. In contrast with most renderers, the intensity of each pixel in the rendered image is continuous and differentiable with respect to the vertices positions even along occlusion boundaries. This is achieved by using a differentiable antialiasing method called *Discontinuity-Edge-Overdraw* [3] that progressively blends the colour of the front triangle with the back triangle along occlusion boundaries, using a linear combination of the front and back triangles with a mixing coefficient that varies continuously as the reprojected vertices move in the image (see [1,2] for more details). This allows us to capture the effect of change of visibility along occlusion boundaries in the gradient of the loss in a principled manner by simply applying the chain rule of derivatives to our differentiable rendering function. Note that this code does not provide explicitly the sparse Jacobian of the rendering function (where each row would correspond to the color intensity of a pixel of the rendered image, like done in [4]) but it provides the vector-Jacobian product operator, which corresponds to the backward function in PyTorch.

This can be used to do efficient analysis-by-synthesis computer vision by minimizing the function E that corresponds to the sum or the squared pixel intensities differences between a rendered image and a reference observed image I<sub>o</sub> with respect to the scene parameters we aim to estimate.

$$E(V)=\sum_{ij} (I(i,j,V)-I_o(i,j))^2$$


Using D(i,j) as the *adjoint* i.e. derivative of the error (for example the squared residual) between observed pixel intensity and synthetized pixel intensity at location (i,j) 

$$ D(i,j)=\partial E/\partial I(i,j))$$

We can use DEODR to obtain the gradient with respect to the 2D vertices locations  and their colors i.e :
 
$$\partial E/\partial V_k =\sum_{ij} D(i,j)(\partial I(i,j)/\partial V_k)$$
 
and 

$$\partial E/\partial C_k = \sum_{ij} D(i,j)(\partial I(i,j)/\partial C_k)$$


In combination with an automatic differentiation tool, this core function allows one to obtain the gradient of 
the error function with respect to the parameters of a complex 3D scene one aims to estimate.

The rendering function implemented in C++ can draw an image given a list of 2D projected triangles with the associated vertices depths (i.e 2.5D scene), where each triangle can have 

* a linearly interpolated color between its three extremities 
* a texture with linear texture mapping (no perspective-correct texture mapping yet but it will lead to noticeable bias only for large triangle that are no fronto-parallel)
* a texture combined with shading interpolated linearly (gouraud shading)

We provide a functions in Matlab and Python to obtain the 2.5D representation of the scene from a textured 3D mesh, a camera and a simple lighting model. This function is used in the example  in which we fit a 3D hand model to an image. 
We kept this function as minimalist as possible as we did not intend to rewrite an entire rendering pipeline but to focus on the part that is difficult to differentiate and that cannot be differentiated easily using automatic differentiation.

Our code provides two methods to handle discontinuities at the occlusion boundaries

* the first method consists in antialiasing the synthetic image before comparing it to the observed image. 
* the second method consists in antialising the squared residual between the observed image and the synthesized one, and corresponds to the method described in [1]. Note that antialiasing the residual (instead of the squared residual) is equivalent to do antialiasing on the synthetized image and then subtract the observed image.

The choice of the method is done through the Boolean parameter *antialiaseError*. Both approaches lead to a differentiable error function after summation of the residuals over the pixels and both lead to similar gradients. The difference is subtle and is only noticeable at the borders after convergence on synthetic antialiased data. The first methods can potentially provide more flexibility for the design of the error function as one can for example use a non-local image loss by comparing image moments instead of comparing pixel per pixel.

**Note:** In order to keep the code minimal and well documented, I decided not to provide here the Matlab code to model the articulated hand and the code to update the texture image from observation used in [1]. The hand fitting example provided here does not relies on a underlying skeleton but on a regularization term that penalizes non-rigid deformations. Some Matlab code for Linear Blend Skinning can be found [here](http://uk.mathworks.com/matlabcentral/fileexchange/43039-linear-blend-skinning/). Using a Matlab implementation of the skinning equations would allow the use of the Matlab automatic differentiation toolbox provided [here](https://github.com/martinResearch/MatlabAutoDiff) to compute the Jacobian of the vertices positions with respect to the hand pose parameters.

# Conventions

## Pixel coordinates: 
 
If integer_pixel_centers is True (default) then pixel centers are at integer coordinates with
* upper left at (0, 0)
* upper right at (width - 1, 0)
* lower left at (0, height - 1)
* lower right at  (width - 1, height - 1)

If integer_pixel_centers is False, then pixel centers are at half-integer coordinates with
* upper left at (0.5, 0.5)
* upper right at (width - 0.5, 0.5)
* lower left at (0.5, height - 0.5)
* lower right at  (width -0.5, height - 0.5)
  
According to [this page](https://www.realtimerendering.com/blog/the-center-of-the-pixel-is-0-50-5/), OpengGL has always used upper left pixel center at (0.5, 0.5) while Direct3D was using pixel center at (0,0) before version 10 and switched to (0.5,0.5) at version 10.

## Texel coordinates: 

Unlike in OpenGL Texel (texture pixel) center are at integer coordinates and origin in in the upper left corner of the texture image.
The coordinate of the upper left texture pixel center (texel) is (0, 0). The color of the texture bilinearly sampled at float position (0.0,0.0) is texture[0, 0].
The value of the texture bilinearly sampled at float position (0.5,0.5) is equal to the average (texture[0, 0] + texture[0, 1] + texture[1, 0] + texture[1, 1])/4


# TO DO

* add support for multiple meshes in the scene
* add support for multiple lights in the scene
* write more unit tests
* add possibility to provide the camera parameters using OpenGL parameterization
* write pure C++ only rendering example
* write pure C++ only mesh fitting example
* accelerate C++ code using SIMD instruction and multithreading
* add automatic texture reparameterization and resampling to avoid texture discontinuities (see section on texture) 
* add phong shading

# License

[BSD 2-clause "Simplified" license](licence.txt).

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

## Projects that use DEODR

* [TraceArmature](https://github.com/WilliamRodriguez42/TraceArmature).  Set of python scripts that allow for high fidelity motion capture through the use of AI pose estimation (using METRABS), fiducial markers, VR body trackers, and optional hand annotations.

Please let me know if you found DEODR useful by adding a comment in [here](https://github.com/martinResearch/DEODR/issues/240).

# Alternatives 

* [**SoftRas**](https://github.com/ShichenLiu/SoftRas) (MIT Licence). Method published in [9]. This method consists in a differentiable render with a differentiable forward pass. To my knowledge, this is at the moment the only method besides ours that has a differentiable forward pass and that computes the exact gradient of the forward pass in the backward pass. 
* [**OpenDR**](https://github.com/mattloper/opendr/wiki) [4] (MIT Licence) is an open source differentiable renderer written in python and make publicly available in 2014. OpenDR calls OpenGL and relies an a python automatic differentiation toolbox by the same author called [chumpy](https://github.com/mattloper/chumpy). Like in our code OpenDR uses a intermediate 2.5D representation of the scene using a set of 2D projected triangles. In contrast to our code OpenDR does not provide a continuous loss function as there is not continuous antialiasing formulation at the occlusion boundaries and the minimised function will have jumps when a pixel at the boundary switch between the front of back object. By providing a continuous differentiable error function using edge-overdraw antialiasing and its exact gradient, our method can lead to better a convergence of continuous optimisation methods..

* [**DIRT**](https://github.com/pmh47/dirt) (MIT licence) is an open source differentiable renderer that uses approximations in the gradient computation similar OpenDR but that is interfaced with tensorflow. It makes considerable effort to return correctly-behaving derivatives even in cases of self-occlusion, where most other differentiable renderers can fail. 

* [**Neural 3D Mesh Renderer**](https://github.com/hiroharu-kato/neural_renderer) (MIT Licence). Method published in [6]. This method consists in a differentiable render whose gradients are designed to be used in neural networks. 
 
* [**tf\_mesh\_renderer**](https://github.com/google/tf_mesh_renderer) (Apache License 2.0). A differentiable, 3D mesh renderer using TensorFlow. Unlike other differentiable renderer it does not provides suppport for occlusion boundaries in the gradient computation and thus is inadequate for many applications.

* Code accompanying the paper [7] [github](https://github.com/ndrplz/differentiable-renderer). It renders only silhouettes. 

* [**redner**](https://github.com/BachiLi/redner) Method published in [10]. It is a differentiable path-tracer that can propagate gradients through indirect illumination. 

* [**Differentiable Surface Splatting**](https://github.com/yifita/DSS) Method publihed in [11]. It is a differentiable implementation of the surface splatting method that allows to render point clouds. The advantage of this approach over mesh based methods is that there is no predefined connectivity associated to the set of points, which allows topological changes during minimization.

* [**DIB-Render**](https://github.com/nv-tlabs/DIB-R) Method published in [12]. This method removes discontinuies along the projected mesh and background boundary (external silhouette) by rendering background pixels colors using a weighted sum of the background colour and nearby projected faces. However, unlike our method, it does not remove discontinuities along self-occlusions.

* [**Differentiable path tracing**](https://github.com/mitsuba-renderer/mitsuba2) Method published in [13]

* [**PyTorch3D's renderer**](https://pytorch3d.org/docs/renderer). The method implemented keeps a list of the K nearest faces intersecting the ray corresponding to each pixel in the image (as opposed to traditional rasterization which returns only the index of the closest face in the mesh per pixel). The top K face properties can then be aggregated using different methods (such as the sigmoid/softmax approach proposed by Li et at in SoftRasterizer). Note however that the K face selection is not a continuous operation and thus may potentially lead to discontinuities in the rendered pixel intensities with respect to the shape parameters.  

* [**Tensorflow graphics**](https://github.com/tensorflow/graphics). Library from google that allows differentiable rendering. At the date of march 2020 discontinuities along occlusion boundaries are not handled yet in a differentiable manner. As a result fitting using a loss defined with the rendered image may not converge well in some scenario.

* [**SDFDiff**](https://github.com/YueJiang-nj/CVPR2020-SDFDiff).[15] Differentiable Implicit surface rendering using signed distance functions. This approach seems not to deal with depth discontinuity along the silhouette and self occlusions in a differentiable way, and thus the change in visibility in these locations is not captured in the back-propagated gradients which will hamper convergence, especially in scenarios where the silhouette is an important signal for the fitting..

* [**DIST**](https://github.com/B1ueber2y/DIST-Renderer)[16] Differentiable Implicit surface rendering using signed distance functions. The depth discontinuity along the object/background boundary is taken into account in the computation of the silhouette mask only, and is not taken into account in the rendered color, depth or normal images. This could hamper convergence of the surface fitting methods that uses these images in the loss for concave objects that exhibit self occlusion in the chosen camera view point(s). 

* [**Nvdiffrast**](https://github.com/NVlabs/nvdiffrast)[17] Fast and modular differentiable renderer implemented using OpenGL and cuda with Pytorch and tensoflow interfaces. The method uses deferred rendering which yields great flexibility for the user by allowing the user to write a custom pixel shader using pytorch . An antialiasing step is applied after deferred rendering in order to blend the colors along self-occlusion edges using weights that depend on the distance of the adjacent pixel centers to the line segment. The colors used for blending of along the self occlusion edges  are extracted from the nearest points with integer coordinates in the adjacent triangles. This is not a continuous operation and thus may result in small temporal color jumps along the self-occlusion when the edges line segment crosses a pixel center. This may degrade the quality of the gradients as a first order approximations of the change in the loss function w.r.t scene parameters. Moreover the method used to detect self occlusion edges is done in the image space using faces indices which is not very reliable when triangles are small, which introduces noise in the gradient that might be biased. In contrast, our anti-aliasing method uses the color of the nearest point on the occlusing triangle edge (euclidean projection), which does not lead to discontinuity when edges cross pixel centers, and our silhouette edges detection is done in the object space and thus is reliable regardless of the triangles size during rasterization.
# References
[1] *Model-based 3D Hand Pose Estimation from Monocular Video.* M. de la Gorce, N. Paragios and David Fleet. PAMI 2011 [paper](http://www.cs.toronto.edu/~fleet/research/Papers/deLaGorcePAMI2011.pdf)

[2] *Model-based 3D Hand Pose Estimation from Monocular Video*. M. de la Gorce. PhD thesis. Ecole centralde de Paris 2009.
[paper](https://tel.archives-ouvertes.fr/tel-00619637)

[3] *Discontinuity edge overdraw* P.V. Sander and H. Hoppe, J.Snyder and S.J. Gortler. SI3D 2001 [paper](http://hhoppe.com/overdraw.pdf)

[4] *OpenDR: An Approximate Differentiable Renderer* Loper, Matthew M. and Black, Michael J. ECCV 2014 [paper](http://files.is.tue.mpg.de/black/papers/OpenDR.pdf)
[code](https://github.com/mattloper/opendr) [online documentation](https://github.com/mattloper/opendr/wiki)

[5] *A Morphable Model For The Synthesis Of 3D Faces*. Volker Blanz and Thomas Vetter. SIGGRAPH 99. [paper](https://gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf)

[6] *Neural 3D Mesh Renderer*. Hiroharu Kato and Yoshitaka Ushiku and Tatsuya Harada.CoRR 2017 [paper](https://arxiv.org/pdf/1711.07566.pdf)

[7] *End-to-end 6-DoF Object Pose Estimation through Differentiable Rasterization* Andrea Palazzi, Luca Bergamini, Simone Calderara, Rita Cucchiara. Second Workshop on 3D Reconstruction Meets Semantics (3DRMS) at ECCVW 2018. [paper](https://iris.unimore.it/retrieve/handle/11380/1167726/205862/palazzi_eccvw.pdf)

[8] *Mesh Color textures* Cem Yuksel. Proceedings of High Performance Graphics 2017. [paper](http://www.cemyuksel.com/research/meshcolors/mesh_color_textures.pdf)

[9] *Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning*. Shichen Liu, Tianye Li, Weikai Chen and  Hao Li. ICCV 2019. [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf)

[10]  *Differentiable Monte Carlo Ray Tracing through Edge Sampling*. zu-Mao Li, Miika Aittala, Frédo Durand, Jaakko Lehtinen.  SIGGRAPH Asia 2018. [project page](https://people.csail.mit.edu/tzumao/diffrt/)

[11] *Differentiable Surface Splatting for Point-based Geometry Processing*. Felice Yifan, Serena Wang, Shihao Wu, Cengiz Oztireli and Olga Sorkine-Hornung. SIGGRAPH Asia 2019. [paper and video](https://yifita.github.io/publication/dss/)

[12] *Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer* 
Wenzheng Chen, Jun Gao, Huan Ling, Edward J. Smith, Jaakko Lehtinen, Alec Jacobson, Sanja Fidler. NeurIPS 2019. [paper](https://nv-tlabs.github.io/DIB-R/files/diff_shader.pdf)

[13] *Reparameterizing discontinuous integrands for differentiable rendering*. Guillaume Loubet, Nicolas Holzschuch and Wenzel Jakob. SIGGRAPH Asia 2019. [project page](https://rgl.epfl.ch/publications/Loubet2019Reparameterizing)

[14] *TensorFlow Graphics: Computer Graphics Meets Deep Learning*. Valentin, Julien and Keskin, Cem and Pidlypenskyi, Pavel and Makadia, Ameesh and Sud, Avneesh and Bouaziz, Sofien. 2019

[15] *SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization*. Yue Jiang, Dantong Ji, Zhizhong Han, Matthias Zwicker. CVPR 2020. [code](https://github.com/YueJiang-nj/CVPR2020-SDFDiff)
 
[16] *DIST: Rendering Deep Implicit Signed Distance Function with Differentiable Sphere Tracing*. Shaohui Liu1, Yinda Zhang, Songyou Peng, Boxin Shi, Marc Pollefeys, Zhaopeng Cui. CVPR 2020. [code](https://github.com/B1ueber2y/DIST-Renderer)

[17] *Modular Primitives for High-Performance Differentiable Rendering*. Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, Timo Aila. [paper](https://arxiv.org/abs/2011.03277). 
