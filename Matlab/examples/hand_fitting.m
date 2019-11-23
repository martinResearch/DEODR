function hand_fitting()

addpath(genpath('..'))

% loading the image and creating mask to discard pixels we don't want to us
% in the loss
handImage=double(imread('hand.png'))/255;
mask=conv2(max(handImage,[],3)>=1 ,ones(3,3) ,'same')==0;

% fixing hand color and backround color
backgroundColor=[0.5,0.6,.7]';
handColor=[0.4,0.3,0.25]';

% loading the hand 3D mesh
objfile='hand.obj';
h=loadawobj(objfile);
faces = h.f3;
vertices = h.v;
vertices_colors = handColor*ones(1,size(vertices,2));

% choosing the position of the camera so that the we get a good
% initialization for the fitter
SizeW=size(handImage,2);
SizeH=size(handImage,1);
objectCenter=mean(vertices,2);
objectRadius=max(std(vertices,1,2));
cameraCenter=objectCenter+[0,0,9]'*objectRadius;
focal=2*SizeW;
R=[1,0,0;0,-1,0;0,0,-1];
T=-R'*cameraCenter;
CameraMatrix=[focal,0,SizeW/2;0,focal,SizeH/2;0,0,1]*[R,T];

lights.ligthDirectional=1*[0.1,0.5,0.4];
lights.ambiantLight=0.6;

options.sigma = 1;% edge antialising width
options.save_images = true;
options.display = 1;
options.antialiaseError = 0;
options.gamma = 0.01;
options.alpha = 0.0;
options.beta = 0.01;
options.inertia = 0.9;
options.damping = 0.1;
options.method = 'heavyBall';
options.nbMaxIter = 100;
options.iter_images_folder='iterations';
options.save_gif = true;

options.cregu=2000; % mesh rigidity regularisation coefficient
mesh_fitting(handImage, mask, vertices, faces,vertices_colors, backgroundColor,lights,CameraMatrix,options)
