function hand_fitting()

addpath(genpath('..'))

% loading the image and creating mask to discard pixels we don't want to us
% in the loss
hand_image=double(imread('hand.png'))/255;
mask=conv2(max(hand_image,[],3)>=1 ,ones(3,3) ,'same')==0;

% fixing hand color and backround color
background_color=[0.5,0.6,.7]';
hand_color=[0.4,0.3,0.25]';

% loading the hand 3D mesh
objfile='hand.obj';
h=loadawobj(objfile);
faces = flipud(h.f3);% reorient face
vertices = h.v;

vertices_colors = hand_color*ones(1,size(vertices,2));

% choosing the position of the camera so that the we get a good
% initialization for the fitter
width=size(hand_image,2);
height=size(hand_image,1);
object_center=mean(vertices,2);
objectRadius=max(std(vertices,1,2));
camera_center=object_center+[0,0,9]'*objectRadius;
focal=2*width;
R=[1,0,0;0,-1,0;0,0,-1];
T=-R'*camera_center;
CameraMatrix=[focal,0,width/2;0,focal,height/2;0,0,1]*[R,T];

lights.ligth_directional=1*[0.1,0.5,0.4];
lights.ambiant_light=0.6;

options.sigma = 1;% edge antialising width
options.save_images = true;
options.display = 1;
options.antialiaseError = 0;
options.gamma = 0.01;
options.alpha = 0.0;
options.beta = 0.01;
options.inertia = 0.95;
options.damping = 0.2;
options.method = 'heavyBall';
options.nb_max_iter = 100;
options.iter_images_folder='iterations';
options.save_gif = true;

options.cregu=1000; % mesh rigidity regularisation coefficient
mesh_fitting(hand_image, mask, vertices, faces,vertices_colors, background_color,lights,CameraMatrix,options)
