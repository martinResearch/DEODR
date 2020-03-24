% load obj using third party library and convert in our format
addpath(genpath('.'))
h=loadawobj('../data/hand.obj');
M.F=flipud(h.f3);
M.V=h.v;
M=mesh_adjacencies(M);

% set colors
M.colors=[200;100;100]*ones(1,size(M.V,2))/255;

background_color=[0.3,0.5,0.7];

% setup camera
width=300;
height=300;
object_center=mean(M.V,2);
objectRadius=max(std(M.V,[],2));
camera_center=object_center+[0,0,7]'*objectRadius;
focal=600;
R=[0,-1,0;-1,0,0;0,0,-1];
T=-R'*camera_center;
CameraMatrix=[focal,0,width/2;0,focal,height/2;0,0,1]*[R,T];

%setup light parameters
light_directional=[-0.1,0.5,0.5];
light_ambient=0.3;

% conversion from 3D mesh to triangle 2.5D soup
scene=mesh2scene(M,CameraMatrix,light_directional,light_ambient,height,width);

% adding a background image
scene.background=repmat(background_color(:),1,scene.height,scene.width);

sigma=1;% edge antialising width

% render the image

tic
[imagec2,z_buffer]=render(scene,sigma,0);
toc
% display
figure(1);
d=z_buffer;
d=(d-min(d(:)))/(max(d(d<inf))-min(d(:)));
image2=cat(2,permute(abs(imagec2),[2,3,1]),repmat(1-d,[1,1,3]));
imshow(image2);
imwrite(image2,'./images/example1.png')