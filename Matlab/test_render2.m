% load obj using third party library and convert in our format
addpath(genpath('.'))
h=loadawobj('../data/hand.obj');
M.F=flipud(h.f3);
M.V=h.v;
M=mesh_adjacencies(M);

% set colors
M.colors=[200;100;100]*ones(1,size(M.V,2))/255;

backgroundColor=[0.3,0.5,0.7];

% setup camera
SizeW=300;
SizeH=300;
objectCenter=mean(M.V,2);
objectRadius=max(std(M.V,[],2));
cameraCenter=objectCenter+[0,0,7]'*objectRadius;
focal=600;
R=[0,-1,0;-1,0,0;0,0,-1];
T=-R'*cameraCenter;
CameraMatrix=[focal,0,SizeW/2;0,focal,SizeH/2;0,0,1]*[R,T];

%setup light parameters
ligthDirectional=[0.1,-0.5,-0.5];
ambiantLight=0.3;

% conversion from 3D mesh to triangle 2.5D soup
scene=mesh2scene(M,CameraMatrix,ligthDirectional,ambiantLight,SizeH,SizeW);

% adding a background image
scene.background=repmat(backgroundColor(:),1,scene.SizeH,scene.SizeW);

sigma=1;% edge antialising width

% render the image

tic
[Abufferc2,Zbuffer]=render(scene,sigma,0);
toc
% display
figure(1);
d=Zbuffer;
d=(d-min(d(:)))/(max(d(d<inf))-min(d(:)));
image2=cat(2,permute(abs(Abufferc2),[2,3,1]),repmat(1-d,[1,1,3]));
imshow(image2);
imwrite(image2,'./images/example1.png')