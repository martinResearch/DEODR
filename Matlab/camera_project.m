function [P2D,depths]=camera_project(cameraMatrix,P3D)       

r=cameraMatrix*[P3D;ones(1,size(P3D,2))];
depths=r(3,:);    
P2D=(r(1:2,:)./([1;1]*depths));
