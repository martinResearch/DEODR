function scene=mesh2scene(M,CameraMatrix,ligthDirectional,ambiantLight,SizeH,SizeW,getSilhouette)

if nargin<7
    getSilhouette=true;
end
[P2D,depths]=camera_project(CameraMatrix,M.V);
M=mesh_normals_and_surfaces(M);
cameraCenter=-(CameraMatrix(1:3,1:3))\CameraMatrix(:,4);
if  getSilhouette
M=mesh_silhouette_edges(M,cameraCenter);
end
luminosity=max(0,(ligthDirectional*M.NormalsV))+ambiantLight;
colorsV=M.colors.*([1;1;1]*luminosity);
Ntri=M.nbF;

scene.ij=reshape(P2D([2,1],M.F),2,3,[]);
scene.colors=reshape(colorsV(:,M.F),3,3,[]);
scene.depths=reshape(depths(M.F),1,3,[]);
if  getSilhouette
scene.edgeflags=M.edge_bool(M.Faces_edges');
end
scene.uv=zeros(2,3,Ntri);
scene.textured=logical(zeros(1,Ntri));
scene.shade=zeros(1,3,Ntri);
scene.SizeH=SizeH;
scene.SizeW=SizeW;
scene.shaded=logical(zeros(1,Ntri));
scene.material=zeros(3,10,10)  ;
