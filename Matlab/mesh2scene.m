function scene=mesh2scene(M,CameraMatrix,light_directional,light_ambient,height,width,getSilhouette)

if nargin<7
    getSilhouette=true;
end
[P2D,depths]=camera_project(CameraMatrix,M.V);
M=mesh_normals_and_surfaces(M);
camera_center=-(CameraMatrix(1:3,1:3))\CameraMatrix(:,4);
if  getSilhouette
    M=mesh_silhouette_edges(M,camera_center);
end
luminosity=max(0,(-light_directional*M.NormalsV))+light_ambient;
colorsV=M.colors.*([1;1;1]*luminosity);
Ntri=M.nb_faces;

scene.ij=flipud(P2D);%fliping i and j coordinate because matlab is not consistent wit numpy
scene.colors=colorsV;
scene.depths=depths;
if  getSilhouette
    scene.edgeflags=flipud(M.edge_bool(M.Faces_edges'));% flipping to reflect flip on ij
end

scene.faces = uint32( M.F([1,3,2],:))-1;% changing order to reflect flip on ij
scene.faces_uv = scene.faces;
scene.uv=zeros(2,M.nb_vertices);
scene.textured=logical(zeros(1,Ntri));
scene.shade=zeros(1,M.nb_vertices);
scene.height=height;
scene.width=width;
scene.shaded=logical(zeros(1,Ntri));
scene.texture=zeros(3,10,10);

