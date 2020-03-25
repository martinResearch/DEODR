function M=mesh_normals_and_surfaces(M)
% compute the surface and normals of the M.F of a mesh and first order
% derivatives

U=M.V(:,M.F(2,:))-M.V(:,M.F(1,:));
V=M.V(:,M.F(3,:))-M.V(:,M.F(1,:));

U1=U(1,:);
U2=U(2,:);
U3=U(3,:);
V1=V(1,:);
V2=V(2,:);
V3=V(3,:);



NSX=U2.*V3-U3.*V2;
NSY =U3.*V1-U1.*V3;
NSZ= U1.*V2-U2.*V1;

NS=0.5*[NSX;NSY;NSZ];
            
   
M.Surfaces=normes(NS,1);
M.NormalsF=NS./(ones(3,1)*M.Surfaces);

%tmp=M._vertices_faces*M.NormalsF';
MeanNormals=( M.NormalsF*M.vertices_faces');  
M.NormalsV=MeanNormals.*(ones(3,1)*(1./normes(MeanNormals,1)));
M.NormalsV_B=zeros(size( M.NormalsV));

function N=normes(M,dim)
N=sqrt(sum(M.^2,dim));

