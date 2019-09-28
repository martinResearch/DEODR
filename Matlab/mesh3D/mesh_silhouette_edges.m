function M=mesh_silhouette_edges(M,Viewpoint)
% find edges on the silhouette given a View point
% adjacent_visible_faces provide the index of the visible face adgacent to
% the edge , this allows the get color attributes from the face to
% antialise the edge

% for each face, compute cross product between normals and rays going
% from surface to the viewpoint.
% for each face we arbitraly choose the first vertex while defining the ray
% we could chose the barycenter of the face but this doesn't change the
% sign of the cross product

% the detection of silhouette edges can be accelerated using some
% randomized search as done in "Real-Time Nonphtorealistic rendering"
% by Lee Markosian & all but this is does not garanties to find all
% edges, which is not suitable 
% An other approach could be to cluster (in a hierarchy?) adjacent faces that have similar
% normals and encode a bound on the normals variation within the cluster
% in order to derive rigorous bound that allow to dicard large set of edges

if not(exist('Viewpoint'))
    error('You should provide a viewpoint for silhouette computation')
end

if any(size(Viewpoint)~=[3,1])
    error('Viewpoint should of size [3,1]')
end


% for each face compute signe of the volume
% of the thetrahedron made of the 3 vertices and the viewpoint
% sNR=zeros(1,M.nbF);
% for k=1:M.nbF
%     sNR(k)=det(M.V(:,M.F(:,k))-repmat(Viewpoint,[1,3]))<0;
% end

visible_bool=sum(M.NormalsF.*(repmat(Viewpoint,[1,M.nbF])-M.V(:,M.F(1,:))),1)>0;


% find edges with a single neighboring face oriented toward the camera :

edge_bool =((double(M.Edges_Faces>0)*visible_bool')==1)';
adjacent_visible_faces=((M.Edges_Faces>0)*((1:M.nbF).*(visible_bool))')';
edges_list=find(edge_bool);
adjacent_visible_faces=adjacent_visible_faces(edges_list);


% Compute which of the silhouette edges are convex and which are concave
% this could be compute only when V are modified
M.edge_bool=edge_bool;
M.edges_list=edges_list;
M.adjacent_visible_faces=adjacent_visible_faces;
M.visible_bool=visible_bool;

% convex_bool=false(1,M.nbE);
% will=M.willmoreLists;
% V=M.V;
% ide=edges_list;
% vi=V(:,will.i(ide));
% vj=V(:,will.j(ide));
% vk=V(:,will.k(ide));
% vl=V(:,will.l(ide));
% convex_bool(ide)=sum(cross(vj-vi,vk-vi).*(vl-vi),1)<0;
% M.convex_bool=convex_bool;




