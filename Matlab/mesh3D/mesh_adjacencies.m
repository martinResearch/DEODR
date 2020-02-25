function M=mesh_adjacencies(M)
% compute some arrays and sparse matrices that facilitate the mesh manipulation
% and that are independant on the vertices positions 
%
% edges_vertices(k,:) : 
%   two indices corresponding to the two vertices
%   linked by the edge k
%
% vertices_edges : sparse matrix 
%   vertices_edges(i,j)==1 means that vertex i is one of the two vertices of
%   edge j
%
% edges_faces : sparse matrix 
%   edges_faces(i,j)==k means that the edge i is one  the k'th  edges of the
%   face j  (0 if it is not an edge of the face) 
%   the edges are listed as follow (a,b),(b,c),(c,a)
%
% adjacency_faces : sparse adjacency matrix of the dual graph
%   adjacency_faces(i,j)==1 means that the faces i and j have on edge in common
%   adjacency_faces(i,i)==3 for all faces (3 edges in common with itself) 
%   adjacency_faces =(edges_faces~=0)'*(edges_faces~=0);
%
% adjacency_vertices : sparse  adjacency matrix of the graph
% adjacency_vertices(i,j)==1 means that the vertex i and j are linked by an
% edge
% adjacency_vertices(i,i)==0 for all vertices
%
% oriented_edge : sparse matrix
% oriented_edge(i,j)==k means that Vertex i and j are directly 
% consecutives in the list of vertice of face k 
%
% degree_v_e : 
% degree_v_e(i)=j means that the vertex i appears in j edges
%
% degree_v_f :
% degree_v_f(i)=j means that the vertex i appears in j Faces
%

M.nb_vertices=size(M.V,2);
M.nb_faces=size(M.F,2);
M.vertices_faces=sparse(M.nb_vertices,M.nb_faces);
M.degree_v_f=zeros(M.nb_vertices,1);

for k=1:M.nb_faces
    M.vertices_faces(M.F(:,k),k)=1;
end

for i=1:M.nb_faces
    M.degree_v_f(M.F(:,i))=M.degree_v_f(M.F(:,i))+1;
end



Mtemp=sparse(M.nb_vertices^2,M.nb_faces);
for f=1:M.nb_faces
 Mtemp (idedge(M.F(1,f),M.F(2,f)),f)=1;
 Mtemp (idedge(M.F(2,f),M.F(3,f)),f)=2;
 Mtemp (idedge(M.F(3,f),M.F(1,f)),f)=3;
end

listEdgeId=find(any(Mtemp,2));

M.nb_edges=numel(listEdgeId);
M.edges_vertices=inv_idedge(listEdgeId);

MtempT=Mtemp';
M.edges_faces=MtempT(:,listEdgeId)';
[i,j,v]=find(M.edges_faces);
M.Faces_edges=full(sparse(j,v,i));

M.degree_e=full(sum(M.edges_faces>0,2));
M.closed=all(M.degree_e==2);
M.vertices_edges=sparse(M.edges_vertices(:),[1:M.nb_edges,1:M.nb_edges],1);
M.ajacency_faces=double(M.edges_faces~=0)'*double(M.edges_faces~=0);
M.adjacency_vertices=sparse([M.edges_vertices(:,1);M.edges_vertices(:,2)],[M.edges_vertices(:,2);M.edges_vertices(:,1)],ones(2*size(M.edges_vertices,1),1));
M.degree_v_e=full(sum(M.adjacency_vertices));

% compute list of vertice for fast computation of the willmore energy
% use to compute concavity of edges , but could be avoid
tmp=M.edges_faces';
willmore.i=zeros(1,M.nb_edges);
willmore.j=zeros(1,M.nb_edges);
willmore.k=zeros(1,M.nb_edges);
willmore.l=zeros(1,M.nb_edges);

for ide=1:M.nb_edges
  
    willmore.i(ide)=M.edges_vertices(ide,1);
    willmore.j(ide)=M.edges_vertices(ide,2);
    
    % for each adjacent face, get the oposite edge
    [adjFaces,~,idEdgeInFace]=find(tmp(:,ide));
    if M.F(mod(idEdgeInFace(1)+2,3)+1,adjFaces(1))==M.edges_vertices(ide,1)
       leftface=1;
       rightface=2;
    else
       leftface=2;
       rightface=1;
    end

    idOpositeInFace=mod(idEdgeInFace+1,3)+1;
    
    if numel(adjFaces)~=2
    %    warning('this is not define for borders yet')
    else
    willmore.k(ide)=M.F(idOpositeInFace(leftface),adjFaces(leftface));
    willmore.l(ide)=M.F(idOpositeInFace(rightface),adjFaces(rightface));
    end    
end
M.willmoreLists=willmore;

  function id=idedge(a,b)
        id= max(a,b)+(min(a,b)-1)*M.nb_vertices;
    end
% inverse of the function idege
    function [a,b]=inv_idedge(id)
        a=mod(id-1,M.nb_vertices)+1;
        b=(id-a)/M.nb_vertices+1;
        if nargout<2
        a=[a,b];
        end    
    end

end

  
