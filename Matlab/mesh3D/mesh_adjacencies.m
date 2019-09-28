function M=mesh_adjacencies(M)
% compute some arrays and sparse matrices that facilitate the mesh manipulation
% and that are independant on the vertices positions 
%
% Edges_Vertices(k,:) : 
%   two indices corresponding to the two vertices
%   linked by the edge k
%
% Vertices_Edges : sparse matrix 
%   Vertices_Edges(i,j)==1 means that vertex i is one of the two vertices of
%   edge j
%
% Edges_Faces : sparse matrix 
%   Edges_Faces(i,j)==k means that the edge i is one  the k'th  edges of the
%   face j  (0 if it is not an edge of the face) 
%   the edges are listed as follow (a,b),(b,c),(c,a)
%
% AdjacencyFaces : sparse adjacency matrix of the dual graph
%   AdjacencyFaces(i,j)==1 means that the faces i and j have on edge in common
%   AdjacencyFaces(i,i)==3 for all faces (3 edges in common with itself) 
%   AdjacencyFaces =(Edges_Faces~=0)'*(Edges_Faces~=0);
%
% AdjacencyVertices : sparse  adjacency matrix of the graph
% AdjacencyVertices(i,j)==1 means that the vertex i and j are linked by an
% edge
% AdjacencyVertices(i,i)==0 for all vertices
%
% OrientedEdge : sparse matrix
% OrientedEdge(i,j)==k means that Vertex i and j are directly 
% consecutives in the list of vertice of face k 
%
% DegreeVE : 
% DegreeVE(i)=j means that the vertex i appears in j edges
%
% DegreeVF :
% DegreeVF(i)=j means that the vertex i appears in j Faces
%

M.nbV=size(M.V,2);
M.nbF=size(M.F,2);
M.Vertices_Faces=sparse(M.nbV,M.nbF);
M.DegreeVF=zeros(M.nbV,1);

for k=1:M.nbF
    M.Vertices_Faces(M.F(:,k),k)=1;
end

for i=1:M.nbF
    M.DegreeVF(M.F(:,i))=M.DegreeVF(M.F(:,i))+1;
end



Mtemp=sparse(M.nbV^2,M.nbF);
for f=1:M.nbF
 Mtemp (idedge(M.F(1,f),M.F(2,f)),f)=1;
 Mtemp (idedge(M.F(2,f),M.F(3,f)),f)=2;
 Mtemp (idedge(M.F(3,f),M.F(1,f)),f)=3;
end

listEdgeId=find(any(Mtemp,2));

M.nbE=numel(listEdgeId);
M.Edges_Vertices=inv_idedge(listEdgeId);

MtempT=Mtemp';
M.Edges_Faces=MtempT(:,listEdgeId)';
[i,j,v]=find(M.Edges_Faces);
M.Faces_edges=full(sparse(j,v,i));

M.DegreeE=full(sum(M.Edges_Faces>0,2));
M.Closed=all(M.DegreeE==2);
M.Vertices_Edges=sparse(M.Edges_Vertices(:),[1:M.nbE,1:M.nbE],1);
M.Adjacency_Faces=double(M.Edges_Faces~=0)'*double(M.Edges_Faces~=0);
M.Adjacency_Vertices=sparse([M.Edges_Vertices(:,1);M.Edges_Vertices(:,2)],[M.Edges_Vertices(:,2);M.Edges_Vertices(:,1)],ones(2*size(M.Edges_Vertices,1),1));
M.DegreeVE=full(sum(M.Adjacency_Vertices));

% compute list of vertice for fast computation of the willmore energy
% use to compute concavity of edges , but could be avoid
tmp=M.Edges_Faces';
willmore.i=zeros(1,M.nbE);
willmore.j=zeros(1,M.nbE);
willmore.k=zeros(1,M.nbE);
willmore.l=zeros(1,M.nbE);

for ide=1:M.nbE
  
    willmore.i(ide)=M.Edges_Vertices(ide,1);
    willmore.j(ide)=M.Edges_Vertices(ide,2);
    
    % for each adjacent face, get the oposite edge
    [adjFaces,~,idEdgeInFace]=find(tmp(:,ide));
    if M.F(mod(idEdgeInFace(1)+2,3)+1,adjFaces(1))==M.Edges_Vertices(ide,1)
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
        id= max(a,b)+(min(a,b)-1)*M.nbV;
    end
% inverse of the function idege
    function [a,b]=inv_idedge(id)
        a=mod(id-1,M.nbV)+1;
        b=(id-a)/M.nbV+1;
        if nargout<2
        a=[a,b];
        end    
    end

end

  
