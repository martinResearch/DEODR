function L=mesh_laplacian(M)

L=spdiags(M.DegreeVE',0,length(M.DegreeVE),length(M.DegreeVE))-M.Adjacency_Vertices;