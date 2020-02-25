function L=mesh_laplacian(M)

L=spdiags(M.degree_v_e',0,length(M.degree_v_e),length(M.degree_v_e))-M.adjacency_vertices;