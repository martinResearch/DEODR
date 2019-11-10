import tangent


def computeVertexNormals(faceNormals):
    n = Vertices_Faces * faceNormals
    l = ((n ** 2).sum(dim=1)).sqrt()
    normals = n / l[:, None]
    return normals


computeVertexNormals_back = tangent.grad(computeVertexNormals, verbose=1)
