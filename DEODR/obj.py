#! /usr/bin/env python

# OBJ file format.  See http://www.fileformat.info/format/wavefrontobj/
# At the moment only v and f keywords are supported

import numpy as np


def readObj(filename):

    faces = []
    vertices = []
    fid = open(filename, "r")
    nodeCounter = 0
    while True:

        line = fid.readline()
        if line == "":
            break
        while line.endswith("\\"):
            # Remove backslash and concatenate with next line
            line = line[:-1] + fid.readline()
        if line.startswith("v"):
            coord = line.split()
            coord.pop(0)
            nodeCounter += 1
            vertices.append(np.array([float(c) for c in coord]))

        elif line.startswith("f "):
            fields = line.split()
            fields.pop(0)

            # in some obj faces are defined as -70//-70 -69//-69 -62//-62
            cleanedFields = []
            for f in fields:
                f = int(f.split("/")[0]) - 1
                if f < 0:
                    f = nodeCounter + f
                cleanedFields.append(f)
            faces.append(np.array(cleanedFields))

    faces = np.row_stack(faces)
    vertices = np.row_stack(vertices)
    return faces, vertices
