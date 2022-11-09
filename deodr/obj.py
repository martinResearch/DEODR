#! /usr/bin/env python
"""Function to oad wavefront OBJ files

See http://www.fileformat.info/format/wavefrontobj/.
At the moment only v and f keywords are supported
"""
from typing import List, Tuple
import numpy as np


def read_obj(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Very limited obj file loading.

    At the moment only v and f keywords are supported
    """

    faces = []
    vertices = []
    fid = open(filename, "r")
    node_counter = 0
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
            node_counter += 1
            vertices.append(np.array([float(c) for c in coord]))

        elif line.startswith("f "):
            fields = line.split()
            fields.pop(0)

            # in some obj faces are defined as -70//-70 -69//-69 -62//-62
            cleaned_fields: List[int] = []
            for f in fields:
                v = int(f.split("/")[0]) - 1
                if v < 0:
                    v = node_counter + v
                cleaned_fields.append(v)
            faces.append(np.array(cleaned_fields))

    faces_np = np.row_stack(faces)
    vertices_np = np.row_stack(vertices)
    return faces_np, vertices_np
