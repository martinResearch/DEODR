import os
from typing import List

from xml.dom import minidom
from .obj import save_obj
from imageio import imwrite
import numpy as np
from .triangulated_mesh import ColoredTriMesh
from .differentiable_renderer import Camera


def export_meshlab(
    filename: str,
    mesh: ColoredTriMesh,
    cameras: List[Camera],
    images: List[np.ndarray],
    obj_name: str = "mesh.obj",
) -> None:

    root = minidom.Document()
    xml = root.createElement("MeshLabProject")
    root.appendChild(xml)

    mesh_group = root.createElement("MeshGroup")
    xml.appendChild(mesh_group)
    ml_mesh = root.createElement("MLMesh")
    ml_mesh.setAttribute("filename", obj_name)
    ml_mesh.setAttribute("visible", "1")
    ml_mesh.setAttribute("label", obj_name)
    save_obj(
        os.path.join(os.path.dirname(filename), obj_name), mesh.vertices, mesh.faces
    )

    mesh_group.appendChild(ml_mesh)
    ml_matrix = root.createElement("MLMatrix")
    txt = root.createTextNode("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")
    ml_matrix.appendChild(txt)
    mesh_group.appendChild(ml_matrix)

    render_group = root.createElement("RasterGroup")

    for i, (camera, image) in enumerate(zip(cameras, images)):
        image_file = f"raster{i:d}.png"
        ml_raster = root.createElement("MLRaster")
        vcg_camera = root.createElement("VCGCamera")
        mtx = camera.camera_to_world_mtx_4x4()
        translation = np.diag([-1, -1, -1, 1]).dot(mtx[:, 3])
        vcg_camera.setAttribute(
            "TranslationVector",
            " ".join([str(v) for v in translation]),
        )

        vcg_camera.setAttribute(
            "CenterPx", " ".join([str(v) for v in camera.intrinsic[0:2, 2]])
        )
        vcg_camera.setAttribute("PixelSizeMm", "1 1")
        vcg_camera.setAttribute("FocalMm", str(camera.intrinsic[0, 0]))
        vcg_camera.setAttribute("LensDistortion", "0 0")
        vcg_camera.setAttribute("CameraType", "0")
        vcg_camera.setAttribute("BinaryData", "0")
        vcg_camera.setAttribute("ViewportPx", f"{image.shape[0]} {image.shape[1]}")
        rotation = np.diag([1, -1, -1, 1]).dot(mtx)
        rotation[:3, 3] = 0
        vcg_camera.setAttribute(
            "RotationMatrix", " ".join([str(v) for v in rotation.flatten()])
        )

        ml_raster.appendChild(vcg_camera)
        plane = root.createElement("Plane")

        plane.setAttribute("fileName", image_file)
        plane.setAttribute("semantic", "1")
        ml_raster.appendChild(plane)

        imwrite(image_file, image)
        ml_mesh.setAttribute("label", image_file)
        render_group.appendChild(ml_raster)

    xml.appendChild(render_group)
    # <VCGCamera TranslationVector="0.136953 0.683761 -2.68297 1" CenterPx="698 592" PixelSizeMm="0.0369161 0.0369161" FocalMm="61.0115" LensDistortion="0 0" CameraType="0" BinaryData="0" ViewportPx="1397 1184" RotationMatrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 "/>
    # <Plane fileName="hand.png" semantic="1"/>
    # </MLRaster>

    xml_str = root.toprettyxml(indent="\t")

    with open(filename, "w") as f:
        f.write(xml_str)
    # <!DOCTYPE MeshLabDocument>
    # <MeshLabProject>
    # <MeshGroup>
    # <MLMesh filename="hand.obj" visible="1" label="hand.obj">
    # <MLMatrix44>
    # 1 0 0 0
    # 0 1 0 0
    # 0 0 1 0
    # 0 0 0 1
    # </MLMatrix44>
    # <RenderingOption wireWidth="1" wireColor="64 64 64 255" pointSize="3" pointColor="131 149 69 255" solidColor="192 192 192 255" boxColor="234 234 234 255">100001000000000000000100000001010100000010100000000100111011110000001001</RenderingOption>
    # </MLMesh>
    # </MeshGroup>
    # <RasterGroup>
    # <MLRaster label="hand.png">
    # <VCGCamera TranslationVector="0.136953 0.683761 -2.68297 1" CenterPx="698 592" PixelSizeMm="0.0369161 0.0369161" FocalMm="61.0115" LensDistortion="0 0" CameraType="0" BinaryData="0" ViewportPx="1397 1184" RotationMatrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 "/>
    # <Plane fileName="hand.png" semantic="1"/>
    # </MLRaster>

    # <MLRaster label="hand.png">
    # <VCGCamera TranslationVector="0.136953 0.683761 -4.68297 1" CenterPx="698 592" PixelSizeMm="0.0369161 0.0369161" FocalMm="61.0115" LensDistortion="0 0" CameraType="0" BinaryData="0" ViewportPx="1397 1184" RotationMatrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 "/>
    # <Plane fileName="hand.png" semantic="1"/>
    # </MLRaster>
    # </RasterGroup>
    # </MeshLabProject>
