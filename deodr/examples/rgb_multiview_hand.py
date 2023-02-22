"""Example with fitting a colored hand mesh model to multiple images."""
from typing_extensions import Literal

import datetime
import glob
import json
import os
import time

import cv2

import deodr
from deodr import read_obj
from deodr.mesh_fitter import MeshRGBFitterWithPoseMultiFrame

from imageio import imread, imwrite

import matplotlib.pyplot as plt

import numpy as np

dl_library_type = Literal["pytorch", "tensorflow", "none"]


def run(
    dl_library: dl_library_type = "pytorch",
    plot_curves: bool = False,
    save_images: bool = False,
    display: bool = True,
    max_iter: int = 400,
) -> None:

    file_folder = os.path.dirname(__file__)
    hand_images = [
        imread(file).astype(np.double) / 255
        for file in glob.glob(os.path.join(deodr.data_path, "./hand_multiview/*.jpg"))
    ]
    nb_frames = len(hand_images)

    obj_file = os.path.join(deodr.data_path, "hand.obj")
    faces, vertices = read_obj(obj_file)

    default_color = np.array([0.4, 0.3, 0.25]) * 1.5
    default_light_directional = -np.array([0.1, 0.5, 0.4])
    default_light_ambient = 0.6
    # default_light = {'directional':np.array([0.0,0.0,0.0]),'ambient':np.array([0.6])}

    euler_init = np.row_stack(
        [np.array([0, yrot, 0]) for yrot in np.linspace(-0.5, 0.5, 3)]
    )

    vertices = vertices - np.mean(vertices, axis=0)
    t_init = np.array([0, -0.2, 0.2])
    translation_init = np.tile(t_init[None, :], [nb_frames, 1])
    # centering vertices

    hand_fitter = MeshRGBFitterWithPoseMultiFrame(
        vertices,
        faces,
        default_color=default_color,
        default_light_directional=default_light_directional,
        default_light_ambient=default_light_ambient,
        update_lights=True,
        update_color=True,
        euler_init=euler_init,
        translation_init=translation_init,
        cregu=2000,
    )
    #  handFitter = MeshRGBFitter(vertices,faces,default_color,default_light,
    # update_lights =  True, update_color= True,cregu=1000)

    hand_fitter.reset()

    max_iter = 300

    hand_image = hand_images[0]
    background_color = np.median(
        np.row_stack(
            (
                hand_image[:10, :10, :].reshape(-1, 3),
                hand_image[-10:, :10, :].reshape(-1, 3),
                hand_image[-10:, -10:, :].reshape(-1, 3),
                hand_image[:10, -10:, :].reshape(-1, 3),
            )
        ),
        axis=0,
    )
    background_color = np.array([0, 0, 0])
    hand_fitter.set_images(hand_images)
    hand_fitter.set_background_color(background_color)
    energies = []
    durations = []
    start = time.time()

    iter_folder = os.path.join(file_folder, "./iterations/depth")
    if not os.path.exists(iter_folder):
        os.makedirs(iter_folder)

    for niter in range(max_iter):

        energy, image, diff_image = hand_fitter.step(check_gradient=False)
        energies.append(energy)
        durations.append(time.time() - start)
        if display or save_images:
            combined_image = np.column_stack(
                (
                    np.row_stack(hand_images),
                    image,
                    np.tile(np.minimum(diff_image, 1)[:, :, None], (1, 1, 3)),
                )
            )
        if display:
            cv2.imshow(
                "animation",
                cv2.resize(combined_image[:, :, ::-1], None, fx=1, fy=1),
            )
        if save_images:
            imwrite(
                os.path.join(iter_folder, f"hand_iter_{niter}.png"),
                (combined_image * 255).astype(np.uint8),
            )
        cv2.waitKey(1)

    # save convergence curve
    with open(
        os.path.join(
            iter_folder,
            f'rgb_image_fitting_result_{str(datetime.datetime.now()).replace(":", "_")}.json',
        ),
        "w",
    ) as f:
        json.dump(
            {
                "label": f"{dl_library} {datetime.datetime.now()}",
                "durations": durations,
                "energies": energies,
            },
            f,
            indent=4,
        )

    # compare with previous runs
    if plot_curves:
        plt.figure()
        for file in glob.glob(
            os.path.join(iter_folder, "rgb_image_fitting_result_*.json")
        ):
            with open(file, "r") as fp:
                json_data = json.load(fp)
                plt.plot(
                    json_data["durations"],
                    json_data["energies"],
                    label=json_data["label"],
                )
                plt.xlabel("duration in seconds")
                plt.ylabel("energies")
        plt.legend()
        plt.figure()
        for file in glob.glob(
            os.path.join(iter_folder, "rgb_image_fitting_result_*.json")
        ):
            with open(file, "r") as fp:
                json_data = json.load(fp)
                plt.plot(json_data["energies"], label=json_data["label"])
                plt.xlabel("iterations")
                plt.ylabel("energies")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    display = True
    save_images = False
    run(dl_library="none", plot_curves=True, display=display, save_images=save_images)
