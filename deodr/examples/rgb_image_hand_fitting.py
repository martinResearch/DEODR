"""Example with fitting a colored hand mesh model to an image."""


import datetime
import glob
import json
import os
import time

import cv2

import deodr
from deodr import read_obj

from imageio import imread, imsave

import matplotlib.pyplot as plt

import numpy as np


def run(
    dl_library="pytorch", plot_curves=True, save_images=True, display=True, max_iter=100
):
    if dl_library == "pytorch":
        from deodr.pytorch import MeshRGBFitterWithPose
    elif dl_library == "tensorflow":
        from deodr.tensorflow import MeshRGBFitterWithPose
    elif dl_library == "none":
        from deodr.mesh_fitter import MeshRGBFitterWithPose
    else:
        raise BaseException(f"unkown deep learning library {dl_library}")

    hand_image = (
        imread(os.path.join(deodr.data_path, "hand.png")).astype(np.double) / 255
    )
    obj_file = os.path.join(deodr.data_path, "hand.obj")
    faces, vertices = read_obj(obj_file)

    default_color = np.array([0.4, 0.3, 0.25])
    default_light = {
        "directional": -np.array([0.1, 0.5, 0.4]),
        "ambient": np.array([0.6]),
    }

    euler_init = np.array([0, 0, 0])
    translation_init = np.mean(vertices, axis=0)
    # centering vertices
    vertices = vertices - translation_init[None, :]

    hand_fitter = MeshRGBFitterWithPose(
        vertices,
        faces,
        default_color=default_color,
        default_light=default_light,
        update_lights=True,
        update_color=True,
        euler_init=euler_init,
        translation_init=translation_init,
        cregu=1000,
    )

    hand_fitter.reset()

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

    background_color = np.array([0.5, 0.6, 0.7])
    hand_fitter.set_image(hand_image, distortion=[-1, 0, 0, 0, 0])
    hand_fitter.set_background_color(background_color)
    energies = []
    durations = []
    start = time.time()

    iterfolder = "./iterations/rgb"
    if not os.path.exists(iterfolder):
        os.makedirs(iterfolder)

    for niter in range(max_iter):
        energy, image, diff_image = hand_fitter.step()
        energies.append(energy)
        durations.append(time.time() - start)
        if display or save_images:
            combined_image = np.column_stack(
                (hand_image, image, np.tile(diff_image[:, :, None], (1, 1, 3)))
            )
            if display:
                cv2.imshow(
                    "animation",
                    cv2.resize(combined_image[:, :, ::-1], None, fx=2, fy=2),
                )
            if save_images:
                imsave(
                    os.path.join(iterfolder, f"hand_iter_{niter}.png"), combined_image
                )
        cv2.waitKey(1)

    # save convergence curve
    with open(
        os.path.join(
            iterfolder,
            "rgb_image_fitting_result_%s.json"
            % str(datetime.datetime.now()).replace(":", "_"),
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
            os.path.join(iterfolder, "rgb_image_fitting_result_*.json")
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
            os.path.join(iterfolder, "rgb_image_fitting_result_*.json")
        ):
            with open(file, "r") as fp:
                json_data = json.load(fp)
                plt.plot(json_data["energies"], label=json_data["label"])
                plt.xlabel("interations")
                plt.ylabel("energies")
        plt.legend()
        plt.show()
    return energies


def main():

    display = True
    save_images = False

    run(
        dl_library="pytorch",
        plot_curves=False,
        display=display,
        save_images=save_images,
    )

    run(dl_library="none", plot_curves=False, display=display, save_images=save_images)
    run(
        dl_library="tensorflow",
        plot_curves=True,
        display=display,
        save_images=save_images,
    )


if __name__ == "__main__":
    main()
