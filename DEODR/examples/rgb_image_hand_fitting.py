from DEODR import readObj
from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import time
import datetime
import glob
import json
import os


def example_rgb_hand_fitting(
    dl_library="pytorch", plot_curves=True, save_images=True, display=True
):
    if dl_library == "pytorch":
        from DEODR.pytorch import MeshRGBFitterWithPose
    elif dl_library == "tensorflow":
        from DEODR.tensorflow import MeshRGBFitterWithPose
    elif dl_library == "none":
        from DEODR.mesh_fitter import MeshRGBFitterWithPose
    else:
        raise BaseException(f"unkown deep learning library {dl_library}")

    handImage = imread("hand.png").astype(np.double) / 255
    w = handImage.shape[1]
    h = handImage.shape[0]
    objFile = "hand.obj"
    faces, vertices = readObj(objFile)

    defaultColor = np.array([0.4, 0.3, 0.25])
    defaultLight = {
        "directional": -np.array([0.1, 0.5, 0.4]),
        "ambiant": np.array([0.6]),
    }

    euler_init = np.array([0, 0, 0])
    translation_init = np.mean(vertices, axis=0)
    # centering vertices
    vertices = vertices - translation_init[None, :]

    handFitter = MeshRGBFitterWithPose(
        vertices,
        faces,
        defaultColor=defaultColor,
        defaultLight=defaultLight,
        updateLights=True,
        updateColor=True,
        euler_init=euler_init,
        translation_init=translation_init,
        cregu=1000,
    )

    handFitter.reset()
    maxIter = 100

    backgroundColor = np.median(
        np.row_stack(
            (
                handImage[:10, :10, :].reshape(-1, 3),
                handImage[-10:, :10, :].reshape(-1, 3),
                handImage[-10:, -10:, :].reshape(-1, 3),
                handImage[:10, -10:, :].reshape(-1, 3),
            )
        ),
        axis=0,
    )

    backgroundColor = np.array([0.5, 0.6, 0.7])
    handFitter.setImage(handImage, dist=[-1, 0, 0, 0, 0])
    handFitter.setBackgroundColor(backgroundColor)
    Energies = []
    durations = []
    start = time.time()

    iterfolder = "./iterations/rgb"
    if not os.path.exists(iterfolder):
        os.makedirs(iterfolder)

    for iter in range(maxIter):
        Energy, Abuffer, diffImage = handFitter.step()
        Energies.append(Energy)
        durations.append(time.time() - start)
        if display or save_images:
            combinedIMage = np.column_stack(
                (handImage, Abuffer, np.tile(diffImage[:, :, None], (1, 1, 3)))
            )
            if display:
                cv2.imshow(
                    "animation", cv2.resize(combinedIMage[:, :, ::-1], None, fx=2, fy=2)
                )
            if save_images:
                imsave(os.path.join(iterfolder, f"hand_iter_{iter}.png"), combinedIMage)
        key = cv2.waitKey(1)

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
                "energies": Energies,
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


if __name__ == "__main__":
    display = True
    save_images = False
    example_rgb_hand_fitting(
        dl_library="tensorflow",
        plot_curves=True,
        display=display,
        save_images=save_images,
    )

    example_rgb_hand_fitting(
        dl_library="pytorch",
        plot_curves=False,
        display=display,
        save_images=save_images,
    )

    example_rgb_hand_fitting(
        dl_library="none", plot_curves=False, display=display, save_images=save_images
    )
