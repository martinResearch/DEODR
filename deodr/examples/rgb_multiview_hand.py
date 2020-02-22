from deodr import readObj
from imageio import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import datetime
import glob
import json
import os
from deodr.mesh_fitter import MeshRGBFitterWithPoseMultiFrame
import deodr


def run(dl_library="pytorch", plot_curves=False, save_images=False, display=True):

    handImages = [
        imread(file).astype(np.double) / 255
        for file in glob.glob(os.path.join(deodr.data_path, "./hand_multiview/*.jpg"))
    ]
    nbFrames = len(handImages)

    objFile = os.path.join(deodr.data_path, "hand.obj")
    faces, vertices = readObj(objFile)

    defaultColor = np.array([0.4, 0.3, 0.25]) * 1.5
    defaultLight = {
        "directional": -np.array([0.1, 0.5, 0.4]),
        "ambiant": np.array([0.6]),
    }
    # defaultLight = {'directional':np.array([0.0,0.0,0.0]),'ambiant':np.array([0.6])}

    euler_init = np.row_stack(
        [np.array([0, yrot, 0]) for yrot in np.linspace(-0.5, 0.5, 3)]
    )

    vertices = vertices - np.mean(vertices, axis=0)
    t_init = np.array([0, -0.2, 0.2])
    translation_init = np.tile(t_init[None, :], [nbFrames, 1])
    # centering vertices

    handFitter = MeshRGBFitterWithPoseMultiFrame(
        vertices,
        faces,
        defaultColor=defaultColor,
        defaultLight=defaultLight,
        updateLights=True,
        updateColor=True,
        euler_init=euler_init,
        translation_init=translation_init,
        cregu=2000,
    )
    #  handFitter = MeshRGBFitter(vertices,faces,defaultColor,defaultLight,
    # updateLights =  True, updateColor= True,cregu=1000)

    handFitter.reset()
    maxIter = 150
    handImage = handImages[0]
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
    backgroundColor = np.array([0, 0, 0])
    handFitter.setImages(handImages)
    handFitter.setBackgroundColor(backgroundColor)
    Energies = []
    durations = []
    start = time.time()

    iterfolder = "./iterations/rgb_multiview"
    if not os.path.exists(iterfolder):
        os.makedirs(iterfolder)

    for iter in range(maxIter):
        Energy, Abuffer, diffImage = handFitter.step()
        Energies.append(Energy)
        durations.append(time.time() - start)
        if display or save_images:
            combinedIMage = np.column_stack(
                (
                    np.row_stack(handImages),
                    np.row_stack(Abuffer),
                    np.tile(
                        np.row_stack(np.minimum(diffImage, 1))[:, :, None], (1, 1, 3)
                    ),
                )
            )
            if display:
                cv2.imshow(
                    "animation", cv2.resize(combinedIMage[:, :, ::-1], None, fx=1, fy=1)
                )
            if save_images:
                imsave(
                    os.path.join(iterfolder, f"hand_iter_{iter}.png"),
                    (combinedIMage * 255).astype(np.uint8),
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
    run(dl_library="none", plot_curves=True, display=display, save_images=save_images)
