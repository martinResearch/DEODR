from deodr import read_obj
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import glob
import datetime
import os
import json


def example_depth_image_hand_fitting(
    dl_library="pytorch", plot_curves=True, save_images=True, display=True
):
    file_folder = os.path.dirname(__file__)

    if dl_library == "pytorch":
        from deodr.pytorch import MeshDepthFitter
    elif dl_library == "tensorflow":
        from deodr.tensorflow import MeshDepthFitter
    elif dl_library == "none":
        from deodr.mesh_fitter import MeshDepthFitter
    else:
        raise BaseException(f"unkown deep learning library {dl_library}")

    depth_image = np.fliplr(
        np.fromfile(os.path.join(file_folder, "depth.bin"), dtype=np.float32)
        .reshape(240, 320)
        .astype(np.float)
    )
    depth_image = depth_image[20:-20, 60:-60]
    max_depth = 450
    depth_image[depth_image == 0] = max_depth
    depth_image = depth_image / max_depth

    obj_file = os.path.join(file_folder, "hand.obj")
    faces, vertices = read_obj(obj_file)

    euler_init = np.array([0.1, 0.1, 0.1])
    translation_init = np.zeros(3)

    hand_fitter = MeshDepthFitter(
        vertices, faces, euler_init, translation_init, cregu=1000
    )
    max_iter = 150

    hand_fitter.set_image(depth_image, focal=241, dist=[1, 0, 0, 0, 0])
    hand_fitter.set_max_depth(1)
    hand_fitter.set_depth_scale(110 / max_depth)
    energies = []
    durations = []
    start = time.time()

    iterfolder = os.path.join(file_folder, "./iterations/depth")
    if not os.path.exists(iterfolder):
        os.makedirs(iterfolder)

    for iter in range(max_iter):
        energy, synthetic_depth, diff_image = hand_fitter.step()
        energies.append(energy)
        durations.append(time.time() - start)
        if save_images or display:
            combined_image = np.column_stack(
                (depth_image, synthetic_depth, 3 * diff_image)
            )
            if display:
                cv2.imshow("animation", cv2.resize(combined_image, None, fx=2, fy=2))
            if save_images:
                imsave(
                    os.path.join(iterfolder, f"depth_hand_iter_{iter}.png"),
                    combined_image,
                )
        cv2.waitKey(1)

    with open(
        os.path.join(
            iterfolder,
            "depth_image_fitting_result_%s.json"
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

    if plot_curves:
        plt.figure()
        for file in glob.glob(
            os.path.join(iterfolder, "depth_image_fitting_result_*.json")
        ):
            with open(file, "r") as fp:
                json_data = json.load(fp)
                plt.plot(
                    json_data["durations"],
                    json_data["energies"],
                    label=json_data["label"],
                )
        plt.legend()
        plt.figure()
        for file in glob.glob(
            os.path.join(iterfolder, "depth_image_fitting_result_*.json")
        ):
            with open(file, "r") as fp:
                json_data = json.load(fp)
                plt.plot(json_data["energies"], label=json_data["label"])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    display = True


    example_depth_image_hand_fitting(
        dl_library="none", plot_curves=False, save_images=False, display=display
    )

    example_depth_image_hand_fitting(
        dl_library="pytorch", plot_curves=False, save_images=False, display=display
    )
    
    example_depth_image_hand_fitting(
        dl_library="tensorflow", plot_curves=True, save_images=False, display=display
    )
