from DEODR import readObj

from scipy.misc import imread,imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import glob
import datetime
import os
import json


def example_depth_image_hand_fitting(dl_library = 'pytorch', plot_curves = True,save_images = True):
    
    if dl_library=='pytorch':
        from DEODR.pytorch import MeshDepthFitter
    elif dl_library == 'tensorflow':
        from DEODR.tensorflow import MeshDepthFitter
    else: 
        raise BaseException(f"unkown deep learning library {dl_library}")
    
    depth_image = np.fliplr(np.fromfile('depth.bin', dtype = np.float32).reshape(240,320).astype(np.float))
    depth_image = depth_image[20:-20,60:-60]
    max_depth = 450
    depth_image[depth_image == 0] = max_depth
    depth_image = depth_image/max_depth
    
    w = depth_image.shape[1]
    h = depth_image.shape[0]
    objFile = 'hand.obj'
    faces,vertices = readObj(objFile)   
    
    euler_init = np.array([0.1,0.1,0.1])
    translation_init = np.zeros(3)
   
    handFitter = MeshDepthFitter(vertices, faces, euler_init, translation_init, cregu = 1000)  
    maxIter = 150
    
    handFitter.setImage(depth_image,focal = 241)
    handFitter.setMaxDepth(1)
    handFitter.setDepthScale(110/max_depth)
    Energies = []
    durations = []
    start = time.time()

    iterfolder = './iterations/depth'
    if not os.path.exists(iterfolder):
        os.makedirs(iterfolder)

    for iter in range(maxIter):        
        Energy,syntheticDepth,diffImage = handFitter.step()
        Energies.append(Energy)
        durations.append(time.time() - start)
        combinedIMage = np.column_stack((depth_image, syntheticDepth, 3*diffImage))
        cv2.imshow('animation', cv2.resize(combinedIMage, None, fx = 2, fy = 2)) 
        if save_images:
            imsave(os.path.join(iterfolder, f'depth_hand_iter_{iter}.png'), combinedIMage)
        key = cv2.waitKey(1) 
        
    with open(os.path.join(iterfolder, 'depth_image_fitting_result_%s.json'%str(datetime.datetime.now()).replace(':','_')), 'w') as f:
        json.dump({'durations':durations, 'energies':Energies}, f , indent = 4)
    if plot_curves:
        plt.figure()
        for file in glob.glob(os.path.join(iterfolder, "depth_image_fitting_result_*.json")):
            with open(file,'r') as fp:
                json_data = json.load(fp)   
                plt.plot(json_data['durations'], json_data['energies'], label = file)
        plt.legend()        
        plt.figure()
        for file in glob.glob(os.path.join(iterfolder, "depth_image_fitting_result_*.json")):
            with open(file,'r') as fp:
                json_data = json.load(fp)               
                plt.plot(json_data['energies'], label = file)
        plt.legend()
        plt.show()    
 
if __name__ == "__main__":  
    example_depth_image_hand_fitting(dl_library='tensorflow', plot_curves=True, save_images = False)
    example_depth_image_hand_fitting(dl_library='pytorch', plot_curves=True, save_images = False)
  

    

