
from DEODR import readObj, MeshDepthFitter
from scipy.misc import imread,imsave
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import torch
import copy
import cv2
import time
import glob
import datetime
import os
import json

def main():
    depth_image = np.fliplr(np.fromfile('depth.bin', dtype = np.float32).reshape(240,320).astype(np.float))
    depth_image = depth_image[20:-20,60:-60]
    max_depth = 450
    depth_image[depth_image == 0] = max_depth
    depth_image = depth_image/max_depth
    
    w = depth_image.shape[1]
    h = depth_image.shape[0]
    objFile = 'hand.obj'
    faces,vertices = readObj(objFile)   
    
    euler_init = np.zeros((3))
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
        combinedIMage = np.column_stack((depth_image, syntheticDepth.detach().numpy(), 3*diffImage.detach().numpy()))
        cv2.imshow('animation', cv2.resize(combinedIMage, None, fx = 2, fy = 2)) 
        imsave(os.path.join(iterfolder, f'depth_hand_iter_{iter}.png'), combinedIMage)
        key = cv2.waitKey(1) 
        
    with open(os.path.join(iterfolder, 'depth_image_fitting_result_%s.json'%str(datetime.datetime.now()).replace(':','_')), 'w') as f:
        json.dump({'durations':durations, 'energies':Energies}, f , indent = 4)
    
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
  main()

    

