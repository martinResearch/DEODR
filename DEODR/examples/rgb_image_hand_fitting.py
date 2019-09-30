from DEODR import readObj, MeshRGBFitter, MeshRGBFitterWithPose
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

def main():
    use_pytotch_optimizer = True
    handImage = imread('hand.png').astype(np.double)/255
    w = handImage.shape[1]
    h = handImage.shape[0]
    objFile = 'hand.obj'
    faces,vertices = readObj(objFile)   
    
    defaultColor = np.array([0.4,0.3,0.25])
    defaultLight = {'directional':np.array([0.1,0.5,0.4]),'ambiant':np.array([0.6])} 

    euler_init = np.zeros((3))
    translation_init = np.zeros(3)
    
    handFitter = MeshRGBFitterWithPose(vertices,faces,defaultColor = defaultColor, defaultLight = defaultLight,  
                                updateLights =  True, updateColor= True, euler_init = euler_init, translation_init = translation_init, cregu=1000)   
    #        handFitter = MeshRGBFitter(vertices,faces,defaultColor,defaultLight,  updateLights =  True, updateColor= True,cregu=1000)
    
    handFitter.reset()
    maxIter = 100
    
    backgroundColor = np.median(np.row_stack((handImage[:10,:10,:].reshape(-1,3),handImage[-10:,:10,:].reshape(-1,3),handImage[-10:,-10:,:].reshape(-1,3),handImage[:10,-10:,:].reshape(-1,3))),axis=0)
    backgroundColor = np.array([0.5,0.6,.7])
    handFitter.setImage(handImage)
    handFitter.setBackgroundColor(backgroundColor)
    Energies = []
    durations = []
    start = time.time()
    
    iterfolder='./iterations/rgb'
    if not os.path.exists(iterfolder):
        os.makedirs(iterfolder)    
     
    for iter in range(maxIter):        
        Energy, Abuffer, diffImage = handFitter.step()
        Energies.append(Energy)
        durations.append( time.time() - start  )
        combinedIMage = np.column_stack((handImage,Abuffer.detach().numpy(),np.tile(diffImage.detach().numpy()[:,:,None],(1,1,3))))
        cv2.imshow('animation', cv2.resize(combinedIMage[:,:,::-1], None, fx=2, fy=2)) 
        imsave(os.path.join(iterfolder,f'hand_iter_{iter}.png'), combinedIMage)
        key = cv2.waitKey(1)  
    plt.plot(Energies)
    plt.figure()
    plt.plot(durations,Energies)
       
    # save convergence curve 
    with open(os.path.join(iterfolder,'rgb_image_fitting_result_%s.json'%str(datetime.datetime.now()).replace(':','_')),'w') as f:
        json.dump({'durations':durations,'energies':Energies}, f, indent=4)
    
    # compare with previous runs
    plt.figure()
    for file in glob.glob(os.path.join(iterfolder, "rgb_image_fitting_result_*.json")):
        with open(file,'r') as fp:
            json_data = json.load(fp)   
            plt.plot(json_data['durations'], json_data['energies'], label = file)
    plt.legend()        
    plt.figure()
    for file in glob.glob(os.path.join(iterfolder, "rgb_image_fitting_result_*.json")):
        with open(file,'r') as fp:
            json_data = json.load(fp)               
            plt.plot(json_data['energies'],label = file)
    plt.legend()   
    plt.show()  

if __name__ == "__main__":
    main()

