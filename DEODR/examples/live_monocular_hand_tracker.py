
import DEODR
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import torch
import copy
import cv2


def main():

    cap = cv2.VideoCapture(0)
    w=200
    h=200
    objFile='hand.obj'
    faces,vertices=DEODR.readObj(objFile)   
    
    defaultColor=np.array([200,150,110])/255
    defaultLight={'directional':np.array([0.1,-0.5,0.5]),'ambiant':np.array([0.3])}
    handFitter=DEODR.MeshRGBFitter(vertices,faces,defaultColor,defaultLight,cregu=4000)

    started=False

    while True:

        handFitter.reset()

        maxIter=10
        H = None
        while True:

            ret,frame = cap.read()
            frame=cv2.flip(frame, 1)
            x_min = int((frame.shape[1]*1.5-w)/2)
            y_min = int((frame.shape[0]-h)/2)
            x_max = int((frame.shape[1]*1.5+w)/2)
            y_max = int((frame.shape[0]+h)/2)
            frameCopy = frame.copy()
            cv2.putText(frameCopy,"put your hand in here",(x_min,y_min-25),cv2.FONT_HERSHEY_SIMPLEX, .5, (0,250,0), 1, cv2.LINE_AA)
            cv2.putText(frameCopy,"use uniform color background",(x_min,y_min-10),cv2.FONT_HERSHEY_SIMPLEX, .5, (0,250,0), 1, cv2.LINE_AA)
               
            cv2.rectangle(frameCopy, (x_min,y_min),(x_max,y_max), (0,255,0),2)
            cv2.rectangle(frameCopy, (x_min+10,y_min+10),(x_max-10,y_max-10), (0,255,0),2)
            cv2.putText(frameCopy,"r: restart,  other key:starts and stop",(10,20),cv2.FONT_HERSHEY_SIMPLEX, .5, (200,0,0), 1, cv2.LINE_AA)
            cv2.imshow('Frame',frameCopy)
            handImage = frame[y_min:y_max,x_min:x_max,::-1].astype(np.double)/255
            backgroundColor = np.median(np.row_stack((handImage[:10,:10,:].reshape(-1,3),handImage[-10:,:10,:].reshape(-1,3),handImage[-10:,-10:,:].reshape(-1,3),handImage[:10,-10:,:].reshape(-1,3))),axis=0)
            if started:
                handFitter.setImage(handImage)
                handFitter.setBackgroundColor(backgroundColor)
                for iter in range(maxIter):
                    Energy,Abuffer,diffImage = handFitter.step()
                displayImage=(cv2.resize(np.column_stack((handImage,Abuffer.detach().numpy(),np.tile(diffImage.detach().numpy()[:,:,None],(1,1,3))))[:,:,::-1],None,fx=2,fy=2)*255).astype(np.uint8)
                cv2.putText(displayImage,"input",(20,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (250,0,0), 2, cv2.LINE_AA)
                cv2.putText(displayImage,"synthetic",(20+w*2,30),cv2.FONT_HERSHEY_SIMPLEX, 1,  (250,0,0), 2, cv2.LINE_AA)
                cv2.putText(displayImage,"residual",(20+w*4,30),cv2.FONT_HERSHEY_SIMPLEX, 1,  (250,0,0), 2, cv2.LINE_AA)
                cv2.putText(displayImage,"eneregy = %f"%Energy,(20+w*4,50),cv2.FONT_HERSHEY_SIMPLEX, 1,  (250,0,0), 2, cv2.LINE_AA)
                cv2.imshow('animation',displayImage ) 
            key = cv2.waitKey(1) 
            if key == ord('r'):
                break
            elif key>0:
                if not(started):
                    started=True
                else:

                    return

if __name__ == "__main__":
    main()

