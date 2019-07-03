import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read image
    im = cv2.imread("videos/baykar_makina_logo.jpg")
     
    # Select ROI
    rects = []
    r = cv2.selectROI("Frame", im, rects, fromCenter=False, showCrosshair=False)
     
    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)