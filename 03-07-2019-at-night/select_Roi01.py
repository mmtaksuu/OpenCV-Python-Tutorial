import cv2
import numpy as np
 
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
 
image = cv2.imread('videos/baykar_makina_logo.jpg')
oriImage = image.copy()
 
 
def mouse_crop(event, x, y, flags, param):
    
    global x_start, y_start, x_end, y_end, cropping
 
    # if the left click of mouse was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left click of mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while True:
 
    i = image.copy()
 
    if not cropping:
        cv2.imshow("image", image)
 
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        #i[y_start:y_end, x_start:x_end]
        cv2.imshow("image", i)
 
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
 
# close all open windows
cv2.destroyAllWindows()