
import cv2
import numpy as np


def convertor(blue, green, red):

    color = np.uint8([[[blue, green, red]]])

    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    hue = hsv_color[0][0][0]

    lower_range = np.array([str(hue-10), 100, 100], dtype=np.uint8)
    upper_range = np.array([str(hue + 10) , 255, 255], dtype=np.uint8)

    print("Lower bound is :", lower_range)
    print("Upper bound is :", upper_range)

    return lower_range, upper_range
    

def main():

    lower_range, upper_range = convertor(0, 255, 135)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 300) # set video widht
    cam.set(4, 400) # set video height

    if cam.isOpened():
        ret, img = cam.read()
    else:
        ret = False

    while True: 
        ret, img = cam.read()
        
        # resize imag to 20% in each axis
        #img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)

        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        #cv2.imshow('BGR', img) 
        #cv2.imshow('HSV', hsv) 
        cv2.imshow('mask',mask)

        if cv2.waitKey(33) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()
