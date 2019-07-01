# OpenCV works with HSV (Hue, Saturation, Value) color model to  interpret colors.
# So, if we want to track a certain color using OpenCV, we must define it using the HSV Model.
# We must convert from BGR model to an HSV model for our object's color.

"""
With BGR, a pixel is represented by 3 parameters, blue, green, and red. Each parameter usually has a value from 0 – 255.
With HSV, a pixel is also represented by 3 parameters, but it is instead Hue, Saturation and Value.
Hue is the color or shade of the pixel. (RENK TONU)
Saturation is the intensity of the color. (YOGUNLUK, RENKLILIK).  A saturation of 0 is white.
Value is just how bright or dark the color. 
The lower range is the minimum shade of red.
The upper range is the maximum shade of red.
"""

import cv2
import numpy as np


def convertor(blue, green, red):

	color = np.uint8([[[blue, green, red]]])
	hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

	hue = hsv_color[0][0][0]

	lower_range = np.array([str(hue-10), 100, 100], dtype=np.uint8)
	upper_range = np.array([str(hue + 10) , 255, 255], dtype=np.uint8)

	return lower_range, upper_range
	

def main():

	path = "C:\\Users\\enesa\\Documents\\MATLAB\\dur.jpg"
	img = cv2.imread(path, 1)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_range, upper_range = convertor(180, 0, 0)
	
	print("Lower bound is :", lower_range)
	print("Upper bound is :", upper_range)

	mask = cv2.inRange(hsv, lower_range, upper_range)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(img, hsv, mask=mask)

	cv2.imshow('BGR', img) 
	cv2.imshow('RES', res) 
	cv2.imshow('mask',mask)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
		

if __name__ == '__main__':
	main()