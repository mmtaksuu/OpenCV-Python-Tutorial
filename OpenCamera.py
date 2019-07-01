#HIGH PASS FILTERS can perform these actions:
# edge detection --> using Hough Line Transform
# This code only for opening Camera


import cv2
#import numpy as np
#import matplotlib.pyplot as plt


def main():

	windowName = 'Hough Line Transform'

	cap = cv2.VideoCapture(0)

	if cap.isOpened():
		ret, frame = cap.read()
	else:
		ret = False	

	while ret:
		
		ret, frame = cap.read()
		

		cv2.imshow(windowName, frame)

		if cv2.waitKey(1)==27:
			break

	cv2.destroyAllWindows()
	cap.release()		

if __name__ == '__main__':
	main()

