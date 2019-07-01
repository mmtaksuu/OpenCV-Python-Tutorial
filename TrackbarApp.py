import cv2
import numpy as np
import matplotlib.pyplot as plt


def emptyFunc():
	pass

def main():
	
	img = np.zeros((512,512,3), np.uint8)

	windowName = 'OpenCV BGR Color Palette'
	cv2.namedWindow(windowName)

	cv2.createTrackbar('B', windowName, 0, 255, emptyFunc)
	cv2.createTrackbar('G', windowName, 0, 255, emptyFunc)
	cv2.createTrackbar('R', windowName, 0, 255, emptyFunc)


	while True:
		
		cv2.imshow(windowName, img)

		blue = cv2.getTrackbarPos('B', windowName)
		green = cv2.getTrackbarPos('G', windowName)
		red = cv2.getTrackbarPos('R', windowName)

		img[:] = [blue, green, red]
		print(blue, green, red)

		if cv2.waitKey(1) == 27:
			break
	
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
