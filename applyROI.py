  #REGION OF INTEREST - ROI

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
	
	img = cv2.imread('example_6.jpg')
	#cv2.imshow('Result', img)

	# Access a pixel value by its row and column coordinates.
	pixel = img[100,100]
	#print(pixel)        # output will be : [ 75  45 118] Blue, Green, Red

	# Accessing only blue pixel
	bluePixel = img[100,100,0] # output will be : 75
	#print(bluePixel)

	# We can modify the pixel values
	img[100,100] = [255,255,255]
	#print(img[100,100])         # output will be : [255 255 255]

	# ROI (Region of Interest)
	# We can use the Roi when we use a specific object on an image.
	# ROI improves accuracy and performance. For example:
	# Suppose, I want to detect eyes on a face. For this purpose,
	# firstly, I find a face in whole image. When face detection is done,
	# I select the face region and search for eyes inside it instead of 
	# searching whole image.

	face = cv2.rectangle(img,(320,10),(850,650),(255,0,0),5)
	eyes1 = cv2.circle(face,(490,340), 60, (255,255,0), 5)
	eyes2 = cv2.circle(face,(690,340), 60, (255,255,0), 5)
	
	plt.imshow(img)
	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
