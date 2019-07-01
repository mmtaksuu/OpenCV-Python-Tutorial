#Image Segmentation04

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
	# Read the image
	img = cv2.imread('j.png',1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Show Image
	#cv2.imshow('Image', img) # This image already is a binary image

	# Create a Rectangular Structuring Element
	se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

	# Create a Elliptical Structuring Element
	se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
 		 
	# Apply Erosion method over the image with Structuring Element
	erosion = cv2.erode(img,se2,iterations = 1)

	# Apply Dilation method over the image with Structuring Element
	dilation = cv2.dilate(img,se2,iterations = 1)

	plt.subplot(1, 3, 1) 
	plt.imshow(img)
	plt.title('Original Image')

	plt.subplot(1, 3, 2)
	plt.imshow(erosion)
	plt.title('Erosion Image')

	plt.subplot(1, 3, 3)
	plt.imshow(dilation)
	plt.title('Dilation Image')

	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

