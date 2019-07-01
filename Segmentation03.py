#Image Segmentation03

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
	# Read the image
	img = cv2.imread('j.png',1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Show Image
	#cv2.imshow('Image', img) # This image already is a binary image

	# Create a simple filter. The kernel slides through the image (as in 2D convolution).
	kernel = np.ones((5,5), np.uint8)

	# Apply Erosion method over the image with kernel
	erosion = cv2.erode(img,kernel,iterations = 1)

	# Apply Dilation method over the image with kernel
	dilation = cv2.dilate(img,kernel,iterations = 1)

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

