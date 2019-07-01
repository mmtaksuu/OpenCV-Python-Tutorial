#Image Segmentation01

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	img_path = "C:\\Users\\enesa\\Documents\\MATLAB\\filtresim2.png"
	img = cv2.imread(img_path, 1) # takes img as bgr
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	cv2.imshow('Image', img)

	
	# hist = cv2.calcHist([img],[0],None,[256],[0,256])
	# plt.hist(img.ravel(),256,[0,256])
	# plt.title('Histogram for Image')

	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	cv2.imshow('Tresholding Image', thresh1)

	plt.subplot(1, 2, 1) 
	plt.imshow(img)
	plt.title('Original Image')

	plt.subplot(1, 2, 2)
	plt.imshow(thresh1)
	plt.title('Tresholding Image = 100')

	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
