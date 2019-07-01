# OpenCV provides mainly four types of blurring techniques. Image Bluring (Image Smoothing)
# 1. Average Filtering   --> cv2.boxFilter()
# 2. Gaussian Blurring   --> cv2.GaussianBlur()
# 3. Median Blurring     --> cv2.medianBlur() 
# 4. Bilateral Filtering --> cv2.bilateralFilter() 
# filter2D() uses conv2 the 2 dimensional convolution function to implement filtering operation.
# It is used for blurring, sharpening, embossing, edge detection, and so on. 
# cv2.filter2D(image, -1, kernel) apply the convolution kernel using filter2D function.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	img_path = "D:\\Dataset\\4.1.07.tiff"
	img = cv2.imread(img_path, 1) # 1 provides taking img as rgb
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


	filter33 = np.ones((3,3), np.float32)/9 # 3-by-3 neighborhood avarage filter

	filter55 = np.zeros((5,5), np.float32)/25 # 5-by-5 neighborhood avarage filter

	filter1 = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), np.float32) 

	filter2 = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), np.float32)/9 # 3-by-3 neighborhood avarage filter

	filter4 = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), np.float32) # 3-by-3 neighborhood sharpening filter

	gauss33 = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1]), np.float32)/16  # 3-by-3 neighborhood gaussian filter

	gauss55 = np.array(([1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]), np.float32)/273 # 5-by-5 neighborhood gaussian filter

	filter6 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), np.float32) # # 3-by-3 neighborhood edge detection filter

	output = cv2.filter2D(img, -1, gauss55) 

	plt.subplot(1, 2, 1) 
	plt.imshow(img)
	plt.title('Original Image')

	plt.subplot(1, 2, 2)
	plt.imshow(output)
	plt.title('Filtered Image')
	

	plt.show()



if __name__ == '__main__':
	main()
