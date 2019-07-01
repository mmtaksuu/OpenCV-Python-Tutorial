# We will see the Median Filter is pretty useful to remove salt and pepper noise  from an image.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def main():

	img_path = "D:\\Dataset\\coins.png"
	img = cv2.imread(img_path, 1) # takes img as bgr
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	output = np.zeros(img.shape, dtype = np.uint8)  # Creates a zeros matrix the same size  img.shape for applying noise.
	p = 0.2

	for i in range(img.shape[0]): # img.shape[0] shows the row of the image
		for j in range(img.shape[1]): # img.shape[1] shows the column of the image
			r = random.random()
			if r < p/2:
				output[i][j] = [0, 0, 0] #pepper noise sprinkles
			elif r < p:
				output[i][j] = [255, 255, 255] #salt noise sprinkles
			else:
				output[i][j] = img[i][j]


	cv2.imshow('Noisy Image', output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	kernel33 = np.ones((3,3), np.float32)/9 # 3-by-3 neighborhood avarage filter

	average = cv2.filter2D(output, -1, kernel33) # Apply the average filter to remove noises.
	gaussian = cv2.GaussianBlur(output, (3,3), 0)
	median = cv2.medianBlur(output, 3) 
	bilateral = cv2.bilateralFilter(output, 9, 75, 75)

	outputs = [img, output, average, gaussian, median, bilateral]

	titles = ['Original', 'Noisy', 'AverageFilter', 'GaussianBlur', 'MedianBlur', 'BilateralFilter']

	for i in range(6):
		plt.subplot(2, 3, i+1) 
		plt.imshow(outputs[i])
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	plt.show()

if __name__ == '__main__':
	main()
