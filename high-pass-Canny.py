# HIGH PASS FILTERS can perform these actions:
# edge detection --> using Canny Edge Detection Algorithm


import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	img_path = "D:\\Dataset\\4.1.05.tiff"
	img = cv2.imread(img_path, 0) # takes img as gray

	L1 = cv2.Canny(img, 50, 300, L2gradient=False)

	L2 = cv2.Canny(img, 100, 150, L2gradient=True)

	outputs = [img, L1, L2]

	titles = ['Original', 'L1 Norm', 'L2 Norm']

	for i in range(3):
		plt.subplot(1, 3, i+1) 
		plt.imshow(outputs[i], cmap='gray')
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	plt.show()

if __name__ == '__main__':
	main()

