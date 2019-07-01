#HIGH PASS FILTERS can perform these actions:
# edge detection --> using Laplacian()


import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	img_path = "D:\\Dataset\\5.1.11.tiff"
	img = cv2.imread(img_path, 1) # takes img as bgr
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	edge = cv2.Laplacian(img, -1, ksize=5, scale=1, delta=0,
						 borderType=cv2.BORDER_DEFAULT)

	outputs = [img, edge]

	titles = ['Original', 'Edge']

	for i in range(2):
		plt.subplot(1, 2, i+1) 
		plt.imshow(outputs[i])
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	plt.show()
if __name__ == '__main__':
	main()

