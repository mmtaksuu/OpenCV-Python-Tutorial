# HIGH PASS FILTERS can perform these actions:
# edge detection --> using Sobel() func.


import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	img_path = "D:\\Dataset\\5.1.11.tiff"
	img = cv2.imread(img_path, 1) # 1 provides taking img as rgb
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	sobel_x = cv2.Sobel(img, -1, dx=1, dy=0, ksize=5, scale=1,
					   delta=0, borderType=cv2.BORDER_DEFAULT)

	sobel_y = cv2.Sobel(img, -1, dx=0, dy=1, ksize=5, scale=1,
					   delta=0, borderType=cv2.BORDER_DEFAULT)

	edges = sobel_x + sobel_y

	outputs = [img, sobel_x, sobel_y, edges]

	titles = ['Original', 'dx=1, dy=0', 'dx=0, dy=1', 'Edges']

	for i in range(4):
		plt.subplot(2, 2, i+1) 
		plt.imshow(outputs[i])
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	plt.show()
if __name__ == '__main__':
	main()

