# LOW PASS FILTERS can perform these actions:

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

	img_path = "D:\\Dataset\\coins.png"
	img = cv2.imread(img_path, 1) # takes img as bgr
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	linear_filter = cv2.boxFilter(img, -1, (3,3)) #linear filter

	blur = cv2.blur(img, (3,3))

	gaussian = cv2.GaussianBlur(img, (7,7), 0)

	titles = ['Original Image', 'Box Filter', 'Blur', 'Gaussian Blur']
			   
	outputs = [img, linear_filter, blur, gaussian]

	for i in range(4):
		plt.subplot(2, 2, i+1) 
		plt.imshow(outputs[i])
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	plt.show()

if __name__ == '__main__':
	main()