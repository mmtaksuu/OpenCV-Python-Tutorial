#Image Segmentation05

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	path = "C:\\Users\\enesa\\Documents\\MATLAB\\"

	imgpath1 =  path + "rsm_snv3.png"
	
	img = cv2.imread(imgpath1, 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	ret, thresh = cv2.threshold(gray, 75, 255, 0)

###########################################################################################################################

	# Create a simple filter. The kernel slides through the image (as in 2D convolution).
	kernel = np.ones((3, 3), np.uint8) 

	# Create a Rectangular Structuring Element
	se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

	# Create a Elliptical Structuring Element
	se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		 
	# Apply Erosion method over the image with kernel
	erosion = cv2.erode(thresh,se1,iterations = 1)

	# Apply Dilation method over the image with kernel
	dilation = cv2.dilate(thresh,se1,iterations = 1)

	# Noise removal using Morphological closing operation
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 1) 

	# Noise removal using Morphological opening operation
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)

###########################################################################################################################

	# Detect and Count Objects in the image
	copy_img = opening.copy()
	_, contours, _ = cv2.findContours(copy_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#i = contours[0]

	# cx = int(M['m10']/M['m00'])
	# cy = int(M['m01']/M['m00'])
	# print(cx)
	# print(cy)

	for i in contours:
		x,y,w,h = cv2.boundingRect(i)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	# 	M = cv2.moments(i)
	# 	cx = int(M['m10']/M['m00']) #gives x-coordinate of the element
	# 	cy = int(M['m01']/M['m00']) #gives y-coordinate of the element	
	# 	#plt.plot(cx, cy, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12)  # belilenen noktaya x isareti koy.
	# 	cv2.putText(opening, "{}".format(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


	cv2.imshow('Original Image', img) 
	#cv2.imshow('Erosion Image', erosion) 
	#cv2.imshow('Dilation Image', dilation) 
	#cv2.imshow('Closing Image', closing) 
	#cv2.imshow('Opening Image', opening) 
	#cv2.imshow('Contours', opening)

	plt.subplot(1, 3, 1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])

	plt.subplot(1, 3, 2), plt.imshow(opening), plt.title('Opening Image'), plt.xticks([]), plt.yticks([])

	plt.subplot(1, 3, 3), plt.imshow(copy_img), plt.title('Contours Image'), plt.xticks([]), plt.yticks([])

	


	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

