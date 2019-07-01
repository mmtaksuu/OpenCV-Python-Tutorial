import cv2
import numpy as np
import imutils

def nothing(x):
	pass

# Load an image
path = "C:\\Users\\enesa\\Documents\\MATLAB\\final3.PNG"
img = cv2.imread(path, 1)

# Resize The image
# if img.shape[1] > 600:
# 	img = imutils.resize(img, width=600)

if img.shape[0] > 400:
     img = imutils.resize(img, height=500)

# Create a window
cv2.namedWindow('Finding Treshold')

# create trackbars for treshold change
cv2.createTrackbar('Treshold','Finding Treshold', 0, 255, nothing)


while(1):
  
	# Clone original image to not overlap drawings
	clone = img.copy()
	
	# Convert to gray
	gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
	
	# get current positions of four trackbars
	r = cv2.getTrackbarPos('Treshold','Finding Treshold')
	
	# Thresholding the gray image
	ret,thresh = cv2.threshold(gray, r, 255, cv2.THRESH_BINARY)
	
	# To remove the noise in the image with a 5x5 Gaussian filter.
	blur = cv2.GaussianBlur(thresh, (5, 5), 0) 
	
	# Detect edges
	edges = cv2.Canny(blur, 50, 150)
	
	# Find contours
	_, contours, _= cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contour_list = []

	for i in contours:
		area = cv2.contourArea(i)
		if (area > 30):
			contour_list.append(i)
	
	# Draw contours on the original image
	cv2.drawContours(clone, contour_list,  -1, (255,0,0), 2)
  

	#Displaying the results     
	cv2.imshow('Detecting Objects', clone)
	cv2.imshow("Treshholding", thresh)
	
	# ESC to break
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

# close all open windows
cv2.destroyAllWindows()