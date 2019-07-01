import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	# Create a black image
	img1 = np.zeros((512, 512, 3), dtype = np.uint8) # output will be 3x3 zeros matrix 
	print(img1)

	# Create a white image
	img2 = np.ones((512, 512, 1), dtype = np.uint8) # output will be 3x3 ones matrix
	print(img2)

	# Draw a diagonal(capraz) red line with thickness of 5 pixel
	cv2.line(img1,(100,50),(100,400),(0,0,255),5)
	cv2.line(img1,(100,50),(200,300),(0,0,255),5)
	cv2.line(img1,(200,300),(300,50),(0,0,255),5)
	cv2.line(img1,(300,50),(300,400),(0,0,255),5)

	# Draw a rectangle(dikdörtgen) green line with thickness of 3 pixel
	cv2.rectangle(img1,(400,200),(500,400),(0,255,0),3)
	
	# Draw a circle(daire) blue line with thickness of 3 pixel
	cv2.circle(img1,(350,250), 50, (0,0,255), -1) #-1 means inside fill
	cv2.circle(img1,(350,350), 50, (255,0,0), 2)

	# Draw an ellipse blue line with thickness of 2 pixel
	cv2.ellipse(img1,(40,40),(400,200),0,0,180,255,2)

	# Draw a polygon of with four vertices in white color.
	# Firstly, we need coordinates of vertices and it should be of type int32.
	points = np.array([[150,500], [250,500], [250,300], [150,300]], np.int32)
	#points = points.reshape((-1,1,2))
	cv2.polylines(img1, [points], True, (255,255,255),3)

	# Add text to Image 
	text = 'openCV-python'
	position = (10,50)
	font = cv2.FONT_HERSHEY_SIMPLEX
	size = 2
	color = (255,255,255)
	thickness = 2
	cv2.putText(img1, text, position, font, size, color, thickness, cv2.LINE_AA)

	cv2.imshow('Result1', img1)
	
	#plt.imshow(img1)
	#plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
