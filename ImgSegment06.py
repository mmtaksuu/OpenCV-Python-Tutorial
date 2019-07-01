import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
	
	path = "C:\\Users\\enesa\\Documents\\MATLAB\\blobs_objects.jpg"

	img = cv2.imread(path, 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	filter1 = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), np.float32) #Sharpening Filter
	output = cv2.filter2D(img, -1, filter1) #convolution filter

	blur = cv2.GaussianBlur(img,(5,5),0)

	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	
	_, thresh = cv2.threshold(gray,170,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
	dilation = cv2.dilate(thresh,se2,iterations = 1)

	# Noise removal using Morphological closing operation
	closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 4) 

	# Noise removal using Morphological opening operation
	opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations = 1)

###########################################################################################################################
	
	dilation = 255 - dilation # Complementing Operation
	
	_, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	print("{} Objects have detected!".format(len(contours))) 
	
	original = cv2.imread(path, 1)
	
	original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

	sayac = 0
	for i in contours:
		# perimeter = cv2.arcLength(i,True)
		# if perimeter > 20:
		sayac = sayac +1
			#cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
		x,y,w,h = cv2.boundingRect(i)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(img, str(sayac), (x+10, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		
		#plt.plot(cx, cy, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12)  # belilenen noktaya x isareti koy.
		#cv2.putText(img, 'x', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		#cv2.putText(closing, str(sayac), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

	print("{} Objects have drown!".format(sayac))
	
###########################################################################################################################
	# output = [original, img]
	# titles = ['Original', 'Contours']
	
	
	# for i in range(2):
	# 	plt.subplot(1, 2, i+1)
	# 	plt.imshow(output[i])
	# 	plt.title(titles[i])
	# 	plt.xticks([])
	# 	plt.yticks([])

	cv2.imshow('Orignal Image', img)
	#cv2.imshow('Erosion Image', erosion) 
	cv2.imshow('Dilation Image', dilation) 
	cv2.imshow('Closing Image', closing) 
	cv2.imshow('Opening Image', opening) 


	plt.show() 
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

if __name__ == "__main__":
	main()
