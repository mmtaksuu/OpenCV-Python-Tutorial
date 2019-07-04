from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

points = []
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

def selectROI(event, x, y, flags, param):
	
	global x_start, y_start, x_end, y_end, cropping, points
 
	# if the left click of mouse was DOWN, start RECORDING
	# (x, y) coordinates and indicate that cropping is being
	if event == cv2.EVENT_LBUTTONDOWN:
		x_start, y_start, x_end, y_end = x, y, x, y
		cropping = True
 
	# Mouse is Moving
	elif event == cv2.EVENT_MOUSEMOVE:
		if cropping == True:
			x_end, y_end = x, y
 
	# if the left click of mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates
		x_end, y_end = x, y
		cropping = False # cropping is finished
 
		
		refPoint = [(x_start, y_start), (x_end, y_end)]
		if len(refPoint) == 2: #when two points were found
			#roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
			#cv2.imshow("Cropped", roi)
			points = np.array([[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end]])


def maskImg_WithROI(frame, ROIPointsList):
	pointsArray = np.array(ROIPointsList)
	mask = np.zeros_like(frame.shape, dtype=np.uint8)
	cv2.fillPoly(mask, np.int32([pointsArray]), 255)
	maskedImage = cv2.bitwise_and(frame, mask)
	return maskedImage 

cv2.namedWindow("Original Frame")
cv2.setMouseCallback("Original Frame", selectROI)

video = 'videos/drone01.mp4'
cap = cv2.VideoCapture(video)

while True:

	ret, frame = cap.read()

	frame = imutils.resize(frame, width=800)

	print(points)

	# original_frame = np.copy(frame)

	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# blur = cv2.GaussianBlur(frame,(5,5),0)
	# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

	# masked_frame = maskImg_WithROI(gray, points)
	# cv2.imshow('ROI Frame', masked_frame)
	
	cv2.imshow('Original Frame', frame)


	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()