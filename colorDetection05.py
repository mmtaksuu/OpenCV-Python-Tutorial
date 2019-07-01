from collections import deque
import numpy as np
import cv2
import imutils



def tracking(mask, img):
	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	_, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	for i in cnts:
		x,y,w,h = cv2.boundingRect(i)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	
def main():

	lower_range = np.array([130, 79, 98], dtype=np.uint8)
	upper_range = np.array([179 , 255, 255], dtype=np.uint8)


	# Initialize and start realtime video capture
	cam = cv2.VideoCapture(0)
	cam.set(3, 300) # set video widht
	cam.set(4, 400) # set video height

	if cam.isOpened():
		ret, img = cam.read()
	else:
		ret = False

	while True: 

		ret, img = cam.read()

		median = cv2.medianBlur(img, 3) 
		#blurred = cv2.GaussianBlur(img, (3, 3), 0)
		hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, lower_range, upper_range)
		mask = cv2.erode(mask, None, iterations=1)
		mask = cv2.dilate(mask, None, iterations=1)

		tracking(mask, img)

		cv2.imshow('BGR', img)  
		cv2.imshow('mask',mask)

		if cv2.waitKey(33) == 27:
			break

	cam.release()
	cv2.destroyAllWindows()
		

if __name__ == '__main__':
	main()
