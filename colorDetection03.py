from collections import deque
import numpy as np
import cv2
import imutils



def convertor(blue, green, red):

    color = np.uint8([[[blue, green, red]]])

    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    hue = hsv_color[0][0][0]

    lower_range = np.array([str(hue-10), 100, 100], dtype=np.uint8)
    upper_range = np.array([str(hue + 10) , 255, 255], dtype=np.uint8)

    print("Lower bound is :", lower_range)
    print("Upper bound is :", upper_range)

    return lower_range, upper_range

def tracking(mask, pts, img):
	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	cnts = imutils.grab_contours(cnts)
	center = None
 
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			
			cv2.circle(img, center, 5, (0, 0, 255), -1)
 
	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

    
def main():

    lower_range, upper_range = convertor(0, 255, 135)

    # initialize the list of tracked points
    pts = deque(maxlen =64)

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
        
        # resize imag to 20% in each axis
        #img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)

        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        tracking(mask, pts, img)

        cv2.imshow('BGR', img)  
        cv2.imshow('mask',mask)

        if cv2.waitKey(33) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()
