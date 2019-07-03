from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	#"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	#"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

video = 'C:\\Users\\enesa\\Desktop\\openCV\\Object Tracking\\videos\\traffic.mp4'
cap = cv2.VideoCapture(video)

tracker_type = "kcf"

if cap.isOpened():
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=800)
	oriFrame = np.copy(frame)
else:
	ret = False	

def mouse_crop(event, x, y, flags, param):
	
	global x_start, y_start, x_end, y_end, cropping
 
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
 
		#refPoint = [(x_start, y_start), (x_end, y_end)]
 
		#if len(refPoint) == 2: #when two points were found
			#roi = oriFrame[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
			#cv2.imshow("Cropped", roi)
 
cv2.namedWindow("Object Tracking")
cv2.setMouseCallback("Object Tracking", mouse_crop)


while True:
	
	frame = cap.read()
	frame = frame[1] 

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=800)
	i = np.copy(frame)
	# Start timer
	timer = cv2.getTickCount()

	# grab the updated bounding box coordinates (if any) for each object that is being tracked
	#(success, boxes) = trackers.update(frame)

	# Calculate Frames per second (FPS)
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

	
	# loop over the bounding boxes and draw then on the frame
	# for box in boxes:
	# 	(x, y, w, h) = [int(v) for v in box]
	# 	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


	if not cropping:
		cv2.imshow("Object Tracking", frame)
 
	elif cropping:
		cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
		cv2.imshow("Object Tracking", i)

		roi = i[y_start:y_end, x_start:x_end]
		cv2.imshow("Cropped", roi)



	# Display tracker type on frame
	#cv2.putText(frame, tracker_type + " Tracker", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
	 
	# Display FPS on frame
	#cv2.putText(frame, "FPS : " + str(int(fps)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

	# show the output frame
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	# if key == ord("s"):
	# 	# select the bounding box of the object we want to track (make
	# 	# sure you press ENTER or SPACE after selecting the ROI)
	# 	box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

	# 	# create a new object tracker for the bounding box and add it to our multi-object tracker 
	# 	tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
	# 	trackers.add(tracker, frame, box)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


cap.release()
cv2.destroyAllWindows()