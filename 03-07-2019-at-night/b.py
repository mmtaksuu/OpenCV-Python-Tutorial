from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

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

video = 'videos/soccer_02.mp4'
cap = cv2.VideoCapture(video)

tracker_type = "kcf"
tracker = OPENCV_OBJECT_TRACKERS["kcf"]()

while True:
	
	frame = cap.read()
	frame = frame[1] 

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=800)

	# Start timer
	timer = cv2.getTickCount()

	# grab the updated bounding box coordinates (if any) for each object that is being tracked
	(success, boxes) = trackers.update(frame)

	# Calculate Frames per second (FPS)
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

	box = cv2.selectROI("Frame", np.copy(frame), fromCenter=False, showCrosshair=True)

	if box:
	# create a new object tracker for the bounding box and add it to our multi-object tracker 
		trackers.add(tracker, np.copy(frame), box)

	
	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(np.copy(frame), (x, y), (x + w, y + h), (0, 255, 0), 2)


	# Display tracker type on frame
	cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
	 
	# Display FPS on frame
	cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


cap.release()
cv2.destroyAllWindows()