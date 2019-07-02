# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

from imutils.video import VideoStream
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

video = 'videos/traffic.mp4'
cap = cv2.VideoCapture(video)

tracker_type = "kcf"

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
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

	
	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		if success:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		else:
			# Tracking failure
			cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


	# Display tracker type on frame
	cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
	 
	# Display FPS on frame
	cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
		success = True
			

		# create a new object tracker for the bounding box and add it to our multi-object tracker 
		tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
		trackers.add(tracker, frame, box)

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break


cap.release()
cv2.destroyAllWindows()