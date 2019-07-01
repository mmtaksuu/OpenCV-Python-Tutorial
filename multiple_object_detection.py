from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np


def main():

	# initialize a dictionary that maps strings to their corresponding OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		#"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create
		#"mosse": cv2.TrackerMOSSE_create
	}

	# initialize OpenCV's special multi-object tracker
	trackers = cv2.MultiTracker_create()

	video_path = 'videos/nascar.mp4'
	cap = cv2.VideoCapture(video_path)

	if cap.isOpened():
		ret, frame = cap.read()
	else:
		ret = False	


	while ret:
		
		ret, frame = cap.read()

		#frame = cv2.resize(frame, (700, 700))

		frame = np.copy(frame)

		print(frame.dtype) # output will be : uint8
		print(frame.shape)

		# frame = imutils.resize(frame, width=600)

		# Get bounding box coordinates (if any) for each object that is being tracked
		(success, boxes) = trackers.update(frame)

		# loop over the bounding boxes and draw then on the frame
		for box in boxes:
			(x, y, w, h) = [int(i) for i in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		

		cv2.imshow("Random Object Tracking", frame)

		key = cv2.waitKey(1) & 0xFF

		# if the 's' key is selected, we are going to "select" a bounding box to track
		if key == ord("s"):
			# select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
			box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
				
			# create a new object tracker for the bounding box and add it to our multi-object tracker
			tracker = OPENCV_OBJECT_TRACKERS["kcf"]() 
			trackers.add(tracker, frame, box)

		# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			break

		else:
			cap.release()


	# close all windows
	cap.release()
	cv2.destroyAllWindows()
				

if __name__ == '__main__':
	main()
