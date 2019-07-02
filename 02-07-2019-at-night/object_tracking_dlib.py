'''
Using Correlation Trackers in Dlib, you can track any object in a video stream without needing to train a custom object detector.
Check out the tutorial at: http://www.codesofinterest.com/2018/02/track-any-object-in-video-with-dlib.html
'''
import numpy as np
import cv2
import dlib

# this variable will hold the coordinates of the mouse click events.
mousePoints = []

def mouseEventHandler(event, x, y, flags, param):
    # references to the global mousePoints variable
    global mousePoints

    # if the left mouse button was clicked, record the starting coordinates.
    if event == cv2.EVENT_LBUTTONDOWN:
        mousePoints = [(x, y)]

    # when the left mouse button is released, record the ending coordinates.
    elif event == cv2.EVENT_LBUTTONUP:
        mousePoints.append((x, y))

# create the video capture.
video_capture = cv2.VideoCapture(0)

# create a named window in OpenCV and attach the mouse event handler to it.
cv2.namedWindow("Webcam stream")
cv2.setMouseCallback("Webcam stream", mouseEventHandler)

# initialize the correlation tracker.
tracker = dlib.correlation_tracker()

# this is the variable indicating whether to track the object or not.
tracked = False

while True:
    # start capturing the video stream.
    ret, frame = video_capture.read()

    if ret:
        image = frame

        # if we have two sets of coordinates from the mouse event, draw a rectangle.
        if len(mousePoints) == 2:
            cv2.rectangle(image, mousePoints[0], mousePoints[1], (0, 255, 0), 2)
            dlib_rect = dlib.rectangle(mousePoints[0][0], mousePoints[0][1], mousePoints[1][0], mousePoints[1][1])

        # tracking in progress, update the correlation tracker and get the object position.
        if tracked == True:
            tracker.update(image)
            track_rect = tracker.get_position()
            x  = int(track_rect.left())
            y  = int(track_rect.top())
            x1 = int(track_rect.right())
            y1 = int(track_rect.bottom())
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

        # show the current frame.
        cv2.imshow("Webcam stream", image)

    # capture the keyboard event in the OpenCV window.
    ch = 0xFF & cv2.waitKey(1)

    # press "r" to stop tracking and reset the points.
    if ch == ord("r"):
        mousePoints = []
        tracked = False

    # press "t" to start tracking the currently selected object/area.
    if ch == ord("t"):
        if len(mousePoints) == 2:
            tracker.start_track(image, dlib_rect)
            tracked = True
            mousePoints = []

    # press "q" to quit the program.
    if ch == ord('q'):
        break

# cleanup.
video_capture.release()
cv2.destroyAllWindows()