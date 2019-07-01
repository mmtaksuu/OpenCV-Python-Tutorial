import cv2
import numpy as np
import imutils

def nothing(x):
	pass

def main():

	cap = cv2.VideoCapture(0)

	cv2.namedWindow("Trackbars")
	cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
	cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
	cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
	cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
	cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
	cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

	while (cap.isOpened()):

		ret, frame = cap.read()

		if ret is True:

			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

			lower_h = cv2.getTrackbarPos("L - H", "Trackbars")
			lower_s = cv2.getTrackbarPos("L - S", "Trackbars")
			lower_v = cv2.getTrackbarPos("L - V", "Trackbars")
			upper_h = cv2.getTrackbarPos("U - H", "Trackbars")
			upper_s = cv2.getTrackbarPos("U - S", "Trackbars")
			upper_v = cv2.getTrackbarPos("U - V", "Trackbars")

			lower_range = np.array([lower_h, lower_s, lower_v])
			upper_range = np.array([upper_h, upper_s, upper_v])

			mask = cv2.inRange(hsv, lower_range, upper_range)
			result = cv2.bitwise_and(frame, frame, mask=mask)

			cv2.imshow("frame", frame)
			cv2.imshow("mask", mask)
			cv2.imshow("result", result)

		# ESC to break
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break

		else:
			continue

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()