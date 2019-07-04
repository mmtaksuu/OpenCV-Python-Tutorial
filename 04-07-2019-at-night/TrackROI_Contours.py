from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2


class TrackROI():

	def __init__(self):
		self.refPt = []
		self.clickEventsEnabled = False
		self.drawingRectangle = False
		self.rectangleDrawn = False
		self.x_start = -1
		self.y_start = -1
		self.ref_frame = None

		
	def clickAndCrop(self, event, x, y, flags, param):

		temp_frame = np.copy(self.ref_frame)

		if(self.clickEventsEnabled == True):
			if event == cv2.EVENT_LBUTTONDOWN:
				if((self.rectangleDrawn == False)):
					self.drawingRectangle = True
					self.x_start,self.y_start = x,y
					self.refPt.append((x,y))
					 
			elif event ==  cv2.EVENT_MOUSEMOVE: 	
				if(self.drawingRectangle == True):
					cv2.rectangle(temp_frame, (self.x_start,self.y_start), (x,y), (0,0,255), 2)
					cv2.imshow("OBJECT TRACKING", temp_frame)
					temp_frame = self.ref_frame

			elif event == cv2.EVENT_LBUTTONUP:
				if((self.rectangleDrawn == False)):
					self.drawingRectangle = False
					self.rectangleDrawn = True               
					cv2.rectangle(self.ref_frame, (self.x_start, self.y_start), (x,y), (0,0,255), 2)
					self.refPt.append((self.x_start, self.y_start))
					self.refPt.append((x, self.y_start))
					self.refPt.append((x, y))
					self.refPt.append((self.x_start, y))

					#roiPoints = [(self.x_start, self.y_start), (x, y)]

					# if len(roiPoints) == 2:
					# 	roi = self.ref_frame[roiPoints[0][1]:roiPoints[1][1], roiPoints[0][0]:roiPoints[1][0]]
					# 	cv2.imshow("Cropped Object", roi)
					
					roi = (self.x_start, self.y_start, x, y)

		return np.array([[self.x_start, self.y_start], [x, self.y_start], [x, y], [self.x_start, y]])


	def maskImg_WithROI(self, frame, ROIPointsList):
		pointsArray = np.array(ROIPointsList)
		mask = np.zeros_like(frame.shape, dtype=np.uint8)
		
		cv2.fillPoly(mask, np.int32([pointsArray]), 255)
		maskedImage = cv2.bitwise_and(frame, mask)
		cv2.imshow("Masked Frame", maskedImage)
		return maskedImage  
	 
	def outputROIMask(self, frame, ROIPointsList):
		pointsArray = np.array(ROIPointsList)
		pointsArray = pointsArray.reshape((-1,1,2))
		mask = np.zeros(frame.shape, dtype=np.uint8)
		white = (255,255,255)
		cv2.fillPoly(mask, np.int32([pointsArray]), white)
		return mask  
	 
	def main(self):

		self.clickEventsEnabled = True

		#logo = cv2.imread('Baykar-Logo.png')
		#logo = cv2.resize(logo, (100,100), interpolation=cv2.INTER_AREA)

		video = 'videos/drone01.mp4'
		cap = cv2.VideoCapture(video)

		if cap.isOpened():
			ret, self.ref_frame = cap.read()
			originalRef_Frame = np.copy(self.ref_frame)
		else:
			ret = False

		
		cv2.namedWindow("OBJECT TRACKING")
		cv2.setMouseCallback("OBJECT TRACKING", self.clickAndCrop)

		while ret:


			self.ref_frame = cap.read()
			self.ref_frame = self.ref_frame[1] 

			self.ref_frame = imutils.resize(self.ref_frame, width=800)

			#self.ref_frame[0:100, 700:800] = logo


			cv2.imshow("OBJECT TRACKING", self.ref_frame)
			
			key = cv2.waitKey(0) & 0xFF
			
			if key == ord("r"):
				self.ref_frame = np.copy(originalRef_Frame)
				self.refPt = []
				self.drawingRectangle = False
				self.rectangleDrawn = False
				self.x_start, self.y_start = -1,-1
			
			elif key == ord("p"):
				#self.clickEventsEnabled = False
				self.ref_frame = np.copy(originalRef_Frame)
				self.drawingRectangle = False
				self.rectangleDrawn = False
				#break

			elif key == ord("q"):
				break

			self.gray = cv2.cvtColor(self.ref_frame, cv2.COLOR_BGR2GRAY)

		#self.ref_frame = self.maskImg_WithROI(self, self.gray, self.refPt)
		#cv2.imshow("Masked Frame", self.maskImg_WithROI(self, self.gray, self.refPt))
		

		#roiMask = self.outputROIMask(self.ref_frame, self.refPt)
		#cv2.imwrite("ROI.jpg", roiMask)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
	trackROI = TrackROI()
	trackROI.main()