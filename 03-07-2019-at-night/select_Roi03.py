# https://github.com/uoaerg/wise/blob/master/control/selectROI.py

import argparse
import cv2
import numpy as np
import copy


class selectROI():

    def __init__(self):
    
        self.refPt = []
        self.currPt = 0
        self.nextPt = 1
        self.clickEventsEnabled = False
        self.drawingRectangle = False
        self.rectangleDrawn = False
        self.drawingPolygonLine = False
        self.polygonLineDrawn = False
        self.ix = -1
        self.iy = -1
        
        self.refImage = None
    
    def clickPolygonPoints(self, event, x, y, flags, param):

        image_temp = self.refImage.copy()
        
        if(self.clickEventsEnabled == True):
            if event == cv2.EVENT_LBUTTONDOWN:
                if((self.drawingRectangle == False) & (self.rectangleDrawn == False)):
                  self.drawingPolygonLine = True
                  self.refPt.append((x,y))
                  if(len(self.refPt) > 1):
                     cv2.line(self.refImage, self.refPt[self.currPt], self.refPt[self.nextPt], (0,0,255), 2)
                     cv2.imshow("refImage", self.refImage)
                     self.currPt += 1
                     self.nextPt += 1  
            elif event == cv2.EVENT_LBUTTONUP:
                if((self.drawingRectangle == False) &(self.drawingPolygonLine == True)):
                  self.polygonLineDrawn == True
            elif event == cv2.EVENT_RBUTTONDOWN:
                if((self.drawingPolygonLine == False) & (self.polygonLineDrawn == False) & (self.rectangleDrawn == False)):
                  self.drawingRectangle = True
                  self.ix,self.iy = x,y
                  self.refPt.append((x,y))
            elif event == cv2.EVENT_RBUTTONUP:
                if((self.drawingPolygonLine == False) & (self.polygonLineDrawn == False) & (self.rectangleDrawn == False)):
                  self.drawingRectangle = False
                  self.rectangleDrawn = True
                  cv2.rectangle(self.refImage, (self.ix,self.iy), (x,y), (0,0,255), 2)
                  self.refPt.append((self.ix,self.iy))
                  self.refPt.append((x,self.iy))
                  self.refPt.append((x,y))
                  self.refPt.append((self.ix,y))
                  cv2.imshow("refImage", self.refImage)
            elif event ==  cv2.EVENT_MOUSEMOVE:
                if(self.drawingRectangle == True):
                  cv2.rectangle(image_temp, (self.ix,self.iy), (x,y), (0,0,255), 2)
                  cv2.imshow("refImage", image_temp)
                  image_temp = self.refImage
                elif((self.drawingPolygonLine == True)):
                  cv2.line(image_temp, self.refPt[self.currPt], (x,y), (0,0,255), 2)
                  cv2.imshow("refImage", image_temp)
                  image_temp = self.refImage

    def maskImgWithROI(self, aImage, aROIPointsList):
        pointsArray = np.array(aROIPointsList)
        mask = np.zeros(aImage.shape, dtype=np.uint8)
        white = (255,255,255)
        cv2.fillPoly(mask, np.int32([pointsArray]), white)
        maskedImage = cv2.bitwise_and(aImage, mask)
        return maskedImage  
        
    def outputROIMask(self, aImage, aROIPointsList):
        pointsArray = np.array(aROIPointsList)
        pointsArray = pointsArray.reshape((-1,1,2))
        mask = np.zeros(aImage.shape, dtype=np.uint8)
        white = (255,255,255)
        cv2.fillPoly(mask, np.int32([pointsArray]), white)
        return mask  
        
    def runInterface(self):

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--refImage", required = True, help = "path to the image" )
        args = vars(ap.parse_args())
        self.refImage = cv2.imread(args["refImage"])
        originalRefImage = self.refImage.copy()

        cv2.namedWindow("refImage")
        self.clickEventsEnabled = True
        cv2.setMouseCallback("refImage", self.clickPolygonPoints)

        while True:
            cv2.imshow("refImage", self.refImage)
            key = cv2.waitKey(0)
           
            if key == ord("r"):
               self.refImage = originalRefImage.copy()
               self.refPt = []
               self.currPt, self.nextPt = 0, 1
               self.drawingRectangle = False
               self.rectangleDrawn = False
               self.drawingPolygonLine = False
               self.polygonLineDrawn = False
               self.ix, self.iy = -1,-1
            
            if key == ord("p"):
               self.clickEventsEnabled = False
               self.refImage = originalRefImage.copy()
               self.drawingRectangle = False
               self.rectangleDrawn = False
               self.drawingPolygonLine = False
               self.polygonLineDrawn = False
               break

        self.refImage = self.maskImgWithROI(self.refImage, self.refPt)
        cv2.imshow("refImage", self.refImage)
        
        roiMask = self.outputROIMask(self.refImage, self.refPt)
        cv2.imwrite("ROI.jpg", roiMask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
   
   
selROI = selectROI()
selROI.runInterface()