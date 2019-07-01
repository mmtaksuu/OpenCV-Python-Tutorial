# Countours Operations

import cv2
import csv
import numpy as np
import pandas as pd

path = "C:\\Users\\enesa\\Documents\\MATLAB\\yavas.jpg"

# First Image
img = cv2.imread(path)
blur = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
_,im = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilation = cv2.dilate(im,se1,iterations = 1)

kernel = np.ones((3, 3), np.uint8) 
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 1)

contours,_ = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours) ,'Contours are detected.')

df = pd.read_csv('data2.csv')
lst = [list(x) for x in df.values]
print(df.values)


#print(your_list)

for i in contours:
	#x,y,w,h = cv2.boundingRect(i)
	#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	#Calculate Central Moments
	M = cv2.moments(i) 

	#Calculate Hu Moments
	huMoments = cv2.HuMoments(M) 

	#if huMoments == dicts:
		#print("DUR")



cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows() 