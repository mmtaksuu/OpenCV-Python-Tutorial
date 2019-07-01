# Countours Operations

import cv2
import numpy as np

path = "C:\\Users\\enesa\\Documents\\MATLAB\\blobs_objects.jpg"

# First Image
img = cv2.imread(path)
blur = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
_,im = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilation = cv2.dilate(im,se1,iterations = 1)

kernel = np.ones((3, 3), np.uint8) 
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 1)

contours,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours) ,'Contours are detected.')

cnt = contours[10]
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# cnt1 = contours[6]
# cnt2 = contours[10]
# d = cv2.matchShapes(cnt1,cnt2,cv2.CONTOURS_MATCH_I2,0)  
# print('Shape Distances Between img1 and img2 : ', d)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 