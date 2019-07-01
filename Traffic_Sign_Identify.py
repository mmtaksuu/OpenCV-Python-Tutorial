# Countours Operations

import cv2
import numpy as np
import pandas as pd
from math import copysign, log10

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

cnt = contours[1]
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

#Calculate Central Moments
M = cv2.moments(cnt) 
#Calculate Hu Moments
huMoments = cv2.HuMoments(M) 

# empty list
my_list = []

# Log scale hu moments
for i in range(0,7):
  huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
  print('huM[',i+1,'] : ', huMoments[i])
  my_list.append(huMoments[i])

my_list = my_list
df = pd.DataFrame(my_list)
print(df)

#df.to_csv('data2.csv')

# Data Dictionary
# my_dict = {'huM[1]':[huMoments[0],  ], 
# 			'huM[2]':[huMoments[1], ], 
# 			'huM[3]':[huMoments[2], ], 
# 			'huM[4]':[huMoments[3], ], 
# 			'huM[5]':[huMoments[4], ], 
# 			'huM[6]':[huMoments[5], ], 
# 			'huM[7]':[huMoments[6]  ]}

# df = pd.DataFrame(my_dict)
# print(df)

# df.to_csv('data.csv')

cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows() 