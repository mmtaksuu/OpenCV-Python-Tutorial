# Countours Operations

import cv2
import numpy as np
import pandas as pd
from math import copysign, log10


# Load an image
#path = "C:\\Users\\enesa\\Documents\\MATLAB\\resim4.png"
img = cv2.imread('j.png')

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold image
_,im = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

#Calculate Central Moments
M = cv2.moments(im)         #calculates the value of the central moment
cx = int(M['m10']/M['m00']) #gives x-coordinate of the central moment
cy = int(M['m01']/M['m00']) #gives y-coordinate of the central moment

print('Central Moments values: \n', M)
print('\n') 
print('X-coordinate of the Central Moment value: ', cx)
print('Y-coordinate of the Central Moment value: ', cy)
print('\n') 

#Calculate Hu Moments
huMoments = cv2.HuMoments(M)       

# empty list
my_list = []

# Log scale hu moments
for i in range(0,7):
  huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
  print('huM[',i+1,'] : ', huMoments[i])
  my_list.append(huMoments[i])

df = pd.DataFrame(my_list)
print(df)

df.to_csv('data2.csv')