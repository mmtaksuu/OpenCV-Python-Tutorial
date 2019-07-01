# Countours Operations

import cv2
import numpy as np
import pandas as pd
from math import copysign, log10


# Load an image
#path = "C:\\Users\\enesa\\Documents\\MATLAB\\Tool001.gif"
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

# Data Dictionary
my_dict = {'huM[1]':huMoments[0], 'huM[2]':huMoments[1], 'huM[3]':huMoments[2], 'huM[4]':huMoments[3], 
'huM[5]':huMoments[4], 'huM[6]':huMoments[5], 'huM[7]':huMoments[6]}


df = pd.DataFrame(my_dict)
print(df)

df.to_csv('data.csv')

