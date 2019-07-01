import cv2
import numpy as np


def main():
	
	image_path = "C:\\Users\\enesa\\Desktop\\openCV\\Tutorial\\lena_output.jpg"
	image = cv2.imread(image_path, 1) #read image as gray
	cv2.imshow('Result', image)

	print(image)       # output will be all of the N dimensional arrays
	print(type(image)) # output will be : <class 'numpy.ndarray'>
	print(image.dtype) # output will be : uint8
	print(image.shape) # output will be : (512, 512, 3)
	print(image.ndim)  # output will be : 3 (for color image)
	print(image.size)  # output will be : 786432 (512x512x3)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
