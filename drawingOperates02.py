import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	# Create a black image
	img1 = np.zeros((512, 512, 3), dtype = np.uint8) # output will be 3x3 zeros matrix 

	w=512
	h=512

	#Lines
	line_up=int(2*(h/5))
	line_down=int(4*(h/5))

	print("Red line y:",str(line_down))
	line_down_color=(255,0,0)
	line_up_color=(255,0,255)
	pt1 =  [0, line_down]
	pt2 =  [w, line_down]
	pts_L1 = np.array([pt1,pt2], np.int32)
	pts_L1 = pts_L1.reshape((-1,1,2))
	cv2.polylines(img1, [pts_L1], False, (0,0,255),2)

	print("Blue line y:",str(line_up))
	pt3 =  [0, line_up]
	pt4 =  [w, line_up]
	pts_L2 = np.array([pt3,pt4], np.int32)
	pts_L2 = pts_L2.reshape((-1,1,2))
	cv2.polylines(img1, [pts_L2], True, (255,0,0),2)


	# Add text to Image 
	text = 'Car Detection App'
	position = (10,50)
	font = cv2.FONT_HERSHEY_SIMPLEX
	size = 0.6
	color = (255,255,255)
	thickness = 1
	cv2.putText(img1, text, position, font, size, color, thickness, cv2.LINE_AA)

	cv2.imshow('Result1', img1)
	
	#plt.imshow(img1)
	#plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
