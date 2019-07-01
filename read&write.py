import cv2


def main():

	img_path = "C:\\Users\\enesa\\Desktop\\openCV\\Tutorial\\lena_output.jpg"
	img = cv2.imread(img_path, 0) # 0 provides taking img as gray, 
								  # 1 provides taking img as RGB.
	cv2.imshow('Image', img)

	
	output_path = "C:\\Users\\enesa\\Desktop\\openCV\\Tutorial\\lena_output1.jpg"
	cv2.imwrite(output_path, img)


	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
