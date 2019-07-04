img = cv2.imread(path, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

filter1 = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), np.float32) #Sharpening Filter
output = cv2.filter2D(img, -1, filter1) #convolution filter

blur = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,170,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

mask = cv2.erode(thresh, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)

def tracking(mask, img):
	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	_, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print("{} Objects have detected!".format(len(cnts))) 
	
	for i in cnts:
		x,y,w,h = cv2.boundingRect(i)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)