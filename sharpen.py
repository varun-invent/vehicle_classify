# Code to sharpen the image

import cv2
import numpy as np

def sharpen(img):
	img =  cv2.imread('bus.jpg')
	cv2.imshow('Bus Image',img)
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow('Bus gray',img_gray)
	canny = cv2.Canny(img_gray,100,200)
	cv2.imshow('Canny',canny)
	
	# Add both images
	canny_3d = np.dstack((canny,canny,canny))
	print 'Size of canny_3d ',canny_3d.shape
	print 'Size of img ',img.shape
	sharp_img = cv2.add(img,canny_3d)
	# cv2.imshow('sharpen',sharp_img)
	# cv2.waitKey(0)
	return sharp_img
