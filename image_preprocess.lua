-- Preprocess the images
-- sharpen the images coz they are blurred
-- Normalize them feature wise (optional)
-- Save the data and normalization params (mean,std) as vehicle_3_class_data

require 'torch'
require 'xlua'
require 'image'


-- Apply canney edge detector on the original image and get the edged image X (let's say)
-- Now add the original image with X 
-- Display the result
-- Display the result after normalization

local py = require('fb.python')

py.exec([=[
import cv2
import numpy as np

def sharpen(img):
	print 'shape of the img received my sharpen function is ',img.shape
	#img =  cv2.imread(img_name)
	print 'datatype of received tensor is ',img.dtype
	img = np.uint8(img)   # converting the signed to unsigned coz cvtColor works with unsigned images
	cv2.imshow('Bus Image',img)
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	cv2.imshow('Bus gray',img_gray)
	canny = cv2.Canny(img_gray,100,200)
	canny =  np.divide(canny,7)
	cv2.imshow('Canny',canny)
	
	# Add both images
	canny_3d = np.dstack((canny,canny,canny))
	print 'Size of canny_3d ',canny_3d.shape
	print 'Size of img ',img.shape
	sharp_img = cv2.add(img,canny_3d)
	cv2.imshow('sharpen',sharp_img)
	cv2.waitKey(0)
	return np.float64(sharp_img)
]=])



-- Image read by torch.load() is of size (Channels,rows,cols) but OpenCV expects (rows, cols, channels) therefore transpose() is necessary
-- moreover we need to scale the pixel values of the image to 0-255 thats why mul(255)


img = image.load('bus.jpg'):transpose(1,3):mul(255):transpose(1,2)

sharpen_img = py.eval('sharpen(img)',{img = img})
print('sharpen image shape ',sharpen_img:size())

-- Normalize the image 

sharpen_img:div(255):transpose(1,2):transpose(1,3) -- Converting back to same shape as Lua tensor


-- The following code checks if the sharpened image converted to lua tensor is still a decent image or not
-- There is some problem with displaying the image in lua(Some colorspace problem maybe), So i decided to display using opencv

-- Take the lua image tensor pass it to the python opencv script 

--Python opencv script

py.exec([=[
import cv2
import numpy as np

def display(img):
	
	img = np.uint8(img)   # converting the signed to unsigned coz cvtColor works with unsigned images
	cv2.imshow('Image_displayed again',img)
	
	cv2.waitKey(0)
	
]=])

sharpen_img:mul(255):transpose(1,3):transpose(1,2)
py.eval('display(img)',{img = sharpen_img})