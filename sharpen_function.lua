require 'nn'
require 'image'
require 'torch'
require 'xlua'


local py = require('fb.python')
py.exec([=[
import cv2
import numpy as np

def sharpen(img):
    img = np.uint8(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img_gray,100,200) 

    # Add both images

    canny_3d = np.dstack((canny,canny,canny))

    sharp_img =  cv2.add(img,canny_3d)

    cv2.imshow('Sharpen Image', sharp_img)
    cv2.waitKey()
    return sharp_img
]=])


function sharpen(img)

    
    print('In sharpen()')

    -- Image read by torch.load() is of size (Channels,rows,cols) but OpenCV expects (rows, cols, channels) therefore transpose() is necessary
    -- moreover we need to scale the pixel values of the image to 0-255 thats why mul(255)


    
    img = img:transpose(1,3):mul(255):transpose(1,2)

    

    sharpen_img = py.eval('sharpen(img)',{img = img})
    print('sharpen image shape ',sharpen_img:size())

    -- Normalize the image 

    sharpen_img:div(255):transpose(1,2):transpose(1,3) -- Converting back to same shape as Lua tensor

    

    return sharpen_img

end


img = image.load('bus.jpg')

x = sharpen(img)
