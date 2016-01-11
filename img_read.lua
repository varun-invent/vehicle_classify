-- Read random 1000 images from each of the following folder :
-------------------------------surveillance_data/car
-------------------------------surveillance_data/three_wheeler
-------------------------------surveillance_data/two_wheeler
-- Resize it to 30 X 30



require 'nn'
require 'image'
require 'torch'
require 'xlua'
local py = require('fb.python')




imgCols =  30
imgRows = 30
nImages =  3000 --1000 per class
nImagePlanes = 3 --RGB



data_path = {
'surveillance_data/vehicle_data/car/',
'surveillance_data/vehicle_data/two_wheeler/',
'surveillance_data/vehicle_data/three_wheeler/'
}

dataset = {
    images = torch.Tensor(nImages,nImagePlanes,imgRows,imgCols),
    labels = torch.Tensor(nImages)
}


local py = require('fb.python')
py.exec([=[
import cv2
import numpy as np

def sharpen(img):
    img = np.uint8(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img_gray,100,200) - 200
    canny =  np.divide(canny,8) # to reduce the affect of sharpening
    # Add both images

    canny_3d = np.dstack((canny,canny,canny))

    sharp_img =  cv2.add(img,canny_3d)

    #cv2.imshow('Sharpen Image', sharp_img)
    #cv2.waitKey()
    return np.float64(sharp_img)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
]=])

function sharpen(img)


    -- print('In sharpen()')

    -- Image read by torch.load() is of size (Channels,rows,cols) but OpenCV expects (rows, cols, channels) therefore transpose() is necessary
    -- moreover we need to scale the pixel values of the image to 0-255 thats why mul(255)



    img = img:transpose(1,3):mul(255):transpose(1,2)

    

    sharpen_img = py.eval('sharpen(img)',{img = img})
    --print('sharpen image shape ',sharpen_img:size())

    -- Normalize the image 

    sharpen_img:div(255):transpose(1,2):transpose(1,3) -- Converting back to same shape as Lua tensor
    --sharpen_img:div(255)
    

    return sharpen_img

end


local img_counter = 0
for i, path in ipairs(data_path) do
    xlua.progress(i,3)   -- # 3 are the number of directories to be read
    local images_list  = paths.dir(path)
    --  create the dataset
    for j =  1, #images_list do
        --xlua.progress(img_counter,1000)
        img_name = path .. images_list[j]
        index,_ = string.find(img_name,'.jpg')
            if index ~= nill then
                --print(img_name)
                
                local status = pcall( -- Exception handling if image cant be loded
                function () 
                    local img = image.load(img_name) 
                    
                    img = image.scale(img,30,30)
                    
                    img_counter = img_counter + 1
                    -- Call a function to Sharpen the image 'img'
                    img = sharpen(img)
                    img:float()
                    dataset.images[{{img_counter},{},{},{}}] = img
                    dataset.labels[img_counter] = i
                    
                end) 
                -- local img = image.load(img_name) 
                
                -- img = image.scale(img,30,30)
                
                -- img_counter = img_counter + 1
                -- -- Call a function to Sharpen the image 'img'
                -- img = sharpen(img)
                -- dataset.images[{{img_counter},{},{},{}}] = img
                -- dataset.labels[img_counter] = i

                if (img_counter % 1000) == 0 then
                    break 
                end 
                    

                --print('Size of image is ', img:size())
            end
    end
end           

print('dataset Created of size ',dataset)

print('==> Saving the dataset')
torch.save('vehicle_data_3_class.dat',dataset)
    








