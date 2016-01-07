-- Read random 1000 images from each of the following folder :
-------------------------------surveillance_data/car
-------------------------------surveillance_data/three_wheeler
-------------------------------surveillance_data/two_wheeler
-- Resize it to 30 X 30



require 'nn'
require 'image'
require 'torch'
require 'xlua'



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

--local img_counter = torch.Tensor(3):fill(0)

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

                    dataset.images[{{img_counter},{},{},{}}] = img
                    dataset.labels[img_counter] = i
                    
                end) 
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
    










-- nImagePlanes = 3 -- rgb
-- kitti_stereo_dataset = {
-- image_left = torch.Tensor(nImages,nImagePlanes,imgRows,imgCols),
-- image_right = torch.Tensor(nImages,nImagePlanes,imgRows,imgCols),
-- image_gt_depth = torch.Tensor(nImages,1,imgRows,imgCols)}


-- --train_img_dataset = {}
-- --gt_train_img_dataset = {}

-- i=1

-- for j,img_name in ipairs(train_images_list_l) do
--     index,_ = string.find(img_name,'.png')
--     if index ~= nill then
--         --print(img_name)
--         if img_name:sub(index-1,index-1) == '0' then
--             print('Image stored' .. i)
--             --print(image.crop(image.load(train_data_path_l .. img_name),x1,y1,x2,y2):size())

--             --image.display(image.crop(image.load(train_data_path_l .. img_name),x1,y1,x2,y2))
--             kitti_stereo_dataset.image_left[i] = image.crop(image.load(train_data_path_l .. img_name),x1,y1,x2,y2)    
--         	--print(kitti_stereo_dataset.image_left[1]:size())
--         	--image.display(kitti_stereo_dataset.image_left[1])
--             kitti_stereo_dataset.image_right[{{i},{},{},{}}] = image.crop(image.load(train_data_path_r .. train_images_list_r[j]),x1,y1,x2,y2)    
--             kitti_stereo_dataset.image_gt_depth[{{i},{1},{},{}}] = image.crop(image.load(gt_train_path .. gt_train_images_list[i+2]),x1,y1,x2,y2)
--             i = i+1    
--         end
--     end
--     if i == maxImages+1 then
--         break
--     end
    
-- end


-- print('Images stored in table')

-- --torch.save('kitti_stereo_dataset_100.dat',torch.Tensor(kitti_stereo_dataset))
-- torch.save('kitti_stereo_dataset_190_cropped.dat',kitti_stereo_dataset)
-- print('Tensor stored to disc.')
-- --image.display(train_img_dataset[3])
-- --image.display(gt_train_img_dataset[3])
