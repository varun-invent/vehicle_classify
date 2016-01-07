-- Preprocess the images
-- sharpen the images coz they are blurred
-- Normalize them feature wise (optional)
-- Save the data and normalization params (mean,std) as vehicle_3_class_data

require 'torch'
require 'xlua'
require 'image'

-- Create a laplacian tensor
-- Apply it on the original image and get the edged image X (let's say)
-- Now add the original image with X 
-- Display the result
-- Display the result after normalization