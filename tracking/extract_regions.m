function ims = extract_regions(im, boxes, opts)
% Extract the bounding box regions from an input image. 


num_boxes = size(boxes, 1);

crop_mode = opts.crop_mode;
crop_size = opts.input_size;
crop_padding = opts.crop_padding;

ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');

parfor i = 1:num_boxes
    bbox = boxes(i,:);
    crop = im_crop(im, bbox, crop_mode, crop_size, crop_padding);
    ims(:,:,:,i) = crop;
end
