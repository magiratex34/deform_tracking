function [ feat ] = extract_deepfeat(net, img, boxes, opts)
% Extract CNN features from bounding box regions of an input image.


n = size(boxes,1);
ims = extract_regions(img, boxes, opts);
nBatches = ceil(n/opts.batchSize_test);

for i=1:nBatches
    
    batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    net.mode = 'test';
    net.eval({'x0', batch});
    
    f = gather(net.vars(end).value);
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
end