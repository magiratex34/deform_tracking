function [ feat ] = classifer(net, ims, opts)
% object classification

n = size(ims,4);
nBatches = ceil(n/opts.batchSize);


for i=1:nBatches
    
    batch = ims(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end

    net.mode = 'test';
    net.eval({'x0', batch});
    fid = net.getVarIndex(net.layers(end).inputs{1});
    f = gather(net.vars(fid).value);
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i)) = f;
    
end