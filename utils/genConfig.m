function [ config ] = genConfig(dataset,seqName)



config.dataset = dataset;
config.seqName = seqName;

switch(dataset)
    case {'demo'}
        seqdir = './data/';
        config.imgDir = fullfile(seqdir, config.seqName, 'img');
        if(~exist(config.imgDir,'dir'))
            error('%s does not exist!!',config.imgDir);
        end
        config.imgList = parseImg(config.imgDir);
        gtPath = fullfile(seqdir, seqName, [seqName,'.txt']);
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        gt = importdata(gtPath);
        config.gt = gt;
        
        nFrames = min(length(config.imgList), size(config.gt,1));
        config.imgList = config.imgList(1:nFrames);
        config.gt = config.gt(1:nFrames,:);
        
    
end
