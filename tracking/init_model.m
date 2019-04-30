function [net_conv, net_fc, opts] = init_model(image, net, mode)

%% set opts
% use gpu
opts.useGpu = true;

% model def
opts.net_file = net;

% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

% bounding box regression
opts.bbreg = true;
opts.bbreg_nSamples = 1000;

% learning policy
opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

% initial training policy
% opts.learningRate_init = 0.0001; % x10 for fc6
opts.learningRate_init = 3e-4; % x10 for fc6
opts.maxiter_init = 30;
% opts.maxiter_init = 50;

opts.nPos_init = 500;
opts.nNeg_init = 5000;
opts.posThr_init = 0.7;
opts.negThr_init = 0.5;

% update policy
opts.learningRate_update = 0.0003; % x10 for fc6
% opts.learningRate_update = 1e-4; % x10 for fc6
opts.maxiter_update = 10;

opts.nPos_update = 50;
opts.nNeg_update = 200;
opts.posThr_update = 0.7;
opts.negThr_update = 0.3;

opts.update_interval = 10; % interval for long-term update

% data gathering policy
opts.nFrames_long = 100; % long-term period
opts.nFrames_short = 20; % short-term period

% cropping policy
opts.input_size = 107;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;

% scaling policy
opts.scale_factor = 1.05;

% sampling policy
opts.nSamples = 256;
opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

% set image size
opts.imgSize = size(image);

%% load net
obj_struct = load(opts.net_file);

if strcmp(mode, 'dagnn')
    load(opts.net_file);
    net_conv = net_obj;
    layernames = {net_obj.layers(:).name};
    lay_id = find(strcmp(layernames,'relu3'));
    varnames = {net_obj.vars(:).name};
    var_id = find(strcmp(varnames,'x10'));
    paramnames = {net_obj.params(:).name};
    param_id = find(strcmp(paramnames, 'conv3b'));
    
    net_conv.layers = net_obj.layers(1:lay_id);
    net_conv.params = net_obj.params(1:param_id);
    net_conv.vars = net_obj.vars(1:var_id);
    net_conv.meta = net_obj.meta;
    
    net_conv = dagnn.DagNN.loadobj(net_conv);
    
    net_fc = net_obj;
    net_fc.layers = net_obj.layers(lay_id+1:end);
    net_fc.params = net_obj.params(param_id+1:end);
    net_fc.vars = net_obj.vars(var_id+1:end);
    net_fc.meta = net_obj.meta;
    
    net_fc = dagnn.DagNN.loadobj(net_fc);
    net_conv.renameVar('input', 'x0');
    for tag = 0:7        
        net_fc.renameVar(['x1',num2str(tag)], ['x',num2str(tag)]);
    end

    net_fc.renameVar('objective', 'x8');
    
    % reset the projection matrix to zeros
    f = net_fc.getParamIndex('st_fc_outf');
    fsz = size(net_fc.params(f).value);
    net_fc.params(f).value = zeros(fsz, 'single');
    
    
    net_fc.move('gpu');
    net_conv.move('gpu');
else
    net = load(opts.net_file);
    if isfield(net,'net'), net = net.net; end
    net_conv.layers = net.layers(1:10);
    net_fc.layers = net.layers(11:end);

    net_conv = dagnn.DagNN.fromSimpleNN(net_conv);
    net_fc = dagnn.DagNN.fromSimpleNN(net_fc);
    
    net_fc.params(1).learningRate = 1;
    net_fc.params(2).learningRate = 2;
    net_fc.params(2).weightDecay = 0;
    net_fc.params(3).learningRate = 1;
    net_fc.params(4).learningRate = 2;
    net_fc.params(4).weightDecay = 0;
    net_fc.params(5).learningRate = 10;
    net_fc.params(6).learningRate = 20;
    net_fc.params(6).weightDecay = 0;
    
    
    net_fc.move('gpu');
    net_conv.move('gpu');
end





end