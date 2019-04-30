function [net, poss, hardnegs] = online_train(net,pos_data,neg_data,varargin)

randSeed = 0;
randStream = parallel.gpu.RandStream('CombRecursive', 'Seed', randSeed);
parallel.gpu.RandStream.setGlobalStream(randStream);

opts.useGpu = true;
opts.conserveMemory = true ;
opts.sync = true ;

opts.maxiter = 30;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts.batchSize_hnm = 256;
opts.batchAcc_hnm = 4;

opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;


% opts = vl_argparse(opts, varargin) ;
for i = 1 : numel(varargin)
    if strcmp(varargin{i},'maxiter')
        opts.maxiter=varargin{i+1};
    elseif strcmp(varargin{i}, 'learningRate')
        opts.learningRate = varargin{i+1};
    end
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i = 1 : numel(net.params)
    net.params(i).momentum = ...
            gpuArray(zeros(size(net.params(i).value), 'single')) ;
end


%% initilizing
if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
res = [] ;

n_pos = size(pos_data,4);
n_neg = size(neg_data,4);
train_pos_cnt = 0;
train_neg_cnt = 0;

% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

% extract negative batches
train_neg = [];
remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter-length(train_neg);
end

% learning rate
lr = opts.learningRate ;

% for saving positives
poss = [];

% for saving hard negatives
hardnegs = [];

% objective fuction
objective = zeros(1,opts.maxiter);

%% training on training set
for t=1:opts.maxiter
    iter_time = tic ;
    
    % ----------------------------------------------------------------------
    % hard negative mining
    % ----------------------------------------------------------------------
    score_hneg = zeros(opts.batchSize_hnm*opts.batchAcc_hnm,1);
    hneg_start = opts.batchSize_hnm*opts.batchAcc_hnm*(t-1);
    for h=1:opts.batchAcc_hnm
        batch = neg_data(:,:,:,...
            train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm));
        if opts.useGpu
            batch = gpuArray(batch) ;
        end
        
        % backprop
        net.vars(end-2).precious = 1; 
        net.vars(10).precious = 1;
        net.mode = 'test';
        net.eval({'x0', batch, 'label', gpuArray(ones(opts.batchSize_hnm,1,'single'))});
        res = net.vars(end-2).value;
        
        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = ...
        squeeze(gather(res(1,1,2,:)));
        

    end
    [~,ord] = sort(score_hneg,'descend');
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = neg_data(:,:,:,hnegs);
    hardnegs = [hardnegs; hnegs];
    
    % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    poss = [poss; train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)];
    
    batch = cat(4,pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)),...
        im_hneg);
    labels = [2*ones(opts.batch_pos,1,'single');ones(opts.batch_neg,1,'single')];
    if opts.useGpu
        batch = gpuArray(batch) ;
        labels = gpuArray(labels);
    end
    
    inputs = {'x0', batch, 'label', labels};
    net.vars(end).precious = 1;
    net.mode = 'normal';
    net.eval(inputs, {'x8', one});
    
    res = net.vars(end).value;
    
    % gradient step
    for l = 1:numel(net.params)
        term1 = opts.momentum * net.params(l).momentum ;
        term2 = (lr * net.params(l).learningRate) * ...
                (opts.weightDecay * net.params(l).weightDecay) * ...
                net.params(l).value;
        term3 = (lr * net.params(l).learningRate) / opts.batchSize * net.params(l).der;
        net.params(l).momentum = term1 - term2 - term3;
        net.params(l).value = net.params(l).value + net.params(l).momentum;
    end
    
    
    % print information
   
    objective(t) = gather(res) / opts.batchSize;
    iter_time = toc(iter_time);
    
end % next batch

