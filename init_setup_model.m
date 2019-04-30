%% Compile MatConvNet

% run matconvnet/matlab/vl_setupnn ;
% cd matconvnet;
% vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/usr/local/cuda-8.0', ...
%                'cudaMethod', 'nvcc');
% cd ..;

%%
if(isempty(gcp('nocreate')))
    parpool;
end

run matconvnet/matlab/vl_setupnn ;

addpath('tracking');
addpath('utils');


