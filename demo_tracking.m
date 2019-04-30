clc
clear;
close all;

global test_seq;  
% test_seq = 'Skating2-2';
test_seq = 'Skiing';

trained_model = 'vot_deform_model.mat';
net = fullfile('models',trained_model);

conf = genConfig('demo', test_seq);
result = run_deformnet(conf.imgList, conf.gt(1,:), net,1);
save(['./results/', test_seq, '.mat'], 'result');
gt = [conf.gt(:,1)+conf.gt(:,3)/2, conf.gt(:,2)+conf.gt(:,4)/2];
res = [result(:,1)+result(:,3)/2, result(:,2)+result(:,4)/2];
error = sqrt(sum((gt - res).^2,2));

fprintf('\n...........\nAverage error is %.3f\n...........\n', mean(error));
fprintf('...........std error is %.3f\n...........\n', std(error));    
