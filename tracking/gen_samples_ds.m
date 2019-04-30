function [bb_samples] = gen_samples_ds(type, bb, n, opts, trans_f, scale_f)
% GEN_SAMPLES
% Generate sample bounding boxes.

bb_top = [bb(1)-bb(3), bb(2:end)];
bb_down = [bb(1)+bb(3), bb(2:end)];
bb_left = [bb(1) bb(2)-bb(4) bb(3:4)];
bb_right = [bb(1) bb(2)+bb(4) bb(3:4)];

bb_samples_center = gen_samples_sub(type, bb, floor(3 * n / 8), opts, trans_f, scale_f);
bb_samples_top = gen_samples_sub('gaussian', bb_top, floor(n / 8), opts, trans_f, scale_f);
bb_samples_down = gen_samples_sub('gaussian', bb_down, floor(n / 8), opts, trans_f, scale_f);
bb_samples_left = gen_samples_sub('gaussian', bb_left, floor(n / 8), opts, trans_f, scale_f);
bb_samples_right = gen_samples_sub('gaussian', bb_right, floor(n / 8), opts, trans_f, scale_f);
bb_samples_whole = gen_samples_sub('xwindow', bb, n - floor(n * 3 / 8) + 4 * floor(n / 8), ...
                                    opts, trans_f, scale_f);

bb_samples = [bb_samples_center; bb_samples_top; bb_samples_right; ...
    bb_samples_left; bb_samples_down; bb_samples_whole];
