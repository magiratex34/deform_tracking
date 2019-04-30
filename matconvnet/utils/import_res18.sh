#! /bin/bash

# set this path to your matconvnet root directory
matconvnet_root=/home/eagle/Codes/matlab/matconvnet

# use the import-caffe.py script as converter
converter="python $matconvnet_root/utils/import-caffe.py"

# set paths to caffe model prototxt and weights
#model_prototxt=~/data/models/caffe/vgg_vd_16/VGG_ILSVRC_16_layers_deploy.prototxt
model_prototxt=/home/eagle/Codes/matlab/matconvnet/new_models/resnet-18.prototxt
model_weights=/home/eagle/Codes/matlab/matconvnet/new_models/resnet-18.caffemodel

# set destination for matconvnet model 
import_dir=/home/eagle/Codes/matlab/matconvnet/new_models

# create destination if it doesn't exist
mkdir -pv "$import_dir"

# set the name of the converted matconvnet network file
#out="$import_dir/vgg-vd-16-matconvnet.mat"
out="$import_dir/resnet-18.mat"

# run the converter
$converter \
    --output-format=dagnn \
    --caffe-variant=caffe \
    --caffe-data=$model_weights \
    $model_prototxt \
    $out

