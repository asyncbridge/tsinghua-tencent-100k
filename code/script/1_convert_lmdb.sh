#!/bin/bash

source config.sh
bin=./../caffe/build/tools/convert_annotation_data2

for type in train test; do
#for type in train_only1; do

    rm -rf $data_dir/lmdb/$type

    $bin $data_dir/annotations.json \
    $data_dir/$type/ids.txt \
    $data_dir/lmdb/$type \
    -resize_height $height -resize_width $width
done
# -resize_height 2048 -resize_width 2048 -height 64 -width 64
