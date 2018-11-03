#!/bin/bash
source config.sh
bin=./../caffe/build/tools/compute_driving_mean

for type in train test; do
    $bin \
    $data_dir/lmdb/"$type" \
    $data_dir/lmdb/"$type"_mean.binaryproto lmdb
done

#./../build/tools/compute_driving_mean \
#  /media/randon/LENOVO/data/tencent/pl120_train_lmdb \
#  /media/randon/LENOVO/data/tencent/pl120_train_mean.binaryproto lmdb
