#2018.11.01, train.sh file
../caffe/build/tools/caffe train --solver ../model/solver.prototxt --gpu 0 2>&1 | tee tt100k_training_01.log
