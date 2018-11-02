# Tsinghua-Tencent 100K

[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)

Tsinghua-Tencent-100K is a benchmark for traffic-sign detection and classification. It provides 100,000 images containing 30,000 traffic-sign instances.
It is focused on small object detection in real world. According to the paper, it said "A typical traffic-sign might be say 80 × 80 pixels, in a 2000 × 2000 pixel image, or just 0.2% of the image.".
It means that the small objects such as traffic-sign occupy the very small proportion in input image.

This benchmark is built by Caffe and installed in Ubuntu 14.04.  
I summarized how to setup this benchmark and learning/testing the CNN model which is proposed from this paper. I also added some codes for my experiment.  

## Table of Contents
* [Requirements: Hardware](#requirements-hardware)
* [Requirements: Software](#requirements-software)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Training and Evaluation](#training-and-evaluation)
* [License and Citation](#license-and-citation)
* [References](#references)


## Requirements: Hardware
I used Nvidia Geforce GTX 1080 Ti for learning the CNN model of this paper. This graphic card spec is as follows.  
Please refer to full specs.(https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/#specs)

- GPU Architecture 	 : Pascal  
- NVIDIA CUDA® Cores : 3584  
- Boost Clock (MHz)  : 1582  
- Memory Spec 	     : 11 GB GDDR5X

You need to check the compute capability version to install correct CUDA SDK version.  
Please refer to https://en.wikipedia.org/wiki/CUDA#Supported_GPUs and find correct CUDA SDK version.  

In my case, I should install CUDA SDK 8.0 version. [CUDA compute capabilities >= 6.1](https://en.wikipedia.org/wiki/CUDA#Supported_GPUs)  
According to CUDA SDK 8.0 version, I used cuDNN 6.0 version.

## Requirements: Software
The prerequisites are as follows.

- Ubuntu 14.04.5 LTS (Trusty Tahr)  
- Nvidia driver 390.87
- CUDA SDK 8.0  
- cuDNN 6.0  
- Python 2.7.x  
- Jupyter Notebook based on Python 2.7.x

## Installation
To install Nvidia driver, CUDA and cuDNN, you need to sign-in to Nvidia web site.  
If you didn't sign-up, please do sign-up and try to download them.

### Install Ubuntu 14.04.5 LTS (Trusty Tahr)
- I downloaded 64-bit PC (AMD64) server install image from [Ubuntu 14.04.5 LTS (Trusty Tahr)](http://mirror.kakao.com/ubuntu-releases/trusty/) and installed the iso image to my PC.

### Install GPU Driver
- I downloaded and installed an official Nvidia driver version [390.87](https://www.nvidia.com/download/driverResults.aspx/137276/en-us)  

```bash
sudo sh NVIDIA-Linux-x86_64-390.87.run
```

### Install CUDA SDK
- I downlaoded the CUDA SDK 8.0 version from [CUDA SDK 8.0 GA2](https://developer.nvidia.com/cuda-80-ga2-download-archive).  
- Please visit to above link and select > Linux > x86_64 > Ubuntu > 14.0 > deb (local) and then click on the download button.

```bash
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
sudo reboot
```

- Please add path information to .bashrc file. The .bashrc file is located in /~ (home directory).

```bash
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

```

### Install cuDNN
- I downloaded the cuDNN 6.0 version from [cuDNN 6.0](https://developer.nvidia.com/rdp/cudnn-archive).
- Please visit to above link and select "Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0" and then click on "cuDNN v6.0 Library for Linux".
- The cudnn-8.0-linux-x64-v6.0-tgz file will be downloaded. Please decompress the tgz file and copy cuDNN include and library files to /usr/local/cuda/ directory as follows.

```bash
tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64
sudo cp include/cudnn.h /usr/local/cuda/include
```

### Install Caffe
- According to [Caffe Installation Guide for Ubuntu O/S](http://caffe.berkeleyvision.org/install_apt.html), please install prerequisites. [For Ubuntu (< 17.04)](http://caffe.berkeleyvision.org/install_apt.html)
- General dependencies  

```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
```

- BLAS
I just installed both of them.  

```bash
sudo apt-get install libatlas-base-dev
sudo apt-get install libopenblas-dev
```

- Python
I updated python version to 2.7.12.

```bash
sudo apt-get install python-dev
```

- Remaining dependencies, 14.04  

```bash
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```

- Build Caffe and pycaffe
According to [Tsinghua-Tencent-100K Tutorial](https://cg.cs.tsinghua.edu.cn/traffic-sign/tutorial.html), download Caffe code.  

```bash
mkdir TT100K && cd TT100K
wget http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/code.zip
unzip code.zip
```

In my case, code.zip is not decompressed well on my Ubuntu. So I converted code.zip to code.7z in Windows O/S and then it worked well.

Please copy the files in python and script folders from my Git to above code/python and code/script directory.  

```Shell
git clone https://github.com/asyncbridge/tsinghua-tencent-100k.git
```

You can see below folder structure.  

```Shell
    |__ code
        |__ caffe
            |__ ...					
        |__ model
            |__ snapshots
            |__ model.caffemodel
            |__ model.prototxt
            |__ solver.prototxt
            |__ train_val.prototxt
        |__ python		
            |__ my-data-show.ipynb	
            |__ my-deployer.ipynb
            |__ my-eval-graph.ipynb
            |__ my-eval.ipynb			
            |__ ...						
        |__ results
            |__ ours_result_annos.json
            |__ ...			
        |__ script			
            |__ 1_convert_lmdb.sh
            |__ 2_calc_img_mean.sh
            |__ clean.sh
            |__ compile.sh
            |__ config.sh
            |__ prepare.sh
            |__ train.sh
```

Next, please build Caffe and pycaffe.  

```bash
cd code/script
./compile.sh
```

### Install Jupyter Notebook
- To evaluate the test result, we need to use Jupyter Notebook.
- Please install Jupiter Notebook according to guide [Installing the Jupyter Notebook](http://jupyter.org/install)  
- This benchmark is based on Python 2.7.x and we have to install it as follows.  

If you have Python 2 installed:
```bash
python -m pip install --upgrade pip
python -m pip install jupyter
```

## Data Preparation
- Download TT100K dataset
Please download the TT100K dataset and decompress zip file. 

```bash
cd TT100k
wget http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip
unzip data.zip
```

- To train the dataset, we need to convert the dataset to LMDB format and calculate the image mean.  

```bash
mkdir -p ../../data/lmdb
./1_convert_lmdb.sh
./2_calc_img_mean.sh
```

## Training and Evaluation
### Training  
After doing data preparation, we can start the training process.  

```bash
cd code/script
./train.sh
```

The train.sh shell file is described as follows. You can also get the training log file.

```bash
../caffe/build/tools/caffe train --solver ../model/solver.prototxt --gpu 0 2>&1 | tee tt100k_training_01.log
```

### Evaluation  
- Before evaluation, you need to install Jupyter Notebook.  
- You need to install python dependencies such as opencv, matplotlib to evaluate the test result.  
- Please install correct dependencies if you see some errors when running python code on Jupyter Notebook.  

```bash
sudo pip install opencv-python
sudo pip install matplotlib
...
```

- For details, please refer to [Tsinghua-Tencent 100K Tutorial](https://cg.cs.tsinghua.edu.cn/traffic-sign/tutorial.html)  
- my-*.ipynb files will be updated. This is because I'm experimenting this dataset.  

### How to observe TT100K data sample  
Please run [my-data-show.ipynb](https://github.com/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-data-show.ipynb) on Jupyter Notebook.([See my-data-show.ipynb](https://nbviewer.jupyter.org/github/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-data-show.ipynb))

### How to deploy the training result of TT100K annotation JSON file  
After training the dataset, the deployer creates annotation JSON file using with .caffemodel file.    
Please run [my-deployer.ipynb](https://github.com/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-deployer.ipynb) on Jupyter Notebook.([See my-deployer.ipynb](https://nbviewer.jupyter.org/github/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-deployer.ipynb))

### How to evaluate the annotation JSON file  
Please run [my-eval.ipynb](https://github.com/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-eval.ipynb) or [my-eval-graph.ipynb](https://github.com/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-eval-graph.ipynb) on Jupyter Notebook.([See my-eval.ipynb](https://nbviewer.jupyter.org/github/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-eval.ipynb), [my-eval-graph.ipynb](https://nbviewer.jupyter.org/github/asyncbridge/tsinghua-tencent-100k/blob/master/code/python/my-eval-graph.ipynb))  
You can evaluate the pre-created annotation JSON file.  

```Shell
    |__ code				
        |__ results
            |__ ours_result_annos.json
            |__ ...			
```

## License and Citation

This project is made available under the [MIT License](https://github.com/asyncbridge/tsinghua-tencent-100k/blob/master/LICENSE).

It is cited from:

    @InProceedings{Zhe_2016_CVPR,
	author = {Zhu, Zhe and Liang, Dun and Zhang, Songhai and Huang, Xiaolei and Li, Baoli and Hu, Shimin},
	title = {Traffic-Sign Detection and Classification in the Wild},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2016}
	}
	

## References
- https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.html  
- https://cg.cs.tsinghua.edu.cn/traffic-sign/  
- https://en.wikipedia.org/wiki/CUDA#Supported_GPUs  
- https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/#specs  
- https://gist.github.com/AlexanderFabisch/6d826b62af87e3c0ac6f  
- http://mirror.kakao.com/ubuntu-releases/trusty/  
- https://www.nvidia.com/download/driverResults.aspx/137276/en-us  
- https://developer.nvidia.com/cuda-downloads  
- https://developer.nvidia.com/rdp/cudnn-archive  
