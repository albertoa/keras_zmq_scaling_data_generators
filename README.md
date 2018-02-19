
# Keras training with horizontal scalable data generators and ZeroMQ

## Introduction

This repository is for the Medium's post introducing an architecture where I use ZeroMQ to tie together 1 or more data generators while training Keras models.

The code is based on what I used for the [Kaggle Competition: IEEE's Signal Processing Society - Camera Model Identification](https://www.kaggle.com/c/sp-society-camera-model-identification). This repository is not the code I used for the competition though. This is a very simplified version just to show the ZeroMQ data pipeline.

![Keras ZeroMQ architecture](doc/keras-zmq-diagram1.png?raw=true "Keras ZeroMQ architecture")

This was a particularly helpful competition with lots of shared information. Andres Torrubia shared his code early and this brought a lot of interesting information. I recommend reading some of the [discussion](https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47896) or visit his [github repo](https://github.com/antorsae/sp-society-camera-model-identification) for a full working solution code used in the competition.

## Datasets and symbolic links

I run code from multiple servers and also from the host operating system as well as containers. So that I don't have to constantly modify the source code or add more flags to the command line, I tend to create multiple directories with symbolic links within the code tree and use the configuration file to determine which ones to use.

In this particular code base I left some of this structure to show readers how this can be useful. If you look at the [base configuration](cfg/base.py) you'll notice that the hostname determines if the code is running inside a container or not. For containers the links at dockerlinks/ will be used to find the datasets, otherwise those under nvmelinks/ will be used.

The container image I used comes from Matt Kleinsmith, you can see his [Dockerfile](https://github.com/antorsae/sp-society-camera-model-identification/blob/master/Dockerfile). It was used to run Andres' code.

## Input images

The test and train directories are the standard ones from the [data provided by Kaggle](https://www.kaggle.com/c/sp-society-camera-model-identification/data).

The flickr_images and val_images directories are based on the extra data as proposed by Gleb Posobin. For more information visit [Gleb's post](https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235)

Note that downloading the flickr images can take several attempts. Also, any file ending in upper case (ie. JPG) needs to be renamed to lower case (ie. jpg)

Gleb's URLs for the Motorola-Droid-Maxx include images that are named the same: Motorola-DROID-MAXX-Sample-Images.jpg After Downloading them you need to rename them from the suffix based naming given by wget to something that preserves the suffix as .jpg Here is what I did:

  * Motorola-DROID-MAXX-Sample-Images-1.jpg
  * Motorola-DROID-MAXX-Sample-Images-2.jpg
  * Motorola-DROID-MAXX-Sample-Images-3.jpg
  * Motorola-DROID-MAXX-Sample-Images-4.jpg
  * Motorola-DROID-MAXX-Sample-Images-5.jpg
  * Motorola-DROID-MAXX-Sample-Images-6.jpg
  * Motorola-DROID-MAXX-Sample-Images-7.jpg
  * Motorola-DROID-MAXX-Sample-Images-8.jpg
  * Motorola-DROID-MAXX-Sample-Images-9.jpg
  * Motorola-DROID-MAXX-Sample-Images-10.jpg

## Running

On the Keras machine:

python train_mobilenet_zmq.py

On multiple servers run this at least once per cpu core:

python generator_train_zmq.py

## Purposely bad performant

This code is to illustrate a concept. It loads JPEG files, takes a small patch, maybe performs some manipulation on the image, then passes the patch along. After sending a single patch it moves to the next file. This is a huge waste of CPU resources as loading the image takes significant effort.

To keep enough data flowing to Keras a large number of cores will be needed. Maybe more than what you have at your disposal. However, you should see how adding generators decreases training times and the usefulness of the architecture.

For this type of datasets I actually store the patches of raw and pre-manipulated images into HDF5 files. I use the techniques from Adrian Rosebrock in his book [Deep learning for computer vision with python](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/) You can actually use the same architecture and have a ZeroMQ sink to create the HDF5 databases. Then use the HDF5 files to train Keras.

Even just taking 10-20 patches per image instead of just 1 would be a significant improvement.

## License

All code and documentation in this repository is Copyright (c) 2018 Alberto Alonso and distributed under the MIT license. See [LICENSE](LICENSE) for details.
