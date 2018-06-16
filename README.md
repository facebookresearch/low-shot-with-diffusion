# Low-shot learning with large-scale diffusion

This repository contains code associated with the following paper:<br>
[Low-shot learning with large-scale diffusion](https://arxiv.org/abs/1706.02332v2) <br>
Matthijs Douze, Arthur Szlam, Bharath Hariharan, Herv&eacute; J&eacute;gou <br>
CVPR'18.

## Prerequisites

This code uses [pytorch](http://pytorch.org/), [Intel MKL](https://software.intel.com/en-us/mkl) and [faiss](https://github.com/facebookresearch/faiss). 
It also uses SWIG and a C++11 compiler.
It requires GPUs and Cuda.
The easiest way to get all these is to use the anaconda install, as described in [the Faiss install file](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

In terms of data and datasets, it relies on:

- Imagenet 2012, that comes as 1.2M training images and 50k validation images

- The [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) dataset.

- the pre-trained networks from [the code acompaining Low-shot Learning by Shrinking and Hallucinating Features](https://github.com/facebookresearch/low-shot-shrink-hallucinate)

Both datasets have their distribution rules and should be obtained from their respective websites.
We provide many intermediate results as links to make the reproduction easier.

## Running the code

Running a low-shot learning experiment involves:
1.  Extract features for the background set
2.  Extract features for the training and test sets
3.  Classify using logistic regression baseline
4.  Construct k-nn graph for the background set
5.  Construct the full knn-graph; perform diffusion from training to test features.


## Extracting features for background images

The feature extraction uses the precomputed model from "Low-shot Learning by Shrinking and Hallucinating Features". 
First assume we have a directory for the data (given in env var DDIR): 
```
cd $DDIR
wget https://s3-us-west-1.amazonaws.com/low-shot-shrink-hallucinate/models.zip
unzip models.zip
```
The model we are interested in is `models/checkpoints/ResNet50/89.tar`, which is not a tar file despite the name.

We assume the images from the dataset are numbered from 0 to 10^8 - 1, videos and unreadable images are ignored. 
The images are numbered in the same way as the YFCC100M metadata file. 
Image number 12345678.jpg is stored as 678.jpg in a zipfile `$DDIR/yfcc100m/12/345.zip` (uncompressed). 

Feature exrtraction can be performed with 
```
python save_features.py \
           --cfg f100m_save_data.yaml \
           --modelfile $DDIR/models/checkpoints/ResNet50/89.tar \
           --outfile $DDIR/features/f100m/block0.hdf5 \
           --i0 0 --i1 1000000 \
           --model ResNet50
```
This will store the descriptors for the first 1M files (a matrix of 1M * 2048) in the features directory.

Precomputed descriptors for bloc0 and block1 (first 2M images) are available here:

[block0.hdf5](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/features/f100m/block0.hdf5)

[block1.hdf5](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/features/f100m/block1.hdf5)

The descriptors are then PCA'd down to 256 dimensions with 

```
python train_apply_pca.py train
python train_apply_pca.py apply
```

The output is a matrix of size 100M * 256 that is stored in raw binary format, this is the data that is actually used in experiments.
The PCA transform (in Faiss format) and the large matrix are available at:

[PCAR256.vt](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/features/f100m/PCAR256.vt)

[concatenated_PCAR256.raw](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/features/f100m/concatenated_PCAR256.raw)

## Extracting the features for Imagenet

The features for Imagenet are extracted with the original code from Hariharan et al in the package https://github.com/facebookresearch/low-shot-shrink-hallucinate. 
The features from the training and validation parts of Imagenet are provided here: 

[train.hdf5](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/features/train.hdf5)
[val.hdf5](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/features/val.hdf5)

Since they are relatively small, the PCA transformation is applied on-the-fly.

## Computing the logistic regression baseline

### Running the test for one set of parameters

The test can be run with:

```
python logreg.py \
     --mode test --nlabeled 2 --seed 1 \
     --maxiter 29500 --lr 0.001 --wd 0.01 --batchsize 128 
```
The meaning of the parameters is: 
- `--mode test`: run an experiment on the test dataset (not the validation) 
- `--nlabeled 2`: use all training images for the base classes and 2 training images for novel classes
- `--seed 1`: use random seed 1 to select the 2 training images (in a whole experiment, the resulting accuracy is averaged over seeds 1 to 5)
- the following options are hyper-parameters that were optimized on the validation set.

The output should be like [this GIST](https://gist.github.com/mdouze/2bc5da096a4ed0ee69a05ac98c56cc0d). 
The final top-5 accuracy is 0.700, which is one of the 5 accuracies averaged to get the "logistic regressoin" column in table 3 of the paper. 
Similarly, the 0.565 number is reported in table 5 (novel classes) of the supplementary material.

### Run script

The whole set of operations, including hyper-parameter selection is performed by a shell script that has 4 phases: 
```
bash run_logreg.bash 1
bash run_logreg.bash 2
bash run_logreg.bash 3
bash run_logreg.bash 4
```

Since these operations are quite costly, it is worthwhile to parallelize the script to submit jobs on a cluster instead of doing the loops explicitly.

The final output looks like [this GIST](https://gist.github.com/mdouze/edb0b10ab2042c39355bb5c4307ba0b8)

## Constructing the knn-graph for the background set

The script 
```
python build_graph.py 1000000
```
builds the background knn-graph for 1M descriptors. 
It uses all available GPUs on the machine for that. 
For 100M background images, it needs 8 GPUs. 
The graph is stored as `$DDIR/knngraph/ndis1000000_k32_I_11.int32` and `$DDIR/knngraph/ndis1000000_k32_D_11.float32` respectively.
The `I_11` represents the edge links and `D_11` the edge weights.

The Faiss index and the graph are also available through the links

[ndis1000000.index](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/knngraph/ndis1000000.index)

[ndis1000000_k32_I_11.int32](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/knngraph/ndis1000000_k32_I_11.int32)

[ndis1000000_k32_D_11.float32](https://s3-us-west-1.amazonaws.com/low-shot-diffusion/knngraph/ndis1000000_k32_D_11.float32)

The graph is provided for ndis=100k, 1M, 10M, 100M, jsut replace the number in the filename. 

## Running label propagation experiments

### Compiling the C++ library

A small C++ library (graph_utils) that makes the processing affordable by reducing the memory usage and parallelizing some operations.
It also uses MKL's sparse matrix-dense matrix product, which is the central operation of the diffusion.

To compile it, make sure g++ and SWIG (version > 3) are available, edit the MKL path in `compile.bash`, and run 
```
bash compile.bash
python test_matmul.py
```

The test_matmul script tests that the code was compiled properly and outputs a few error values that should be < 1e-6.

### Running the diffusion experiment

On diffusion experiment can be run with:
```
python diffusion.py \
     --mode test --nlabeled 2 --seed 1 \
     --nbg 1000000 --k 30 \
     --niter 3
```
The parameters are:
- `--mode test --nlabeled 2 --seed 1`: see the `logreg.py` experiment
- `--nbg 1000000`: number of background images to run the diffusion on
- `--k 30`: number of neighbors per node to consider
- `--niter 3`: number of diffusion iterations, this is the (single) hyper-parameter of the method

The output is given in [this GIST](https://gist.github.com/mdouze/a204a3460393a630a8708a2a4f01f5fb). 
The program is logging lots of information about runtime and memory usage (critical for larger graphs). 
It operates by: 
- build the full knn-graph of the dataset by completing the knn-graph on background images
- symmetrize and normalize the graph
- perform diffusion steps

The resulting performance (top-5 accuracy: 0.678) is averaged over 5 runs with seed=1..5 to produce the F1M resutls in table 3 (66.8%).

### Running all diffusion experiments

The whole set of operations, including hyper-parameter selection is performed by a shell script that has 4 phases: 
```
bash run_diffusion.bash 1
bash run_diffusion.bash 2
bash run_diffusion.bash 3
bash run_diffusion.bash 4
```

Since these operations are quite costly, it is worthwhile to parallelize the script to submit jobs on a cluster instead of doing the loops explicitly.

The final output looks like [this GIST](https://gist.github.com/mdouze/89e30746d37f98224610df5a21c84ecc)
