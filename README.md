# NNDL_Kaggle
This is the project for NNDL class competition assignment

## Objective:
The objective of this project is to train images in the given dataset and predict superclass and subclass of an image.
One of the key aspects in this project is that, while testing the model, the model needs to predict unseen subclasses from the training data.

## Dataset:
The training dataset in the competition consists of 6.4K images of resolution 8x8x3 and spread over
3 super classes and 89 sub classes. Similarly, the test dataset contains about 9.1K images belonging
to the same 3 super classes, but some of them belong to sub classes not present in the training dataset.
Also, there is an imbalance in the number of samples for each sub class(50 or 100) in the training
dataset.

## Hierarchy
This directory consists of following items, <br>
1)Utils folder: Code for submitting predictions <br>
2)Approach_1_Transformer+Ensemble.ipynb which contains code for implementation of Approach 1 discussed below <br>
3)Approach_2_First.ipynb and Approach_2_MLP.ipynb which contains code for implementation of Approach 2 discussed below <br>

## Approach:
For this task, we have used two approaches <br>
1)Approach 1: An ensemble of CNN and Transformer <br>
2)Approach 2: A simple feed forward network with flattened images as input <br>

### Approach 1:
Steps followed <br>
1) Downloading the data <br>
2) upscaling the images to (64,64) <br>
3) Preparing the training and validation sets <br>
4) Preparing a baseline model <br>
5) Training on super class <br>
6) Training on sub class <br>
7) Training on both classes simulataneously using a common encoder <br>
8) Training Swin T on Augemented dataset <br>
9) Training Effnet B3 with and without using dropout <br>
10) Creating an ensemble of the 3 models [Swin T, EffnetB3, EffnetB4] <br>
11) Performance comparison through tensorboard logging <br>
12) Predicting on test data with threshold for sub classes <br>

Architecture:
![arc](https://github.com/ug2146/NNDL_Kaggle/blob/main/ensemble_architecture.png)

### Approach 2:
Steps followed <br>
1) Downloading the data <br>
2) Flatenning an 8x8x3 image to 192 length array.
3) Preparing the training and validation sets <br>
4) Preparing a baseline model of a feed forward network<br>
5) Training on sub class with combinations of hyper parameters.<br>
6) Comparative analysis <br>
7) Predicting on test data.<br>

Architecture of best MLP we got:
![MLP ARC.png](https://github.com/ug2146/NNDL_Kaggle/blob/main/mlp_architecture.png)






