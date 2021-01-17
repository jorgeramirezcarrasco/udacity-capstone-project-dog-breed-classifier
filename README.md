# Udacity Capstone Project Dog Breed Classifier

Udacity Capstone Project Dog Breed Classifier

## Project Overview

The problem of the proposal is to categorize dog and human images estimating the closer canine's breed. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed. From the initial dataset a subset of images will be stored for testing in order to measure the estimations' quality.

## Project Instructions

The code have been developed working with the notebook instances of AWS Sagemaker. Everything is designed to work with the jupyter lab environment pytorch_p36 and make usage of the SageMaker training and endpoint capabilities.

## Datasets

* [Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). It contains dog images of 133 breeds.
* [Human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). It contains celebrities' images for training a face detector.

## Directories and Files

* **haarcascades/:** folder with the required xml file for opencv to make the face detection.
* **images/:** test images for the notebook.
* **lambda/:** folder that contains the file with the lambda function code.
* **models/:** folder that contains the required pytorch code and requirements for make the model training with AWS SageMaker.
* **serve/:** folder that contains the required pytorch code for make the inference with AWS SageMaker.
* **web/:** folder that contains the webapp html & js code.
* **capstone_project_dog_breed_classifier.ipynb:** Notebook with all the code and steps of the project from downloading the datasets until the testing of the inference code.
* **capstone_project_dog_breed_classifier.html:** HTML version of the notebook.
* **model_transfer.pt:** Pytorch model trained from scratch
* **README.md:** Readme file.
