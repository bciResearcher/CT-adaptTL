# CT-adaptTL
This is the PyTorch implementation of the **Explainable Cross-Task Adaptive Transfer Learning for Motor Imagery EEG Classification.** 
# Flowchart of pre-training, fine-tuning and explainability analysis
![](https://github.com/yzmmmzjhu/CT-adaptTL/blob/main/CT-adaptTL.jpg)
The aim of this work is to explore the feasibility and interpretability of cross-task knowledge transfer between MI and ME. This can largely relax the constraint of training samples for MI BCIs and thus has important practical sense.
# Resources
## Datasets
HGD: [Link](https://gin.g-node.org/robintibor/high-gamma-dataset)

openBMI: [Link](http://dx.doi.org/10.5524/100542)

GIST:[Link](http://dx.doi.org/10.5524/100295)

## Pre-trained models
## Fine-tuned models

# Instructions
## Install the dependencies
It is recommended to create a virtual environment with python version 3.6 and running the following:

    pip install -r requirements.txt

You can try different mirror sources if you encounter version problems.

## Obtain the raw dataset
Download the raw dataset from the [resources](https://github.com/yzmmmzjhu/CT-adaptTL/blob/main/README.md#datasets) above, and save them to the same folder. Please download the ME/MI data in mat file format.

## Data alignment
For HGD run：

For openBMI run：

## Model pre-training
To pre-train the model with HGD, run:

This process is likely to take some time. We have provided the pre-trained models used in paper in above resources.

## Adaptive fine-tuning
To fine-tune the pre-trained model with MI dataset in scheme1 , run:

This process is likely to take some time. We have provided the fine-tuned models for each subject in openBMI dataset used in paper in above resources .

## Model explaining
To explain the fine-tuned models with MI test data in scheme1, run:

## Results
The classification results for our method and other competing methods are as follows:






