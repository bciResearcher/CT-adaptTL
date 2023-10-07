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
When HGD is the source domain and openBMI is the target domain,run
        
When HGD is the source domain and GIST is the target domain, run


## Model pre-training
When openBMI is the target domain, run 

When GIST is the target domain, run 

This process is likely to take some time. We have provided the pre-trained models used in paper in above [resources](https://github.com/yzmmmzjhu/CT-adaptTL#pre-trained-models).

## Adaptive fine-tuning
To fine-tune the pre-trained model with openBMI dataset, run:

To fine-tune the pre-trained model with GIST dataset, run:

This process is likely to take some time. We have provided the fine-tuned models for each subject used in paper in above [resources](https://github.com/yzmmmzjhu/CT-adaptTL#fine-tuned-models) .

## Model explaining
To explain the fine-tuned models with openBMI dateset, run:

To explain the fine-tuned models with GIST dateset, run:

# Results
The classification results for our method and other competing methods are as follows:
## openBMI
| Methodology  | Mean (SD) |  Median | Range (Max-Min)|
| :------------- | :---------- | ------------ | ------------ |
|Subject-Specific |73.48(16.16) |69.50 |53.00(100.00-47.00) |
|Subject-Independent |69.00(15.58) |64.50 |51.00(99.00-48.00) |
|Subject-Adaptive |76.59(15.93) |74.50 |51.00(100.00-49.00) |

## GIST
| Methodology  | Mean (SD) |  Median | Range (Max-Min)|
| :------------- | :---------- | ------------ | ------------ |
|Subject-Specific |61.48(12.84) |57.50 |55.00(99.00-44.00) |
|Subject-Independent |58.96(10.39) |56.50 |39.00(82.00-43.00) |
|Subject-Adaptive |69.77(12.54) |67.00 |51.00(100.00-49.00) |

# Cite:
If used, please cite:

Minmin Miao, Zhong Yang, Hong Zeng, Wenbin Zhang, Baoguo Xu, Wenjun Hu. "Explainable Cross-Task Adaptive Transfer Learning for Motor Imagery EEG Classification",submit to Journal of Neural Engineering, 2023. 

# Acknowledgment
We thank Mane ZHANG20211 et al for their wonderful works.

[](https://doi.org/10.1016/j.neunet.2020.12.013)

