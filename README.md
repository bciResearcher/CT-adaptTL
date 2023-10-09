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

## Sample pre-trained models

For openBMI: [Link](https://github.com/yzmmmzjhu/CT-adaptTL/tree/main/code_openBMI/pretrain/pretrain14_54)

For GIST: [Link](https://github.com/yzmmmzjhu/CT-adaptTL/tree/main/code_GIST/pretrain/pretrain14_52)

## Sample fine-tuned models

For openBMI: [Link](https://github.com/yzmmmzjhu/CT-adaptTL/tree/main/code_openBMI/adapt/model_54_)

For GIST: [Link](https://github.com/yzmmmzjhu/CT-adaptTL/tree/main/code_GIST/adapt/model_52_)

# Instructions
## Install the dependencies
It is recommended to create a virtual environment with python version 3.6 and running the following:

    pip install -r requirements.txt

You can try different download sources if you encounter version problems.

## Obtain the raw dataset
Download the raw dataset from the [resources](https://github.com/yzmmmzjhu/CT-adaptTL/blob/main/README.md#datasets) above(Please download the ME/MI data in mat file format), and save them to the same folder. 

        datasets/GIST/s01.mat
                     /s02.mat
                     /...

        datasets/openBMI/sess01_subj01_EEG_MI.mat
                        /sess01_subj02_EEG_MI.mat
                        /...
                        /sess02_subj01_EEG_MI.mat
                        /sess02_subj02_EEG_MI.mat
                        /...

        datasets/HGD/test
                        -/1.mat
                        -/2.mat
                        -/...
                    /train
                        -/1.mat
                        -/2.mat
                        -/...

## Data alignment
When HGD is the source domain and openBMI is the target domain,run

        PROCESS14_54.py
        
        process54.py
        
When HGD is the source domain and GIST is the target domain, run

        PROCESS14_52.py
        
        process52.py

## Model pre-training
When openBMI is the target domain, run 

        base14_54.py
        
When GIST is the target domain, run 

        base14.py

This process is likely to take some time. We have provided sample pre-trained models in above [resources](https://github.com/yzmmmzjhu/CT-adaptTL#pre-trained-models).

## Subject-specfic
When openBMI is the target domain, run 

        specific_54.py
        
When GIST is the target domain, run 

        specific_52.py

## Subject-independent
When openBMI is the target domain, run 

        subject_independent14_54.py     
        
When GIST is the target domain, run 

        subject_independent14_52.py


## Adaptive fine-tuning
To fine-tune the pre-trained model with openBMI dataset, run:

        subject_adaptive14_54_sin.py

To fine-tune the pre-trained model with GIST dataset, run:

        subject_adaptive14_54_sin.py

This process is likely to take some time. We have provided sample fine-tuned models for each subject in above [resources](https://github.com/yzmmmzjhu/CT-adaptTL#fine-tuned-models) .

## Model explaining
To explain the fine-tuned models with openBMI dateset, run:

        shap_value_adjust_54.py     
        
To explain the fine-tuned models with GIST dateset, run:

        shap_value_adjust_52.py
        
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
We thank Kaishuo Zhang et al for their wonderful works.

https://doi.org/10.1016/j.neunet.2020.12.013

