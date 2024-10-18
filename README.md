# Overview
<span>
Repository for the chest xray pneumonia problem from kaggle. Problem can be seen [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).


Data won't be uploaded, can dowload from kaggle.

Will try to make the solution modular where possible(not main focus right now).

</span>

# The Problem

We are given a set of x-ray images from reprospective cohorts of pediatric patients of one to five year olds  from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

# The data

The training data was massively skewed towards pneuomnia cases, this massively inbalanced set leads us to need to look at classification metrics other then accuracy for our inital models and later to data augmentation - a way to artifically increase test data size.

# Model change log

Models changes and their effect of classifcation metrics will be log in md file in the Experimentation file.

# Final Model
Final model was a EfficientNetV2S, with data augmentation and a weighted sampler. It achieved an accuracy of 88%, an achievement of 8% over the baseline mode which has 10%.

# Potential future improvements
Further potential improvements to the model could involve things like moving to albumenations for data augmentation (I tried my pc didn't like it), adding early stopping, playing around with different learn rate scedular, and testing different pretrained models.


# What this project taught me
Altough there is definately room for further improvements on my final model, without the resources (computing units on google colab) most will take too much time to implement and run to effectively look at impact so at this time I don't think it teach me more then looking at other areas will.

Granted this I still learnt a few things doing this project

* 1. Modular functionality - throughout this project I a few times had to adapt or make new functions and in the future I think i will have a better base for developing these functions out from the start.
* 2.  Bigger doesnt mean better - Just because a model has more layers, more weights, more filters, more whatever doesn't make it inherently better.
* 3. Training models takes time :(