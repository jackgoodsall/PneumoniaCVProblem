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

