All trained on RTX 2070 Super - Google colabs free TPU was only slighty faster but was too much of a farse and can train
models on days where I don't need my PC like this.

All models state dicts will be saved in ../Models/


Since all models only trained once there may be large variance in these numbers - will repeat for statistical signifance on later models (once I started with transfer 
learning) but for early models i'm not too worried.

# Inital model (v1) - FC model
in InitialInvestigation

Image transforms - resize to 56 * 56
Model - Input 3 * 56 * 56 -> DENSE(512) -> Relu -> Output
Loss function - cross entropy
Optimiser - Adam
batch size - 64

* Accuracy - 82%
* F1 - 0.86

# Model (v2) - Same as above
Iin nitialInvestigation

Added image cropping - stops ratio issues when resizing.

* Accuracy - 80%
* F1 - 0.86

Slight improvement

# Model (V3) - Same FFWN model 
in InitialInvestigation


Added data augmentation (instead of croping)

This did the following
* random rotation
* random flip
* random crop 
* colour jitter
* random affine
* random perspecive
* resize
* normalise


* Accuracy -  82%
* F1 - 0.86

# Model (V4) - Same as above with longer training - 10 epoches
in InitialInvestigation

* Accuracy - 85%
* F1 0.87