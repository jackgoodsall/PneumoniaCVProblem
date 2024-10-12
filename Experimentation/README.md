All trained on RTX 2070 Super - Google colabs free TPU was only slighty faster but was too much of a farse and can train
models on days where I don't need my PC like this.

All models state dicts will be saved in ../Models/

# Inital model (v1) - FC model
in ChestXrayInitalInvestigation

Image transforms - resize to 56 * 56
Model - Input 3 * 56 * 56 -> DENSE(512) -> Relu -> Output
Loss function - cross entropy
Optimiser - Adam
batch size - 64

Accuracy - 76%
F1 - 0.80

# Model (v2) - Same as above

Added image cropping - stops ratio issues when resizing.

Accuracy - 79%
F1 - 0.82#

Slight improvement

# Model (V3) - Same FFWN model 

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

# Model (V4) - Same as above with longer training - 10 epoches
