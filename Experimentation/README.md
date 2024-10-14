# Quick introduciton
All trained on RTX 2070 Super - Google colabs free TPU was only slighty faster but was too much of a farse and can train
models on days where I don't need my PC like this.

Edit (14/10) - Was going to save all models to be able to load them but that was way too much memory - might save final model but probably keep to just documenting results.


Since all models only trained once there may be large variance in these numbers - will repeat for statistical signifance on later models (once I started with transfer 
learning) but for early models i'm not too worried.

The metrics we will be concerned with are accuarcy and f1 score. 
* Accuracy - measure of (TP + TN)/(TP + TN + FP + FN)
* F1 score - harmonic mean of precision and recall

## Inital model (v1) - FC model
in InitialInvestigation

Image transforms - resize to 56 * 56
Model - Input 3 * 56 * 56 -> DENSE(512) -> Relu -> Output
Loss function - cross entropy
Optimiser - Adam
batch size - 64

* Accuracy - 82%
* F1 - 86%

## Model (v2) - Same as above
Iin nitialInvestigation

Added image cropping - stops ratio issues when resizing.

* Accuracy - 80%
* F1 - 86%

Slight improvement

## Model (V3) - Same FFWN model 
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
* F1 - 86%

## Model (V4) - Same as above with longer training - 10 epoches
in InitialInvestigation

* Accuracy - 85%
* F1 - 87%

# Moving onto convolutional neural networks
All models in this section in DelvingintoComplexity notebook.


Convolutional neural networks allow the network to learn from the structure of the image, a bit like a humans perceptive field. CNNs are also easily able to use bigger image sizes so we will also be sizing our images at 256 x 256.

## Model v5
Our first network will be a simple one consisting of a convolutional layer of 16 filters of size 3x3x3 into a FC 512 -> 2 dense layer.

* Accuracy - 89%
* F1 - 92%

## Model v6
Additional convolutional layer of 32 filters of size 16x5x5.
* Accuracy - 87%
* F1 - 90% 

Didnt perform any better then model v5 and is more complex, next model will downsize the channel size each time instead.

## Model v7
32 filters size 3 -> 16 filters size 5 

## Model v8


## Model v9


## Model v10
