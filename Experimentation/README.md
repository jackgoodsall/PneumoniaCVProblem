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

* Accuracy - 84%
* F1Score - 88%

Model didnt seem to do that well, might have been a little bit too complex or maybe just the effect of having 10X the filters then channels. ( Effectively 2x the input space after pooling)

## Model v8
16(3) -> 32(3) -> Dense

Returning to a simpler model

* Accuracy - 89%
* F1Score - 91%

## Model v9
Folllowing on the theme of the above 16(3) -> 32(3) -> 64(3) -> Dense

This effectively halfs the feature space each time after the pooling.

* Accuracy - 88%
* F1Score - 91%


## Model v10
We know introduce a weighted random sampler, this should help balance the data set giving it better performance on the under represented class.

* Accuracy -  89%
* F1Score - 89%

Notice a slight decrease in f1score but a slight increase in accuracy - showing better performance on the non pnuomnia class.

## Model 11 
Same as above with 30 epoches

* Accuracy - 90%
* F1Score - 92%

The sampler seems to have made the model take longer to train, but has definately improved the overall performance.

## Model 12 - Drop out
One way that can help improve model performance is by introducing drop out on the dense layers, this in effect makes it not a dense layer, effectively turning of a % or neurons every epoche.

* Accuracy - 
* F1Score -

# Transfer learning
One way to avoid the computationaly expensive task of training the CNN's is to use transfer learning, that is using pretrained, premade models as a base of our network.

# Model 13 - Baseline efficentnet

Baseline training model on un augmented and un balanced data set. Image size was 256x256
* F1SCORE - 86%
* Accuracy - 80%

# Model 14 - data augmentation + balancing
* F1SCORE - 90%
* Accuracy - 87%

# Model 15 - abov with 224 image size 
Both decreased. So will stick with 256x256 iamges.

# Model 16 - 14 with longer training 
Not much increase over the shorter training.
* F1SCCORE - 90%
* Accuracy - 87%

# Model 17 - LR schedular
Model tested a LR schedular to adapatively change the LR as it trained.

* f1score - 87%
* accuracy - 86%

Showed decrease in accuaracy and f1 score, maybe could try different schedulars in the future or early stopping.