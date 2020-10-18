-   Having more than 1 photo of each person really helps
-   Would cutting every image in half (removing the bottom) help to do masked predictions?
-   If you do a training/testing split of the original test data, it's quite possible to have a situation where some people's images are almost entirely only in the validation set, so the model hasn't trained on them. Preparing the data is super important.

Accuracy when training on all faces with 0.0 validation split, testing on only masked: dunno because it crashed, but the accuracy said 0.6 in the last epoch

Accuracy when training on all faces cut in half, testing on only masked cut in half, making sure all people have some entries in the training and some in the test set:

accuracy in 5th epoch: 70+ %
accuracy in validation: 7%

overfitting? yes

when removing all images that don't have entries in both lists:

accuracy in 5th epoch: 71+ %
accuracy in validation: 6%

Idea: Identify which images tend to be guessed wrong, look at the images and figure out why, and optimize for that. May involve detecting facial features or perhaps cutting off a larger part of the image to remove the mask

Another idea: Make sure the amount of classes is consistent with the amount of classes in the training and validation data

Today

Try to reduce amount of classes to the exact amount
Reduced classes to 128 to save time

use bias false = 0.41
use bias true = 0.41

activation linear = 0.41
activation relu = 0.37
activation sigmoid = almost zero

with 2 more intermediate convolutional layers = 0.36
removing 1 added conv layer, changing first layer to 32 = 0.42
Same as above but changed val split to 0.3 = 0.36
Same as above but changed val split to 0.5 = 0.29
Same as above but changed val split to 0.1 = 0.43
Same as above but changed batch size to 50 = 0.45
Same as above but changed batch size to 20 = 0.45
Same as above but initial layer to linear = 0.42
Changed initial layer back, validation data actually used = 0.05

changed first to 16, reduced dropouts, added 1 layer = 0.38
same as above but first layer is 64 = 0.36
1 conv layer (64), 1 conv layer (128), then exit layers = not good

Added batch normalisation, removed dropouts, doubled conv layers before each pooling = 0.0063
Same as above but changed activations to elu and re-added dropouts = 0.0085
Same as above but changed activations to linear and used Adam = 0.006

Best is 0.45
RMSProp results in complete overfitting
