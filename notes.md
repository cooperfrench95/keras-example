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
