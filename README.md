# CS634_final_project

Approach:
Firstly a convolutional neural network is used to segment the image, using the bounding boxes directly as a mask.
Secondly connected components is used to separate multiple areas of predicted pneumonia.
Finally a bounding box is simply drawn around every connected component.

Network:
The network consists of a number of residual blocks with convolutions and downsampling blocks with max pooling.
At the end of the network a single upsampling layer converts the output to the same shape as the input.
As the input to the network is 256 by 256 (instead of the original 1024 by 1024) and the network downsamples a number of times without any meaningful upsampling (the final upsampling is just to match in 256 by 256 mask) the final prediction is very crude. If the network downsamples 4 times the final bounding boxes can only change with at least 16 pixels.

We also make some change to the origin images.
1.Change input image size to 320x320.
2.Added transpose convolutions to allow greater flexibility in final predictions as per note above.
3.Usu cross-validation using StratifiedKFold or KFold
