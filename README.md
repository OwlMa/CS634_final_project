# CS634_final_project

Approach
Firstly a convolutional neural network is used to segment the image, using the bounding boxes directly as a mask.
Secondly connected components is used to separate multiple areas of predicted pneumonia.
Finally a bounding box is simply drawn around every connected component.

Network
The network consists of a number of residual blocks with convolutions and downsampling blocks with max pooling.
At the end of the network a single upsampling layer converts the output to the same shape as the input.
As the input to the network is 256 by 256 (instead of the original 1024 by 1024) and the network downsamples a number of times without any meaningful upsampling (the final upsampling is just to match in 256 by 256 mask) the final prediction is very crude. If the network downsamples 4 times the final bounding boxes can only change with at least 16 pixels.

Change input image size to 320x320.
Added transpose convolutions to allow greater flexibility in final predictions as per note above.
Edit by Chirag :

Load pneumonia locations
Table contains [filename : pneumonia location] pairs per row.

If a filename contains multiple pneumonia, the table contains multiple rows with the same filename but different pneumonia locations.
If a filename contains no pneumonia it contains a single row with an empty pneumonia location.
The code below loads the table and transforms it into a dictionary.

The dictionary uses the filename as key and a list of pneumonia locations in that filename as value.
If a filename is not present in the dictionary it means that it contains no pneumonia.

Data generator
The dataset is too large to fit into memory, so we need to create a generator that loads data on the fly.

The generator takes in some filenames, batch_size and other parameters.

The generator outputs a random batch of numpy images and numpy masks.

Cross validation using StratifiedKFold or KFold
