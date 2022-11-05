# Explanation of UNET
The approach that was used in the model was Semantic Segmentation with U-Net. U-Net is a type of classification algorithm that was created to help identify medical imagery. It is based on a convolutional neural network or CNN which is a type of network that specializes in grid type classification, such as images. There are a few steps in the application of U-net, first is obtaining the images, and if possible masks for the items in the images. Secondly, you need to make sure the images are all the same size, to do this you down sample the image, however you do not crop the images, since there will be a loss of critical information. You can take the approach that was used in this application, where you “Patch” the objects into smaller pieces of the same size. This is also called ‘Encoding’. When you have more pieces the network can get a better understanding of how everything links together. Once it has the patches, it corrects the color format from hex to RGB, and takes the first channel so it can classify. It then makes adjustments, and samples the images back to as close to their real size as possible, though there may be some loss. This is also called ‘Decoding’. Once the images are back to nearly their original size, the algorithm then uses a cross-entropy loss function to determine how close it is to the provided mask and corrects the corresponding weights accordingly. 


# Baseline Performance Results

## 10 Segmented Images from the Validation Set

You can see that the model performs quite well. Below are highlighted some special parts. The leftmost image in a row corresponds to the real-world sattellite imagery (patchified), the center image corresponds to the human-generated segmentation mask, and the rightmost image corresponds to the model's predicted mask.

![1](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/1.png)

You can see in the image above our model began to detect two roads that the human-generated masks did not.

![2](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/2.png)
![3](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/3.png)
![4](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/4.png)

In the image above, there is incorrect labelling of water as greenery.

![5](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/5.png)

In the image above, the model actually correctly labels a golf course as greenery whereas the human-generated mask labeled it as water.

![6](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/6.png)
![7](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/7.png)

The image above shows an example of the granularity that the model provides as opposed to the given mask.

![8](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/8.png)
![9](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/9.png)
![10](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/10.png)

## Training and Validation Loss vs. Epochs

As can be seen, the training and validation loss decrease for the first couple of dozen epochs then plateaus.

![graph](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/tv_loss_over_epochs.png)

## Precision and Recall Values

Below are the final precision and recall values (for training and validation sets) as computed by keras metrics.

![pr](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/pr.png)
