# Explanation of UNET
The approach that was used in the model was Semantic Segmentation with U-Net. U-Net is a type of classification algorithm that was created to help identify medical imagery. It is based on a convolutional neural network or CNN which is a type of network that specializes in grid type classification, such as images.

![image](https://user-images.githubusercontent.com/7800697/200138975-fa6df178-2ad1-41f5-b630-7ffeff8dc119.png)

In the above image you can see the general outline of the algorithm as a whole. There are two major steps in how U-net functions. The first step is encoding, within the encoding step there are a few more functions that happen. First the images are put into the Input image tile where they are broken down into ‘Patches’. The U-net algorithm takes images that are the same size, and large images would be overly computationally intensive, they must be patched. In the model for this assignment we used patch sizes of 256. This allows for the most information to be retained as cropping the images would cause data loss that would impact how the masks are checked. The more patches that are out of an image the more the algorithm can learn about relationships. 

![image](https://user-images.githubusercontent.com/7800697/200139014-e29c20d4-9e50-4a25-ad47-5be350d5fd9a.png)

Here is a similar image laid out in a linear manner. 

The next major step in the process is Decoding, in this step the images are guessed upon using the specified weights, to produce a mask for that corresponding patch. Once all images are masked they are upsampled, or resized back to the original and checked against the provided masks using a cross entropy loss function to determine the correctness of the mask. If the image was to be cropped in the encoding section it would not resize back correctly causing the loss function to return incorrect values. 



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

## Precision and Recall Curve

Below are the final mode's precision and recall values (for validation set) as computed by sklearn's precision_recall_curve function.

![prcurve](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/pr_curve.png)

## Precision and Recall Values

Below are the final precision and recall values (for training and validation sets) as computed by keras metrics.

![pr](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-2/docs/images/pr.png)