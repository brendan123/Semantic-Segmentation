# Explanation of UNET
INSERT EXPLANATION OF UNET HERE


# Results

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