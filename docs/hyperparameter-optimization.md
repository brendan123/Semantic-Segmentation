# Hyperparameter Optimization Performance Results (with HyperBand minimizing loss)

## 10 Segmented Images from the Validation Set

It is difficult to tell visually that the model performs even better than before. However, there are some slight differences such as greater granularity in difficult-to-mask images. We believe that at this point the model's performance is hampered almost only by the original masks which lack enough granularity to be trained upon further. Below, the leftmost image in a row corresponds to the real-world sattellite imagery (patchified), the center image corresponds to the human-generated segmentation mask, and the rightmost image corresponds to the model's predicted mask.

![1](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/1.png)
![2](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/2.png)
![3](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/3.png)
![4](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/4.png)
![5](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/5.png)
![6](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/6.png)
![7](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/7.png)
![8](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/8.png)
![9](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/9.png)
![0](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/0.png)

## Training and Validation Loss vs. Epochs

As can be seen, the training and validation loss still decrease for the first couple of dozen epochs then plateaus, but the best HPO model hits a lower loss for validation than the previous best model.

![graph](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/loss.png)

When superimposing the previous best model's (without HPO) loss curve, you can really see that the new one achieves a lower validation loss.

![edited_graph](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/loss_edit.png)

## Precision and Recall Curve

Below are the new best model's precision and recall values (for validation set) as computed by sklearn's precision_recall_curve function.

![prcurve](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/pr.png)

## Precision and Recall Values

Below are the final precision and recall values (for training and validation sets) as computed by keras metrics.

> Training Precision:	0.8633995652198792
> 
> Training Recall:	0.8405570983886719
> 
> Validation Precision:	0.8623082637786865
> 
> Validation Recall:	0.8403835296630859
