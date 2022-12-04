# Knowledge Distillation

Knowledge Distillation is a practice in compressing the model to be run on more efficent devices. Often Models are compromised of millions of different weights, samples, losses, etc. and most devices are not equpped to run these types of programs. This program required premium GPUS on Google Colab to run in a timley manner. The goal of Knowledge Distillation is to transfer what the fully trained larger model knows to a smaller model with less weights, samples, etc. The smaller model cannot be used from the start as it does not have a large knowlege capacity. The larger model is used first to determine the correct weights, learning rate, samples that the smaller model is not capable of. Once the larger model, the Parent Model, has completed running it is then 'distilled into a smaller student model by using labels given from the teacher model. 

![image](https://miro.medium.com/max/640/1*DdClMPqhErordaun8Dw14Q.webp)

First, create and train a Teacher model on the full dataset, with all the parameters needed to obtain the correct amount of knowledge in the network. 
  
Then distill the knowledge of the network to the student model, first establish the correspondances between the layers in the student and teacher networks. Often this correspondance can be directly passing the output of the teacher layer to the layer in the student network.  This knowledge is transferred from the teacher to student by minimizing the loss function. Since most of the proabilities are likely going to be close to zero the softmax that is produced by the function has a temperature. When the temperature is equal to 1 the softmax function is standard but as the temperature grows the proability distrobutions changes. Obtain the distillation loss by taking the loss from the students and comparing it to the loss from the teachers. 


![image](https://intellabs.github.io/distiller/imgs/knowledge_distillation.png)

By distilling the knowledge from the teacher to the student model, the student is able to learn much faster, and much more information for having a smaller amount of layers. Without knowledge distillation the smaller models would be innacurate, take longer to train, and be unable to run on different and less efficient devices. 

# Model Compression / Knowledge Distillation

## Framework Limitations

There were a number of framework limitations due to NNI. The main one is that there is currently no existing way to do model compression with models created from the Tensorflow framework. [This discussion](https://github.com/microsoft/nni/issues/4350) shows this to be the case as of November 2021, and it remains true to this day. There is very limited support for generation of masks for pruning a pre-trained model, but no way to actually apply the masks and begin the fine-tuning process. Further evidence of this is that even though Tensorflow code examples exist in [NNI - Model Compression](https://nni.readthedocs.io/en/v1.9/Compression/QuickStart.html), it is deprecated. Though we were able to get it working, there is and never was a [ModelSpeedup](https://nni.readthedocs.io/en/v1.9/Compression/ModelSpeedup.html) method, the crucial application of the prune masks, for Tensorflow in NNI

Because of these limitations in NNI, and instead of giving up there, we decided to try to implement an article we found for Model Compression using Tensorflow Keras. An overview can be found [here](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras), and was used in part in our code. It creates masks, generates a pruned model, and re-trains (fine-tunes) the new pruned model on the training data. Unfortunately, no inference cost improvements were realized. However, there was a substaintial decrease of model size due to pruning, as can be seen in the table below. GZipped Size is the one that matters in this instance.

| Model | Size on Disk | <ins>GZipped Size</ins> |
| --- | --- | --- |
| HPO | 23,523 KB | 21,336 KB |
| Pruned | 31,368 | 17,513 KB |

As you can see, the Pruned Model, when GZipped, takes up <ins>18%</ins> less storage space when compressed. This could be improved even more with an optimized file format, `.tflite`, for example.

## 10 Segmented Images from the Validation Set

There is not much difference between the HPO optimized version and the compressed version. This indicates that the compressed version maintains the knowledge learned in the dense model.

![1](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/1.png)
![2](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/2.png)
![3](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/3.png)
![4](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/4.png)
![5](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/5.png)
![6](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/6.png)
![7](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/7.png)
![8](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/8.png)
![9](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/9.png)
![10](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/10.png)

## Validation Loss vs. Epochs

As can be seen, the validation loss still decrease for the first couple of dozen epochs then plateaus. The model achieved a similar validation loss to the prior HPO model. We believe this to be due to the high number of epochs we tried this time.

![graph](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/loss.png)

## Precision and Recall Curve

Below are the new best model's precision and recall values (for validation set) as computed by sklearn's precision_recall_curve function.

![prcurve](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-4/docs/images/prcurve.png)

## Precision and Recall Values

Below are the final numerical precision and recall values (for validation sets) as computed by keras metrics.

```
Final Validation Precision: 0.9308
Final Validation Recall:    0.9244
```
