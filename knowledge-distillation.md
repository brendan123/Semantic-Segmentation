# Knowledge Distillation

Knowledge Distillation is a practice in compressing the model to be run on more efficent devices. Often Models are compromised of millions of different weights, samples, losses, etc. and most devices are not equpped to run these types of programs. This program required premium GPUS on Google Colab to run in a timley manner. The goal of Knowledge Distillation is to transfer what the fully trained larger model knows to a smaller model with less weights, samples, etc. The smaller model cannot be used from the start as it does not have a large knowlege capacity. The larger model is used first to determine the correct weights, learning rate, samples that the smaller model is not capable of. Once the larger model, the Parent Model, has completed running it is then 'distilled into a smaller student model by using labels given from the teacher model. 

![image](https://miro.medium.com/max/640/1*DdClMPqhErordaun8Dw14Q.webp)

First, create and train a Teacher model on the full dataset, with all the parameters needed to obtain the correct amount of knowledge in the network. 
  
Then distill the knowledge of the network to the student model, first establish the correspondances between the layers in the student and teacher networks. Often this correspondance can be directly passing the output of the teacher layer to the layer in the student network.  This knowledge is transferred from the teacher to student by minimizing the loss function. Since most of the proabilities are likely going to be close to zero the softmax that is produced by the function has a temperature. When the temperature is equal to 1 the softmax function is standard but as the temperature grows the proability distrobutions changes. Obtain the distillation loss by taking the loss from the students and comparing it to the loss from the teachers. 


![image](https://intellabs.github.io/distiller/imgs/knowledge_distillation.png)

By distilling the knowledge from the teacher to the student model, the student is able to learn much faster, and much more information for having a smaller amount of layers. Without knowledge distillation the smaller models would be innacurate, take longer to train, and be unable to run on different and less efficient devices. 
