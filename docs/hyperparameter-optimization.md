# Explanation of HyperBand HPO

HyperBand is a type of optimization of Hyperparameters in a learning algorithm, that expands off the fundementals of successive halving optimaztion. 

Hyperparameters in learning algorithms are varibales whos values control the learning process, they are the higest level of parameters available to the model. They are values such as: The number of epochs, the amount of branches in a decision tree, the minimum sample leaf, the learning rate etc. 

The choice in the hyperparameters has a direct relation to the performace on the model, which is where the opimization comes in. 

Since Hyperband is an expansion of Successive Halving Optimization we must first know what Successive Halving is. 

Picture diagram of Successive Halving.
![image](https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-05318-5_1/MediaObjects/453309_1_En_1_Fig3_HTML.png)  


Successive Halving is an optimization method that starts off by assigning a budget, and equally assigns the buget to all models that are running with different hyperparameters. 
At the end of the run, rate all the performances and drop the lower 50%, repeat until you only have one model left with the full budget, that would be your best performer. 

Hyperband works almost the same way. Take a look at the overview of the algorithm. 

![image](https://2020blogfor.github.io/images/blog_images/hyperband/pseudoCode.png)

Take the same Successive Halving algorithm, and randomly sample 64 hyperperameters and evaluate after 100 iterations, discard the lower performers, and rerun the higer performers until you have one model left. 

Hyperband allows for multiple configurations to be tested at one time per iteration, or several possible values for a fixed Budget.

Hyperband also allows you to checkout the model during learning, and stop it and start it. 

![Image](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/hyperband.png?resize=768%2C424&ssl=1)

# Hyperparameter Optimization Performance Results (with HyperBand minimizing loss)

## Search Space

We defined the search space for our HPO to be the following (below). This choice would let us explore the affects of different batch sizes on model performance, along with different learning rates. The most interesting optimization, in our opinion, was `activation_type`. The initial model was using `relu` and we were curious to see how other activations functions performed on the image segmentation task. This turned out to be a good decision because many of our best models used [Swish](https://en.wikipedia.org/wiki/Swish_function), a modified SiLU (sigmoid-weighted linear unit).

```python
search_space = {
    'batch_size': {'_type': 'randint', '_value': [1, 32]},
    'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'activation_type': {'_type': 'choice', '_value': ['relu', 'tanh', 'swish', None]}
}
```

We set the initial values to be the following (indicated by the initial seed repo):

```python
params = {
    'batch_size': 16,
    'learning_rate': 0.007,
    'activation_type': 'relu'
}
```

We set up the NNI experiment using the following values. This means we used HyperBand as the tuner, with the goal of minimizing loss. We ran a total of 72 trials, with a maximum of 4 at a tmie. The whole ordeal took about 2.5 hours and utilized about 30 Google Colab Premium Compute Units.

```python
experiment.config.tuner.name = 'HyperBand'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 72
experiment.config.trial_concurrency = 4
```

## Best Hyperparameters

After running the Hyperparameter Optimization with HyperBand, the NNI experiment converged on the following values for our best model:

```python
best_params = {
    'batch_size': 24,
    'learning_rate': 0.001990717520183526,
    'activation_type': 'swish'
}
```

As explained below, this produced a final validation loss of `0.9069907069206238`.

## 10 Segmented Images from the Validation Set

It is difficult to tell visually that the model performs even better than before. However, there are some slight differences such as greater granularity in difficult-to-mask images. We believe that at this point the model's performance is hampered almost only by the original masks which lack enough granularity to be trained upon further. Below, the leftmost image in a row corresponds to the real-world sattellite imagery (patchified), the center image corresponds to the human-generated segmentation mask, and the rightmost image corresponds to the model's predicted mask.

![compare](https://github.com/brendan123/Semantic-Segmentation/blob/milestone-3/docs/images/beforeafter.png)

The image above happens to be one that was generated for validation from the initial model before HPO. We have a unique opportunity here to show them side-by-side.

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

Below are the final numerical precision and recall values (for training and validation sets) as computed by keras metrics.

```
Training Precision:	0.8633995652198792
Training Recall:	0.8405570983886719
Validation Precision:	0.8623082637786865
Validation Recall:	0.8403835296630859
```
