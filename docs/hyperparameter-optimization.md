# Explanation of HyperBand HPO

TODO: Explain in 2 pages including figures the method used.

# Hyperparameter Optimization Performance Results (with HyperBand minimizing loss)

## Search Space

We defined the search space for our HPO to be the following (below). This choice would let us explore the affects of different batch sizes on model performance, along with different learning rates. The most interesting optimization, in our opinion, was `activation_type`. The initial model was using `relu` and we were curious to see how other activations functions performed on the image segmentation task. This turned out to be a good decision as it turned out many of our best models used [Swish](https://en.wikipedia.org/wiki/Swish_function), a modified SiLU (sigmoid-weighted linear unit).

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

Below are the final numerical precision and recall values (for training and validation sets) as computed by keras metrics.

```
Training Precision:	0.8633995652198792
Training Recall:	0.8405570983886719
Validation Precision:	0.8623082637786865
Validation Recall:	0.8403835296630859
```
