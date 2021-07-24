## My colab python notebook experiment on Fashion MNIST database using Convolutional Neural Network

### Key points learned from this experiment:

- Earlystopping helps the model from overfitting
- Saving the best model using Checkpoints during different epochs is a good strategy. You can load it later.
- Dropouts also help model against overfitting
- He_Normal initialization helps in convergence of error

### Things to do:

- Data augmentation will help in the cases where you have specifically less data or unbalanced data. The experiments were done on original FMNIST data with 10 labels. The modified 3 class experiment from FMNIST will need data augmentation for sure.
- L1/L2 Regulaizer testing

![](https://raw.githubusercontent.com/dexterdev/Machine_Learning_Experiments/master/CNN_fashion_MNIST/loss_acc.png)
