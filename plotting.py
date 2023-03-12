import matplotlib.pyplot as plt
from typing import Dict

def plot_loss_curves(results: Dict):
    """Plots training results: loss and accuracy for test and validation sets.
    
    Args:
        results (Dict): a dictionary with lists of values, e.g.
            {"train_loss": [1.04, 0.82, 0.34],
             "val_loss": [1.14, 0.92, 0.54],
             "train_acc": [0.38, 0.59, 0.83],
             "val_acc": [0.33, 0.49, 0.75]}
    """ 
    
    epochs = range(len(results["train_loss"]))

    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    axs[0].plot(epochs, results["train_loss"], label="Training loss")
    axs[0].plot(epochs, results["val_loss"], label="Validation loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].plot(epochs, results["train_acc"], label="Training accuracy")
    axs[1].plot(epochs, results["val_acc"], label="Validation accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    fig.suptitle("Training results")

    plt.show()


results = {
    "train_loss": [1.04, 0.82, 0.34],
    "val_loss": [1.14, 0.92, 0.54],
    "train_acc": [0.38, 0.59, 0.83],
    "val_acc": [0.33, 0.49, 0.75]
}

plot_loss_curves(results)