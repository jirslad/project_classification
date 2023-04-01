import matplotlib.pyplot as plt
from typing import Dict, List
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix as confmat_plot

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

def plot_predictions(model,
                     img_paths
                     ):
    """ Makes predictions on several images and plots them.
    
    Args:

    """

    pass


def plot_confusion_matrix(class_names: List,
                          pred_idxs: List,
                          target_idxs: List,
                          task: str="multiclass"):
    """ Plots a confusion matrix.
    
    Args:
        class_names (List): Class names as strings.
        pred_idxs (List): Predicted class indexes as ints.
        target_idxs (List): Target class indexes as ints.
        task (str): Task for confusion matrix (default "multiclass").
    """
    confmat = ConfusionMatrix(num_classes=len(class_names), task=task)
    confmat_tensor = confmat(preds=torch.Tensor(pred_idxs),
                            target=torch.Tensor(target_idxs))
    fig, ax = confmat_plot(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(8, 6)
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_dataset_distribution(dataloader):
    try:
        classes = dataloader.dataset.classes
    except:
        classes = dataloader.dataset.dataset.classes
    labels = torch.empty(0, dtype=torch.long)
    for _, targets in dataloader:
        labels = torch.cat((labels, targets))
    class_counts = torch.bincount(labels).numpy()
    bins = torch.arange(len(classes)+1).numpy()-0.5
    plt.hist(bins[:-1], bins, weights=class_counts)
    plt.title("Dataset class distribution.")
    plt.xlabel("Class index")
    plt.ylabel("Number of instances")
    plt.show()