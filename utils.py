from torch.nn import Module
from torch import save, load
from pathlib import Path
from typing import List
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os


### ACCURACY FUNCTIONS ###

# TODO:
# implement Hamming Loss for multi-label accuracy meassure,
# or use "So given a single example where you predict classes A, G, E and the test case has E, A, H, P as the correct ones you end up with Accuracy = Intersection{(A,G,E), (E,A,H,P)} / Union{(A,G,E), (E,A,H,P)} = 2 / 5"
# or use TorchMetrics https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#multilabel-accuracy


def multiclass_accuracy(y, y_class_idxs):
    '''Returns accuracy in range [0, 1]'''
    return (y == y_class_idxs).sum().item() / len(y)

def multilabel_accuracy(y, y_logits):
    return 999.99

### MODEL SAVING ###

def save_model(model:Module,
               class_names: List,
               folder_path:str,
               model_name:str):
    '''Saves dictionary with PyTorch model state_dict and class names into specific folder.
    
    Args:
        model: PyTorch Module model.
        class_names: List of strings of class names.
        folder_path: Path to a folder to save the model into.
        model_name: Name of the model with ".pt" or ".pth file extension".
    '''

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Model name must end with '.pth' or '.pt'."
    model_path = folder_path / model_name

    print(f"Saving {model_path}.")

    save_dict = {
        "state_dict": model.state_dict(),
        "class_names": class_names
    }
    save(obj=save_dict, f=model_path)

def load_model(model:Module,
               model_path: str,
               device: str) -> Module:
    """Loads PyTorch Module model state_dict and assigns it to a model. Also loads class names.
    
    Args:
        model: Empty PyTorch Module model.
        model_path: Path to model parameters (".pt" or ".pth" file).
    
    Returns:
        model: PyTorch Module model with trained state_dict.
        class_names: List of class names.
        device: torch.cuda device ("cpu" or "cuda").
    """

    assert str(model_path).endswith(".pth") or str(model_path).endswith(".pt"), \
        "Model path must end with '.pth' or '.pt'."
    
    loaded_dict = load(model_path, map_location=device)
    model.load_state_dict(loaded_dict["state_dict"])
    class_names = loaded_dict["class_names"]

    return model, class_names


def create_writer(experiment_name: str, 
                  model_name: str,
                  num_epochs: str,
                  extra: str=None) -> SummaryWriter():
    """Creates a tensorboard SummaryWriter() saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        num_epochs (str): Number of trainig epochs.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, num_epochs, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, num_epochs, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")

    return SummaryWriter(log_dir=log_dir)

