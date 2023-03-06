from torch.nn import Module
from torch import save
from pathlib import Path

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
               folder_path:str,
               model_name:str):
    '''Saves PyTorch model into specific folder
    
    Args:
    model: PyTorch Module model.
    folder_path: Path to a folder to save the model into.
    model_name: Name of the model with ".pt" or ".pth file extension".
    '''

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Model name must end with '.pth' or '.pt'."
    model_path = folder_path / model_name

    print(f"Saving {model_path}.")

    save(obj=model.state_dict(), f=model_path)

