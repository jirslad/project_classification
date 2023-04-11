''' Functions for training, validation and testing '''
import torch
from time import time
from pathlib import Path
from .utils import save_model

def train_step(model, dataloader, loss_fn, optim, device, accuracy_fn):
    
    train_loss, train_acc = 0, 0

    model.train()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        y_class_idxs = torch.argmax(y_logits, dim=1)
        train_acc += accuracy_fn(y, y_class_idxs)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def val_step(model, dataloader, loss_fn, device, accuracy_fn):
    """ Step for validation or testing. """
    val_loss, val_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()
            y_class_idxs = torch.argmax(y_logits, dim=1)
            val_acc += accuracy_fn(y, y_class_idxs)
    
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)

    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          accuracy_fn,
          lr_scheduler: torch.optim.lr_scheduler._LRScheduler=None,     
          writer: torch.utils.tensorboard.SummaryWriter=None,
          checkpoint_saving: bool=False,
          model_path: str=None):
    """ Training procedure. 
    
    Performs training of a PyTorch model with a training DataLoader using
    specified loss function and optimizer, all on passed device. Runs for
    given number of epochs. Validates training on a validation DataLoader.
    Records ongoing losses and accuracies.
    
    Args:
        model (torch.nn.Module): Model to train.
        train_dataloader (torch.utils.data.DataLoader): Training dataloader.
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        device (torch.device): Device (e.g. "cuda", "", "cpu")
        accuracy_fn (function): Accuracy function for classification.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.   
        writer (torch.utils.tensorboard.SummaryWriter): Tensorboard writer.
        checkpoint_saving (bool): Flag to save model after every epoch.
        model_path (str): Path to a model to continue training on.

    Returns:
        results (Dict): a dictionary with lists of values, e.g.
            {"train_loss": [1.04, 0.82, 0.34],
             "val_loss": [1.14, 0.92, 0.54],
             "train_acc": [0.38, 0.59, 0.83],
             "val_acc": [0.33, 0.49, 0.75]}
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    # for tensorboard
    imgs = next(iter(train_dataloader))
    dummy_tensor = torch.randn(imgs[0].shape)
    
    time_start = time()
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn,
            optimizer, device, accuracy_fn)
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, device, accuracy_fn)
        lr = optimizer.param_groups[0]["lr"]
        if lr_scheduler:
            lr_scheduler.step()

        time_s = time() - time_start
        if epoch == 0:
            print(f"Epoch | Train loss | Val loss | Train acc | Val acc | Learn Rate | Time [s]")
        print(f"{epoch:^5} |   {train_loss:.4f}   |  {val_loss:.4f}  |   {train_acc:.3f}   |  {val_acc:.3f}  | {lr:.8f} |   {time_s:.0f}")

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)

        # TensorBoard experiment tracking
        if writer:
            writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict={"train": train_loss,
                                                "val": val_loss},
                            global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={"train": train_acc,
                                                "val": val_acc},
                            global_step=epoch)
            # writer.add_graph(model=model,
            #                 input_to_model=dummy_tensor.to(device))
            writer.close()
        
        # Save checkpoint
        if checkpoint_saving:
            try:
                class_names = train_dataloader.dataset.classes
            except:
                class_names = train_dataloader.dataset.dataset.classes
            save_model(model=model,
                       class_names=class_names,
                       folder_path=Path(model_path).parent,
                       model_name=Path(model_path).name)

    return results

