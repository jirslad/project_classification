''' Functions for training, validation and testing '''
import torch
from tqdm.auto import tqdm


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
          accuracy_fn):
    """ Training procedure. 
    
    Performs training of a PyTorch model with a training DataLoader using
    specified loss function and optimizer, all on passed device. Runs for
    given number of epochs. Validates training on a validation DataLoader.
    Records ongoing losses and accuracies.
    
    Args:
      model:
      ...

    Returns:
      A dictionary with training results.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }


    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn,
            optimizer, device, accuracy_fn)
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, device, accuracy_fn)

        if epoch == 0:
            print(f"Epoch | Train loss | Val loss | Train acc | Val acc")
        print(f"{epoch:^5} |   {train_loss:.4f}   |  {val_loss:.4f}  |   {train_acc:.3f}   | {val_acc:.3f} ")

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)


    return results

