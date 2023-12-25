import torch

from typing import Tuple, Dict, List
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device='cpu') -> Tuple[float, float]:

    """This function defines the training step for a model.
    
    Args:
        model (torch.nn.Module): The model that is to be trained.
        dataloader (torch.utils.data.DataLoader): The training dataloader
        loss_fn (torch.nn.Module): Loss function to evaluate the model
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): The device used to train the model. Default='cpu'

    Returns:
        A tuple consisting the train loss and accuracy for an epoch.

        Example Usage: train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
    """

    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch,(X,y) in enumerate(tqdm(dataloader, desc="Train")):
        X,y = X.to(device), y.to(device)
        X = X.unsqueeze(dim=1)
        X = X.type(torch.float32)
        # Optimizer zero grad
        optimizer.zero_grad()
        # Forward pass
        y_logits = model(X)

        # Calculate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # Back Propagation
        loss.backward()

        # Optimizer Step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_logits)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device='cpu') -> Tuple[float, float]:
    """This function defines the validation step for a model.
    
    Args:
        model (torch.nn.Module): The model that is to be validated.
        dataloader (torch.utils.data.DataLoader): The validation dataloader
        loss_fn (torch.nn.Module): Loss function to evaluate the model
        device (torch.device): The device used to train the model. Default='cpu'

    Returns:
        A tuple consisting the validation loss and accuracy for an epoch.

        Example Usage: val_loss, val_acc = val_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)
    """
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.inference_mode():
        for batch,(X,y) in enumerate(tqdm(dataloader, desc="Val")):
            X,y = X.to(device), y.to(device)
            X = X.unsqueeze(dim=1)
            X = X.type(torch.float32)
            
            # Forward pass
            y_logits = model(X)
    
            # Calculate loss
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            y_labels = y_logits.argmax(dim=1)
            val_acc += ((y_labels == y).sum().item()/len(y_labels))

    val_loss = val_loss/len(dataloader)
    val_acc = val_acc/len(dataloader)
    return val_loss, val_acc

def train(model: torch.nn.Module,
         train_dataloader: torch.utils.data.DataLoader,
         val_dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         epochs: int = 32,
         device: torch.device = 'cpu') -> Dict[str, List[float]]:
    """This function defines the training functionality of the model.
    
    Args:
        model (torch.nn.Module): The model that is to be trained.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader
        loss_fn (torch.nn.Module): Loss function to evaluate the model
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        epochs (int): Number of epochs. Default=32
        device (torch.device): The device used to train the model. Default='cpu'

    Returns:
        A tuple consisting the train loss and accuracy for an epoch.

        Example Usage: train_loss, train_acc, val_loss, val_acc = train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, loss_fn=loss_fn,
                                                                        optimizer=optimizer, epochs=32, device=device)
    """
    CHECKPOINT_PATH = 'checkpoint/'
    
    model_results = {'model_name': model.__class__.__name__,
                    'train_loss': [],
                    'train_acc': [],
                    'val_loss': [],
                    'val_acc': []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)
        val_loss, val_acc = val_step(model=model,
                           dataloader=val_dataloader,
                           loss_fn=loss_fn,
                           device=device)

        
        model_results['train_loss'].append(train_loss)
        model_results['train_acc'].append(train_acc)
        model_results['val_loss'].append(val_loss)
        model_results['val_acc'].append(val_acc)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.2f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%")

        if (epoch+1) % 5 == 0:
            filename = f'{model.__class__.__name__}_{epoch+1}.pth'
            save_path = CHECKPOINT_PATH + filename
            
            torch.save(obj=model.state_dict(), f=save_path)

    np.savez(file=f"results/{model.__class__.__name__}.npz", arr=model_results)
    return model_results