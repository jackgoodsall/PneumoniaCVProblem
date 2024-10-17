# Importing again just to write to file
import torch.nn as nn
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, accuracy_score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import *
import matplotlib.pyplot as plt



def train_step(
               model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device = DEVICE
               ) -> Tuple[float, float]:
  """
  Trains a PyTorch model for a single epoch.
  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader object.
    loss_fn: A PyTorch loss function.
    optimizer: A PyTorch optimizer.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)
      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def test_step(
              model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))


  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc


def train_model(
        train_data_loader : DataLoader,
        test_data_loader : DataLoader,
        model : nn.Module,
        loss_funcion : nn.Module ,
        epoches : int = 10,
        optim_func : torch.optim = torch.optim.Adam,
        learn_rate : float = 0.001,
        device = DEVICE,
        plot_loss_rates : bool = True,
        schedular = False,
        schdular_paras = {}
        ) -> torch.nn.Module:
    '''
    Function for fitting a model to a data
    '''
    print(schdular_paras)
    optimiser = optim_func(model.parameters(), learn_rate)
    if schedular:
       schedularr = schedular(optimizer = optimiser, **schdular_paras)

    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
   }

    train_losses = []
    test_losses = []
    for epoch in range(epoches):
        # Put model in train mode
        train_loss, train_acc = train_step(
           model,
           train_data_loader,
           loss_funcion,
           optimiser,
           device
          )
        
        test_loss, test_acc = test_step(
           model,
           test_data_loader,
           loss_funcion,
           device
            ) 
        if schedular:
          schedularr.step()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
          )
    if plot_loss_rates:
      plt.figure(figsize=(8,6))
      plt.plot(range(epoches), train_losses)
      plt.plot(range(epoches), test_losses)
      plt.show()


    # Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
   

def evalutation_model(
      model : nn.Module,
      evaluation_data : DataLoader,
      loss_fn : nn,
      device = DEVICE,
      ):
    '''
    Function for model evalutation, should of made it work with test_step but doesn't matter too much.
    Args:

    Returns:
    '''
    
    true_labels = []
    predicted_labels = []
    total_loss = 0
    correct = 0
    total = 0
    model.eval().to(device)

    with torch.no_grad():
        for inputs, labels in evaluation_data:

            inputs, labels = inputs.to(device), labels.to(device)          

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item() 

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    test_loss = total_loss / len(evaluation_data)
    test_accuracy = correct / total
    print(f'The test f1 score of this model is {f1_score(true_labels, predicted_labels)}')
    print(f'The test accuracy of this model is {test_accuracy}')

    return test_loss, test_accuracy, true_labels, predicted_labels
    
