import torch
from pandas import DataFrame
from train import mapk

def train(model, device, train_loader, optimizer, criterion):
    """
    The function to train the model 

    params
    ======
    model: pytorch model to train
    device: device to load the dataset
    train_loader: the dataloader batching training dataset
    optimizer: optimizer for training
    criterion: criterion to measure the loss for training

    return
    ======
    train_loss(float)
    train_score(float)
    """

    # set the model in training mode
    model.train()
    
    train_score, train_loss = 0, 0

    for X_batch, y_batch in train_loader:

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # calculate the logits
        output = model(X_batch)
        # calculate the loss
        loss = criterion(output, y_batch)        
        # initialise the optimizer
        optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the parameters
        optimizer.step()

        # metric on the current batch
        train_score += mapk(y_batch, output, 3)
        train_loss += loss.item()
        
    
    train_loss /= len(train_loader.dataset)
    train_score /= len(train_loader.dataset)
    
    return train_loss, train_score





def evaluate(model, device, loader, criterion):
    """
    The function to evaluate the model 

    params
    ======
    model: pytorch model to train
    device: device to load the dataset
    loader: the dataloader batching validation/test dataset
    criterion: criterion to measure the loss for training

    return
    ======
    eval_loss(float)
    eval_score(float)
    """

    # set the model to evaluation mode
    model.eval()
    
    eval_score, eval_loss = 0, 0

    for X_batch, y_batch in loader:

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # calculate the logits
        output = model(X_batch)
        # calculate the loss
        loss = criterion(output, y_batch)
        # metric on the current batch     
        eval_score += mapk(y_batch, output, 3)
        # save the loss for the batch
        eval_loss += loss.item()
            
    eval_score /= len(loader.dataset)
    eval_loss /= len(loader.dataset)

    return eval_loss, eval_score
    


def fit(model, train_loader, valid_loader, optimizer, criterion, epochs=10):
    """
    The function to train the test dataset and evaluate the validation set

    params
    ======
    model: pytorch model to train
    train_loader: the dataloader batching train dataset
    valid_loader: the dataloader batching validation dataset
    optimizer: optimizer for training
    criterion: criterion to measure the loss for training
    epochs(int): the number of epochs to train

    return
    ======
    result(Dataframe): the result for training(loss and score for train/validation dataset)
    best_model_param: the best model parameters
    """

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_score = -1

    for epoch in range(epochs):
        train_loss, train_mapk = train(model, device, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}| train_loss: {train_loss}, train_mapk: {train_mapk}")

        val_loss, val_mapk = evaluate(model, device, valid_loader, criterion)
        print(f"Epoch {epoch+1}| val_loss: {val_loss}, val_mapk: {val_mapk}")

        # if validation score gets higher, save the model params
        if val_mapk > best_score:
            best_score = val_mapk
            print(f"model saves at {val_mapk}")
            torch.save(model.state_dict(), "best_model")

        result = {
            "Epoch": epoch+1,
            "train_loss": train_loss,
            "train_mapk": train_mapk,
            "val_loss": val_loss, 
            "val_mapk": val_mapk
        }
        
        results.append(result)


    return DataFrame(results), torch.load("best_model")
      