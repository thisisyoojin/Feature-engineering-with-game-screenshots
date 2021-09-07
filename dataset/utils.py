import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

def calculate_mean_std(dataset, dim=(1,2)):
    """
    This function is calculating mean and std from training set with unnormalised tensors

    params
    ======
    dataset(tensor): training set with unnormalised tensors
    dim(int or tuple): dimension to reduce (default: (1,2))

    return
    ======
    mean(float): mean of training set
    std(float): standard deviation of training set
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for x, _ in iter(dataset):
        channels_sum += torch.mean(x, dim=dim)
        channels_squared_sum += torch.mean(x**2, dim=dim)
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std


def split_data(dataset, random=False, train_ratio=0.8):
    """
    This function split the data with given ratio

    params
    ======
    dataset(tensors): the dataset to split.
    random(bool): if it sets True, it randomly split the data. default is False.
    train_ratio(float): the portion for spliting the data. default is 0.8.
    """
    train_size = int(len(dataset)*train_ratio)
    test_size = len(dataset) - train_size
    
    params = {
        "dataset": dataset,
        "lengths": [train_size, test_size]
    }

    if not random:
        params["generator"] = torch.Generator().manual_seed(42)

    train_data, test_data = random_split(**params)
    
    return train_data, test_data



def get_dataset(dataset, batch_size=64, random=False):
    """
    This function is split the data and pass the data to create data loaders.

    params
    ======
    dataset(tensor): all dataset before splitting
    batch_size(int): the number for batch. default is 64.
    random(bool): if the function split train and validation randomly or not. default is False.

    return
    ======
    train_loader(Dataloader): dataloader for training set
    valid_loader(Dataloader): dataloader for validation set
    test_loader(Dataloader): dataloader for testing set
    """
    train_data, test_data = split_data(dataset, random=False)
    train_data, valid_data = split_data(train_data, random=random)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader