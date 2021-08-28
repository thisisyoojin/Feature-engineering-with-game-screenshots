from datasets import CustomDataset
from torch.utils.data import DataLoader

def get_data(root_dir, transform, batch_size=16):
    """
    Creates Dataloaders by handing over the custom dataset created.
    
    =============================================================================
    Params
    - root_dir: root directory
    - transform: transform functions which will be applied on image data

    =============================================================================
    Return
    - It returns tuples for train_loader, valid_loader, test_loader
    
    """
    train_dataset = CustomDataset(root_dir, transform, data_type='train', split=True)
    valid_dataset = CustomDataset(root_dir, transform, data_type='validation', split=False)
    test_dataset = CustomDataset(root_dir, transform, data_type='test', split=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader