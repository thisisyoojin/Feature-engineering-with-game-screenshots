
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from datasets import split_data


class CustomDataset(Dataset):
    """
    ************************************************************************************************************
    This class is a custom dataset for reading image files from root directory.
    It is inheritance from torch.utils.data.Dataset, and it will be handed to dataloader class.
    
    To get the dataset, the images should be inside of folders(train/validation/test).
    ************************************************************************************************************
    Init params
    
    - root_dir: root directory with image files
    - transform: 
    - data_type: train / validation / test
    - split: True when the folder needs to be created and images should be divided for train / validation / test
    
    """

    def __init__(self, root_dir, transform, data_type='train', split=False):
        # split the data based on the param
        if split:
            split_data(root_dir)

        # set the initial params
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.get_data(data_type)
        

    def get_data(self, data_type):
        # Get the dataset based on the data type
        return ImageFolder(root=f"{self.root_dir}/{data_type}", transform=self.transform)


    def __getitem__(self, index):
        return self.data[index]


    def __len__(self):
        return len(self.data)

