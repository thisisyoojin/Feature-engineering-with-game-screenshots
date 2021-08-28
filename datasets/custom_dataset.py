
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from datasets import split_data


class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform, data_type='train', split=False):
        
        if split:
            split_data(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.data = self.get_data(data_type)
        

    def get_data(self, data_type):
        return ImageFolder(root=f"{self.root_dir}/{data_type}", transform=self.transform)


    def __getitem__(self, index):
        return self.data[index]


    def __len__(self):
        return len(self.data)

