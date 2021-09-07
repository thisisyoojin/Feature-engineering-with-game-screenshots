import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd


class Screenshots2D(Dataset):
    """
    Dataset with game screenshots, which will be passed to Dataloader for iterating data for training an evluation.

    params
    ======
    transforms(dict): dictionary with list of transforms for train and eval. It should be the list of transforms, not composed one.
    train(bool): when it sets True, it uses transform functions for training, otherwise it uses ones for evaluation(default: True).
    data_dir(str): folder directory where images are stored(default: ./steam-data)

    """
    def __init__(self, transforms, train=True, data_dir="./steam-data"):
        
        self.train = train
        self.data_dir = data_dir
        self.transforms = transforms
        self.images, self.targets = self.read_data()        


    def read_data(self):
        """
        The function to read csv file for target, and convert images to tensor

        returns
        ======
        images(3D torch): normalised tensor
        targets(2D torch): multilabel targets
        """
        # read target files created by preprocess module
        raw_data = pd.read_csv(f'{self.data_dir}/multilabel.csv', error_bad_lines=False)

        # get the transform functions for the purpose
        if self.train:
            transform = T.Compose(self.transforms['train'])
        else:
            transform = T.Compose(self.transforms['eval'])
        
        images = []
        targets = []

        # images
        for filepaths, label in zip(raw_data['filepaths'], raw_data['label']):

            label = torch.tensor([int(label) for label in label.split(',')])
            
            for filepath in filepaths.split(','):
                # open image file and convert to RGB (to match shape), and convert to float tensor
                float_img_tensor = transform(Image.open(f"{self.data_dir}/{filepath}").convert('RGB')).float()
                images.append(float_img_tensor)
                targets.append(label)
        
        return torch.stack(images), torch.stack(targets).float()


    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


    def __len__(self):
        return len(self.images)


