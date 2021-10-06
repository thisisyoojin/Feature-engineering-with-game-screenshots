import torch.nn as nn

class BaseModel(nn.Module):
    """
    Multilabel classification model for predicting genres(9 labels)
    - action, adventure, rpg, shooting, simulation, puzzle, casual, strategy, arcade
    """
    def __init__(self, num_output):
        
        super(BaseModel, self).__init__()

        self.num_output = num_output
        
        self.cnn_layers= nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 512, kernel_size=2, padding=1),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(41472, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_output),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.cnn_layers(X)
        X = X.view(X.size(0), -1)
        output = self.linear_layers(X)
        return output