from torchvision import models
import torch.nn as nn

class CustomResnet50(nn.Module):
    """
    Transfer learnt resnet50 model for multilabel classification model for predicting genres(9 labels)
    - action, adventure, rpg, shooting, simulation, puzzle, casual, strategy, arcade
    """

    def __init__(self, num_output):
        
        super(CustomResnet50, self).__init__()

        resnet = models.resnet50(pretrained=True)

        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, num_output),
            nn.Sigmoid()
        )

        ct = 0

        for child in resnet.children():
            ct += 1
            if ct < 4:
                for param in child.parameters():
                    param.requires_grad = False

        self.resnet = resnet


    def forward(self, x):
        return self.resnet(x)