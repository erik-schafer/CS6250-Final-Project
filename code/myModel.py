import torchvision
import torch.nn as nn


'''
Wrapper class for densenet
    Dense net has an output of 1000 classes, which are mapped to the desired
    number of output classes using a linear layer with sigmoid activation
'''
class DenseNet121(nn.Module):
    # init takes the desired output size
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # initialize models
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        # get the input features
        num_ftrs = self.densenet121.classifier.in_features
        # map num of input features to output size using linear sigmoid layer
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    # trivial forward function
    def forward(self, x):
        x = self.densenet121(x)
        return x