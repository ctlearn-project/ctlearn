
import torch.nn.functional as F
import torch
from torchvision import models
import torch.nn as nn


class SingleChannelRegNet(nn.Module):
    def __init__(self, num_inputs=1, num_classes=2):
        super(SingleChannelRegNet, self).__init__()
        # self.regnet = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        self.regnet = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)

        # self.regnet = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.DEFAULT)
        # self.regnet = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.DEFAULT)
        # self.regnet = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)
        # Modify the first layer to accept the desired number of channels
        self.regnet.stem[0] = nn.Conv2d(num_inputs, 32, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Modify the Linear layer to change the number of outputs (classes) 
        num_features = self.regnet.fc.in_features
        self.regnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.regnet(x)
    
class DBBRegNet(nn.Module):
    def __init__(self, task, use_concat=False, num_inputs=1, num_classes=2,dropout_rate = 0.1):
        super(DBBRegNet,self).__init__()
        self.use_concat= use_concat
        self.task = task.lower()
        self.bb1 = SingleChannelRegNet(num_inputs=num_inputs, num_classes=num_classes)
        self.bb2 = SingleChannelRegNet(num_inputs=num_inputs, num_classes=num_classes)

        num_features = self.bb1.regnet.fc.in_features

        if self.use_concat:
           num_features*=2

        # Remove the final layer
        self.bb1.regnet.fc = nn.Identity()
        self.bb2.regnet.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate,inplace=True)    
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, y):

        energy = None
        classification = None
        direction = None

        feature_1 = self.bb1(x)
        feature_2 = self.bb2(y)

  
        # Combine outputs
        if self.use_concat:
            out = torch.cat((feature_1, feature_2), dim=1)
        else:
            out = feature_1 + feature_2
        
        out = self.dropout(out)
        out = self.fc(out)  
         
        if self.task == "type":
            classification = out
        elif self.task == "energy":
            energy = out
        elif self.task == "direction":
            direction = out


        return classification, energy, direction        