import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from ctlearn.core.pytorch.nets.models.EffientNet_pytorch.model import EfficientNet
from ctlearn.core.pytorch.nets.block.cnn_blocks import Dirichlet

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            MemoryEfficientSwish()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleBBEfficientNet(nn.Module):
    def __init__(self,model_variant:str= "efficientnet-b3",task:str="Energy",num_outputs=2, device_str="cuda", energy_bins=None):
        super(DoubleBBEfficientNet, self).__init__()
        self.task = task.lower()
        self.num_outputs = num_outputs
        self.energy_bins = energy_bins
        self.device = torch.device(device_str)
        self.num_outputs = num_outputs
        hidden_size= 512
        if 'b3' in model_variant:
            feature_size= 1536*1 # vb3
        elif 'b5' in model_variant: 
            feature_size = 2048 # vb5
        else:
            raise ValueError(f"Model variant {model_variant} not tested. Adapt the feature_size.")
        
        if self.task=="type":
            self.backbone1 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs,use_batch_norm=True)
            self.backbone2 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs,use_batch_norm=True)
            # self.Dirichlet = Dirichlet(hidden_size,num_outputs)
        if self.task=="energy":
            self.backbone1 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs)
            self.backbone2 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs)
            #--------------------------------------------------------------------------------
            # Old code
            #--------------------------------------------------------------------------------
            # use_swish=True 
            # use_batch_norm = True
            # self.backbone1 = EfficientNet.from_name(
            #     'efficientnet-b3', in_channels=1, num_classes=1,use_swish=use_swish,use_batch_norm=use_batch_norm)
            # self.backbone2 = EfficientNet.from_name(
            #     'efficientnet-b3', in_channels=1, num_classes=1,use_swish=use_swish,use_batch_norm=use_batch_norm)
        
        if task=="direction":
            self.backbone1 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs)
            self.backbone2 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs)
        

        # feature_size = 2048 # v5
        # feature_size= 1280*2
        # Fusion module
        self.fusion_conv = nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=1)
        
        # Attention modules for each backbone
        # self.attention1 = SEBlock(int(feature_size/2))
        # self.attention2 = SEBlock(int(feature_size/2))
        # Add new layers
        # Asumiendo 1536 características de EfficientNet-B3
        if self.task=="type":
            self.fc_classification_1 = nn.Linear(feature_size, hidden_size)
            self.fc_classification_2 = nn.Linear(hidden_size, num_outputs)
            # v2
            # self.prelu_classification = nn.PReLU(hidden_size)

        if self.task=="energy":
            self.fc_energy_1 = nn.Linear(feature_size, hidden_size)
            self.fc_energy_2 = nn.Linear(hidden_size, int(num_outputs-1))
            # self.fc_energy_2 = nn.Linear(hidden_size, num_outputs)

            self.fc_energy_1_reg = nn.Linear(feature_size, hidden_size)
            # self.fc_energy_2_reg = nn.Linear(hidden_size, num_outputs)
            self.fc_energy_2_reg = nn.Linear(hidden_size, int(1))
            
            # if self.energy_bins is not None:
            #     # Get the difference con bin[index+1]-bin[index]
            #     differences = np.diff(self.energy_bins)
            #     anchor_values= np.insert(differences, 0, 1).astype(np.float32)
            #     # anchor_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            #     # Create the tensor from these values with gradient tracking enabled
            #     # self.energy_anchors = torch.tensor(anchor_values, dtype=torch.float32,device=self.device, requires_grad=False)
            #     self.energy_anchors = nn.Parameter(torch.tensor(anchor_values, dtype=torch.float32, device= self.device))
        
        if self.task=="direction":
            self.fc_direction_1 = nn.Linear(feature_size, hidden_size)
            self.fc_direction_2 = nn.Linear(hidden_size, num_outputs)
            
        self.swish = MemoryEfficientSwish()
        # self.batch_norm = nn.BatchNorm1d(feature_size*2)
        self.dropout_energy = nn.Dropout(0.3)
        self.dropout_direction = nn.Dropout(0.3)
        self.prelu_direction = nn.PReLU(num_parameters=hidden_size)
        self.prelu_energy_1 = nn.PReLU(num_parameters=hidden_size)
        self.prelu_energy_2 = nn.PReLU()
        self.relu_energy = nn.ReLU()

        self.batch_norm = nn.BatchNorm1d(feature_size)
        self.dropout_energy_1 = nn.Dropout(0.1)        
        self.dropout_energy_2 = nn.Dropout(0.3)     
           
    def extract_feature_vector(self,x1, x2):

        x1 = self.backbone1.extract_features(x1)
        # Backbone 2
        x2 = self.backbone2.extract_features(x2)

        # Normalización previa a la fusión
        # x1 = F.normalize(x1, p=2, dim=1)
        # x2 = F.normalize(x2, p=2, dim=1)
        # Fusion point
        fused_features = torch.add(x1, x2)  # Sum fusion
        # fused_features = torch.cat((x1, x2), dim=1)
        
        # Apply fusion module
        fused_features = self.fusion_conv(fused_features)
        
        # Global average pooling
        fused_features = F.adaptive_avg_pool2d(fused_features, 1)
        
        # Flatten
        fused_features = fused_features.view(fused_features.size(0), -1)

        return fused_features
    
    def forward(self, x1, x2):

        energy = [None,None]
        classification = [None,None]
        direction = [None,None]


        fused_features = self.extract_feature_vector(x1, x2)

        # Full connect layer and activation
        if self.task == "type":

            # v_2
            # classification = self.fc_classification_1(fused_features)
            # classification = self.prelu_classification(classification)  
            # classification = F.dropout(classification, p=0.1, training=self.training)  
            # classification = self.fc_classification_2(classification)

            # v_1
            classification = self.fc_classification_1(fused_features)
            classification = self.fc_classification_2(classification)
            classification = self.swish(classification)
            #v_x
            # classification = self.Dirichlet(classification)
            # if not self.training:
               
            #    reliability = (1.0-(self.num_outputs / classification.sum())) 
            #    # Normalize
            #    classification = classification / classification.sum()    
               
            #    classification = [classification, reliability]
        if self.task == "energy":

            # Exp 10
            #----------------------------------------------------------
            # NEW: Adding dropout and batch_norm here
            fused_features = self.batch_norm(fused_features)
            fused_features = self.dropout_energy_1(fused_features)
            #----------------------------------------------------------
            # if self.training:
            energy_class = self.fc_energy_1(fused_features)
            energy_class = self.fc_energy_2(energy_class)
            energy_class = self.swish(energy_class)
            energy_pred_class = self.swish(energy_class)
            # else:
            #     energy_pred_class= None

            energy_reg = self.fc_energy_1_reg(fused_features)
            energy_reg = self.dropout_energy_2(energy_reg)  # NEW: Adding dropout here
            energy_reg  = self.prelu_energy_1(energy_reg)
            energy_regresion = self.fc_energy_2_reg(energy_reg)


            # predicted = torch.softmax(energy_pred_class, dim=1)
            # predicted = predicted.argmax(dim=1)
            # # energy[:,-int(self.num_outputs/2):]=energy[:,-int(self.num_outputs/2):]*self.energy_anchors[predicted].unsqueeze(1)
            # energy_regresion = energy_reg.clone()  # First, clone the original tensor to preserve the computational graph
            # energy_regresion = energy_regresion * self.energy_anchors[predicted].unsqueeze(1)
            # energy = torch.concat([energy_pred_class,energy_regresion],dim=1)  # Assign the updated tensor back to energy
            energy = [energy_pred_class,energy_regresion]  # Assign the updated tensor back to energy

            # # Exp 8-x
            # #----------------------------------------------------------
            # # NEW: Adding dropout and batch_norm here
            # fused_features = self.batch_norm(fused_features)
            # fused_features = self.dropout_energy_1(fused_features)
            # #----------------------------------------------------------
            # energy_class = self.fc_energy_1(fused_features)
            # energy_class = self.fc_energy_2(energy_class)
            # energy_class = self.swish(energy_class)
            # energy_pred_class = self.swish(energy_class)
 
            # energy_reg = self.fc_energy_1_reg(fused_features)
            # energy_reg = self.dropout_energy_2(energy_reg)  # NEW: Adding dropout here
            # energy_reg  = self.prelu_energy_1(energy_reg)
            # energy_reg = self.fc_energy_2_reg(energy_reg)


            # predicted = torch.softmax(energy_pred_class, dim=1)
            # predicted = predicted.argmax(dim=1)
            # # energy[:,-int(self.num_outputs/2):]=energy[:,-int(self.num_outputs/2):]*self.energy_anchors[predicted].unsqueeze(1)
            # energy_regresion = energy_reg.clone()  # First, clone the original tensor to preserve the computational graph
            # energy_regresion = energy_regresion * self.energy_anchors[predicted].unsqueeze(1)
            # energy = torch.concat([energy_pred_class,energy_regresion],dim=1)  # Assign the updated tensor back to energy
            #---------------------------------------------------------------------------------------------------------------------------
            # Exp 3-7
            # energy = self.fc_energy_1(fused_features)
            # energy = self.fc_energy_2(energy)
            # energy = self.swish(energy)
            # energy_pred_class = self.swish(energy[:,0:int(self.num_outputs/2)])
 
            # predicted = torch.softmax(energy_pred_class, dim=1)
            # predicted = predicted.argmax(dim=1)
            # # energy[:,-int(self.num_outputs/2):]=energy[:,-int(self.num_outputs/2):]*self.energy_anchors[predicted].unsqueeze(1)
            # energy_regresion = energy[:, -int(self.num_outputs/2):].clone()  # First, clone the original tensor to preserve the computational graph
            # energy_regresion = energy_regresion * self.energy_anchors[predicted].unsqueeze(1)
            # energy = torch.concat([energy_pred_class,energy_regresion],dim=1)  # Assign the updated tensor back to energy
            #---------------------------------------------------------------------------------------------------------------------------
            # classification = self.swish(classification)
            # Old code 
            # energy = self.fc_energy_1(fused_features)
            # energy = self.dropout_energy(energy)
            # energy = self.fc_energy_2(energy)

        if self.task == "direction":
            fused_features = self.dropout_direction(fused_features)
            direction = self.fc_direction_1(fused_features)
            direction = self.fc_direction_2(direction)

         
 
        return [classification,fused_features], energy, direction

    def eval(self):
        super().eval()  
        self.backbone1.eval()
        self.backbone2.eval()

    def train(self, mode=True):
        super().train(mode) 
        self.backbone1.train(mode)
        self.backbone2.train(mode)
