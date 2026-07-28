"""
This module defines the ``CTLearnModel`` classes, which holds the basic functionality for creating a PyTorch model to be used in CTLearn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these custom attention blocks are updated to return torch.nn.Module or used dynamically
from ctlearn.core.model import (
    SingleCNN,
    ResNet,
    LoadedModel,
)
from ctlearn.core.pytorch.attention import (
    DualSqueezeExciteBlock,
    ChannelSqueezeExciteBlock,
    SpatialSqueezeExciteBlock,
)

__all__ = [
    "BasicBlock",
    "BottleneckBlock",
    "MultiHeadClassifier",
    "build_fully_connect_pytorch_head",
    "PyTorchSingleCNN",
    "PyTorchResNet",
    "PyTorchLoadedModel",
]


class MultiHeadClassifier(nn.Module):
    """
    A PyTorch container module to hold the multi-task fully connected heads.
    """
    def __init__(self, heads_dict, single_output_task=None):
        super().__init__()
        # Sanitize keys because 'type' conflicts with nn.Module.type() method
        self._task_mapping = {
            task: f"head_{task}" if hasattr(nn.Module, task) else task
            for task in heads_dict.keys()
        }
        sanitized_heads = {
            self._task_mapping[task]: module for task, module in heads_dict.items()
        }
        self.heads = nn.ModuleDict(sanitized_heads)
        self.single_output_task = single_output_task

    def forward(self, x):
        # Flatten the backbone output if it's spatially aggregated (B, C, 1, 1) -> (B, C)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
            
        classification = None
        energy = None
        direction = None

        for original_task, internal_key in self._task_mapping.items():
            head = self.heads[internal_key]
            out = head(x)
            
            if original_task == "type":
                classification = F.softmax(out, dim=-1)
            elif original_task == "energy":
                energy = out
            elif original_task in ["direction", "skydirection", "cameradirection"]:
                direction = out
                
        return classification, energy, direction


def build_fully_connect_pytorch_head(in_features, layers, activation_function, tasks):
    """
    Build the fully connected head for the PyTorch-based CTLearn model.
    """
    heads = {}
    
    # Activation mapping from Keras string to PyTorch Module
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid
    }

    for task in tasks:
        task_layers = []
        current_features = in_features
        
        for i, units in enumerate(layers[task]):
            task_layers.append(nn.Linear(current_features, units))
            if i != len(layers[task]) - 1:
                act_cls = act_map.get(activation_function[task].lower(), nn.ReLU)
                task_layers.append(act_cls())
            current_features = units
            
        heads[task] = nn.Sequential(*task_layers)

    single_output_task = tasks[0] if (len(tasks) == 1 and tasks[0] == "type") else None
    return MultiHeadClassifier(heads, single_output_task=single_output_task)


class FullModelPipeline(nn.Module):
    """
    Combines the backbone and multi-task heads into a unified executable nn.Module pipeline.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class PyTorchSingleCNN(SingleCNN):
    """
    ``SingleCNN`` is a simple convolutional neural network model implemented in PyTorch.
    """

    def __init__(self, input_shape, tasks, config=None, parent=None, **kwargs):
        super().__init__(tasks=tasks, config=config, parent=parent, **kwargs)
        
        # Build modules
        self.backbone_model, out_features = self._build_backbone(input_shape)     
        self.logits_head = build_fully_connect_pytorch_head(
            out_features, self.head_layers, self.head_activation_function, tasks
        )
        # Final native PyTorch module pipeline saved into self.model
        self.model = FullModelPipeline(self.backbone_model, self.logits_head)

    def _build_backbone(self, input_shape):
        # input_shape format: (channels, height, width)
        in_channels = input_shape[-1]
        modules = []

        if self.batchnorm:
            modules.append(nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)) # PyTorch momentum = 1 - Keras momentum

        for i, layer in enumerate(self.architecture):
            filters = layer["filters"]
            kernel_size = layer["kernel_size"]
            number = layer["number"]

            for nr in range(number):
                # padding="same" calculates padding dynamically in PyTorch based on kernel size
                padding = kernel_size // 2 
                modules.append(nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=padding))
                modules.append(nn.ReLU())
                in_channels = filters
            
            if self.pooling_type is not None:
                p_size = self.pooling_parameters["size"]
                p_stride = self.pooling_parameters["strides"]
                if self.pooling_type == "max":
                    modules.append(nn.MaxPool2d(kernel_size=p_size, stride=p_stride))
                elif self.pooling_type == "average":
                    modules.append(nn.AvgPool2d(kernel_size=p_size, stride=p_stride))
                    
            if self.batchnorm:
                modules.append(nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3))

        if self.bottleneck_filters is not None:
            modules.append(nn.Conv2d(in_channels, self.bottleneck_filters, kernel_size=1))
            modules.append(nn.ReLU())
            in_channels = self.bottleneck_filters
            if self.batchnorm:
                modules.append(nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3))

        if self.attention is not None:
            mech = self.attention.get("mechanism")
            ratio = self.attention.get("reduction_ratio", 16)
            if mech == "Dual-SE":
                attention_layer = DualSqueezeExciteBlock(in_channels=in_channels, ratio=ratio)
            elif mech == "Channel-SE":
                attention_layer = ChannelSqueezeExciteBlock(in_channels=in_channels, ratio=ratio)
            elif mech == "Spatial-SE":
                attention_layer = SpatialSqueezeExciteBlock(in_channels=in_channels)
            modules.append(attention_layer)

        # Global Average Pooling wrapper block
        class GlobalAvgPool(nn.Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        modules.append(GlobalAvgPool())
        
        backbone_model = nn.Sequential(*modules)
        return backbone_model, in_channels


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_shortcut=True, attention=None):
        super().__init__()
        self.conv_shortcut = conv_shortcut
        self.attention_config = attention
        # Projection Shortcut Branch
        if conv_shortcut:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=True
            )
        else:
            self.shortcut = None
        # Main Branch Convolutions (Matching Keras _1_conv and _2_conv)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        # Setup the attention mechanism
        self.setup_attention(out_channels)

    def setup_attention(self, channels):
        self.attn_layer = None
        if self.attention_config:
            mech = self.attention_config.get("mechanism")
            ratio = self.attention_config.get("reduction_ratio", 16)
            if mech == "Dual-SE":
                self.attn_layer = DualSqueezeExciteBlock(in_channels=channels, ratio=ratio)
            elif mech == "Channel-SE":
                self.attn_layer = ChannelSqueezeExciteBlock(in_channels=channels, ratio=ratio)
            elif mech == "Spatial-SE":
                self.attn_layer = SpatialSqueezeExciteBlock(in_channels=channels)

    def forward(self, x):
        # Shortcut path
        if self.conv_shortcut and self.shortcut is not None:
            identity = self.shortcut(x)
        else:
            identity = x

        # Main path (Matches Keras activation order: conv1 -> relu -> conv2)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        if self.attn_layer is not None:
            out = self.attn_layer(out)

        out += identity
        return F.relu(out)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, base_filters, stride=1, conv_shortcut=True, attention=None):
        super().__init__()
        self.conv_shortcut = conv_shortcut
        self.attention_config = attention

        # Shortcut connection
        if conv_shortcut:
            self.shortcut = nn.Conv2d(
                in_channels, 4 * base_filters, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()
        # Main branch convolutions matching Keras layout:
        self.conv1 = nn.Conv2d(
            in_channels, base_filters, kernel_size=1, stride=stride, bias=False
        )
        # Keras _2_conv is 3x3 spatial convolution with stride=1 (since stride is handled in _1_conv)
        self.conv2 = nn.Conv2d(
            base_filters, base_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Keras _3_conv restores channels back to 4 * base_filters
        self.conv3 = nn.Conv2d(
            base_filters, 4 * base_filters, kernel_size=1, bias=False
        )
        # Setup the attention mechanism
        self.setup_attention(4 * base_filters)

    def setup_attention(self, channels):
        self.attn_layer = None
        if self.attention_config:
            mech = self.attention_config["mechanism"]
            ratio = self.attention_config.get("reduction_ratio", 16)
            if mech == "Dual-SE":
                self.attn_layer = DualSqueezeExciteBlock(in_channels=channels, ratio=ratio)
            elif mech == "Channel-SE":
                self.attn_layer = ChannelSqueezeExciteBlock(in_channels=channels, ratio=ratio)
            elif mech == "Spatial-SE":
                self.attn_layer = SpatialSqueezeExciteBlock(in_channels=channels)

    def forward(self, x):
        identity = self.shortcut(x)
        
        # Matches Keras sequence: conv1 (with stride) -> relu -> conv2 -> relu -> conv3
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        if self.attn_layer:
            out = self.attn_layer(out)
            
        out += identity
        return F.relu(out)

class PyTorchResNet(ResNet):
    """
    ``PyTorchResNet`` is a residual neural network model implemented in PyTorch.
    """

    def __init__(self, input_shape, tasks, config=None, parent=None, **kwargs):
        super().__init__(tasks=tasks, config=config, parent=parent, **kwargs)

        # Build PyTorch backbone and track final out_features channel size
        self.backbone_model, out_features = self._build_backbone(input_shape)
        # Build the fully connected head
        self.logits_head = build_fully_connect_pytorch_head(
            out_features, self.head_layers, self.head_activation_function, tasks
        )
        # Unify into our structural pipeline wrapper module
        self.model = FullModelPipeline(self.backbone_model, self.logits_head)

    def _build_backbone(self, input_shape):
        in_channels = input_shape[-1]
        modules = []

        # Initial Zero Padding
        if getattr(self, "init_padding", 0) > 0:
            modules.append(nn.ZeroPad2d(self.init_padding))

        # Initial Conv Layer
        if getattr(self, "init_layer", None) is not None:
            out_ch = self.init_layer["filters"]
            k_size = self.init_layer["kernel_size"]
            stride = self.init_layer["strides"]
            padding = k_size // 2 
            
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_ch,
                    kernel_size=k_size,
                    stride=stride,
                    padding=padding,
                    bias=True,  # Match Keras default bias if applicable
                )
            )
            modules.append(nn.ReLU())
            in_channels = out_ch

        # Initial Max Pooling
        if getattr(self, "init_max_pool", None) is not None:
            p_size = self.init_max_pool["size"]
            p_stride = self.init_max_pool["strides"]
            modules.append(
                nn.MaxPool2d(kernel_size=p_size, stride=p_stride, padding=0)
            )

        # Assemble Stacked Residual Architecture blocks
        res_blocks, final_channels = self._stacked_res_blocks(
            in_channels,
            architecture=self.architecture,
            residual_block_type=self.residual_block_type,
            attention=self.attention,
        )
        modules.extend(res_blocks)

        # Global Average Pooling setup
        class GlobalAvgPool(nn.Module):
            def forward(self, x):
                x = F.adaptive_avg_pool2d(x, (1, 1))
                return x.view(x.size(0), -1)

        modules.append(GlobalAvgPool())

        return nn.Sequential(*modules), final_channels

    def _stacked_res_blocks(self, in_channels, architecture, residual_block_type, attention):
        blocks_list = []
        current_channels = in_channels
        
        filters_list = [layer["filters"] for layer in architecture]
        blocks_count = [layer["blocks"] for layer in architecture]
        
        # First layer block sequence (stride=1)
        blocks_list.extend(
            self._stack_fn(
                current_channels,
                filters_list[0],
                blocks_count[0],
                residual_block_type,
                stride=1,
                attention=attention,
            )
        )
        
        multiplier = 4 if residual_block_type == "bottleneck" else 1
        current_channels = filters_list[0] * multiplier
        
        # Subsequent downsampling levels (stride=2)
        for filters, blocks in zip(filters_list[1:], blocks_count[1:]):
            blocks_list.extend(
                self._stack_fn(
                    current_channels,
                    filters,
                    blocks,
                    residual_block_type,
                    stride=2,
                    attention=attention,
                )
            )
            current_channels = filters * multiplier
            
        return blocks_list, current_channels
    
    def _stack_fn(self, in_channels, filters, blocks, residual_block_type, stride=2, attention=None):
        stack = []
        # Bottleneck blocks expand channels by 4x; Basic blocks do not expand.
        multiplier = 4 if residual_block_type == "bottleneck" else 1
        out_channels = filters * multiplier

        def build_block(in_c, s):
            # Only use a conv shortcut if channels change or if downsampling (stride > 1)
            needs_shortcut = (in_c != out_channels) or (s != 1)            
            if residual_block_type == "basic":
                return BasicBlock(
                    in_channels=in_c,
                    out_channels=filters,
                    stride=s,
                    conv_shortcut=needs_shortcut,
                    attention=attention,
                )
            else:
                return BottleneckBlock(
                    in_channels=in_c,
                    base_filters=filters,
                    stride=s,
                    conv_shortcut=needs_shortcut,
                    attention=attention,
                )

        # First block transition
        stack.append(build_block(in_channels, s=stride))
        
        # Remaining blocks in the layer
        for _ in range(1, blocks):
            stack.append(build_block(out_channels, s=1))
            
        return stack


class PyTorchLoadedModel(LoadedModel):
    """
    ``PyTorchLoadedModel`` handles loading a pre-saved PyTorch model weight layout.
    """

    def __init__(self, input_shape, tasks, config=None, parent=None, **kwargs):
        super().__init__(tasks=tasks, config=config, parent=parent, **kwargs)

        # In PyTorch, instead of load_model returning an arbitrary configuration blindly, 
        # you instantiate the structure first and pass a state_dict or weights file path.
        self.model = torch.load(self.load_model_from)
        
        if self.overwrite_head:
            # Freeze/unfreeze backbone based on config choice
            for param in self.model.backbone.parameters():
                param.requires_grad = self.trainable_backbone
                
            # Fetch out_features dynamically from existing backbone configuration
            # This is a representative placeholder pattern for your architecture
            out_features = self.head_layers[tasks[0]][0] 
            
            self.logits_head = build_fully_connect_pytorch_head(
                out_features, self.head_layers, self.head_activation_function, tasks
            )
            self.model.head = self.logits_head