import torch
import torch.nn as nn
import torch.nn.functional as F

class DualSqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.cse = ChannelSqueezeExciteBlock(in_channels=in_channels, ratio=ratio)
        self.sse = SpatialSqueezeExciteBlock(in_channels=in_channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)

class ChannelSqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))
        excitation = self.gate(squeeze)
        return x * excitation

class SpatialSqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        spatial_mask = torch.sigmoid(self.spatial_conv(x))
        return x * spatial_mask

class MultiHeadClassifier(nn.Module):
    def __init__(self, heads_dict, task):
        super().__init__()
        self.heads = nn.ModuleDict(heads_dict)
        self.task = task

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
            
        classification = None
        energy = None
        direction = None

        if self.task == "type" and "type" in self.heads:
            classification = self.heads["type"](x)
        if self.task == "energy" and "energy" in self.heads:
            energy = self.heads["energy"](x)
        if self.task == "direction" and "direction" in self.heads:
            direction = self.heads["direction"](x)
            
        return classification, energy, direction

def pytorch_build_fully_connect_head(in_features, layers, activation_function, tasks):
    heads = {}
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid
    }

    for task in tasks:
        if task not in layers:
            continue
        task_layers = []
        current_features = in_features
        
        for i, units in enumerate(layers[task]):
            task_layers.append(nn.Linear(current_features, units))
            if i != len(layers[task]) - 1:
                act_cls = act_map.get(activation_function[task].lower(), nn.ReLU)
                task_layers.append(act_cls())
            current_features = units
            
        heads[task] = nn.Sequential(*task_layers)

    task_str = tasks[0] if len(tasks) > 0 else "type"
    return MultiHeadClassifier(heads, task=task_str)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_shortcut=True, attention=None):
        super().__init__()
        self.conv_shortcut = conv_shortcut
        self.attention_config = attention
        
        if conv_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.setup_attention(out_channels)

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
        
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        
        if self.attn_layer:
            out = self.attn_layer(out)
            
        out += identity
        return F.relu(out)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, base_filters, stride=1, conv_shortcut=True, attention=None):
        super().__init__()
        self.conv_shortcut = conv_shortcut
        self.attention_config = attention
        out_channels = 4 * base_filters
        
        if conv_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
            
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(base_filters, out_channels, kernel_size=1, bias=False)
        
        self.setup_attention(out_channels)

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
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        if self.attn_layer:
            out = self.attn_layer(out)
            
        out += identity
        return F.relu(out)

class PyTorchResNetModel(nn.Module):
    def __init__(
        self,
        task="type",
        num_inputs=1,
        num_outputs=2,
        init_padding=0,
        init_layer=None,
        init_max_pool=None,
        residual_block_type="bottleneck",
        architecture=None,
        head_layers=None,
        head_activation_function=None,
        attention_mechanism="Dual-SE",
        attention_reduction_ratio=16,
    ):
        super().__init__()
        self.task = task
        self.init_padding = init_padding
        self.init_layer = init_layer
        self.init_max_pool = init_max_pool
        self.residual_block_type = residual_block_type
        
        if architecture is None:
            architecture = [
                {"filters": 48, "blocks": 2},
                {"filters": 96, "blocks": 3},
                {"filters": 128, "blocks": 3},
                {"filters": 256, "blocks": 3},
            ]
        self.architecture = architecture
        
        if head_layers is None:
            head_layers = {
                "type": [512, 256, num_outputs],
                "energy": [512, 256, 1],
                "direction": [512, 256, 2],
            }
        
        if head_activation_function is None:
            head_activation_function = {
                "type": "relu",
                "energy": "relu",
                "direction": "tanh",
            }
            
        self.attention = None
        if attention_mechanism is not None:
            self.attention = {
                "mechanism": attention_mechanism,
                "reduction_ratio": attention_reduction_ratio,
            }

        input_shape = (num_inputs, 224, 224) # Spatial dims don't affect init
        self.backbone_model, out_features = self._build_backbone(input_shape)
        
        self.logits_head = pytorch_build_fully_connect_head(
            out_features, head_layers, head_activation_function, [task]
        )

    def _build_backbone(self, input_shape):
        in_channels = input_shape[0]
        modules = []

        if self.init_padding > 0:
            modules.append(nn.ZeroPad2d(self.init_padding))

        if self.init_layer is not None:
            out_ch = self.init_layer["filters"]
            k_size = self.init_layer["kernel_size"]
            stride = self.init_layer["strides"]
            padding = k_size // 2 
            
            modules.append(nn.Conv2d(in_channels, out_ch, kernel_size=k_size, stride=stride, padding=padding, bias=False))
            modules.append(nn.ReLU())
            in_channels = out_ch

        if self.init_max_pool is not None:
            p_size = self.init_max_pool["size"]
            p_stride = self.init_max_pool["strides"]
            modules.append(nn.MaxPool2d(kernel_size=p_size, stride=p_stride, padding=p_size // 2))

        res_blocks, final_channels = self._stacked_res_blocks(
            in_channels,
            architecture=self.architecture,
            residual_block_type=self.residual_block_type,
            attention=self.attention
        )
        modules.extend(res_blocks)

        class GlobalAvgPool(nn.Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        modules.append(GlobalAvgPool())

        return nn.Sequential(*modules), final_channels

    def _stacked_res_blocks(self, in_channels, architecture, residual_block_type, attention):
        blocks_list = []
        current_channels = in_channels
        
        filters_list = [layer["filters"] for layer in architecture]
        blocks_count = [layer["blocks"] for layer in architecture]
        
        blocks_list.extend(self._stack_fn(
            current_channels, filters_list[0], blocks_count[0], residual_block_type, stride=1, attention=attention
        ))
        
        multiplier = 4 if residual_block_type == "bottleneck" else 1
        current_channels = filters_list[0] * multiplier
        
        for filters, blocks in zip(filters_list[1:], blocks_count[1:]):
            blocks_list.extend(self._stack_fn(
                current_channels, filters, blocks, residual_block_type, stride=2, attention=attention
            ))
            current_channels = filters * multiplier
            
        return blocks_list, current_channels
    
    def _stack_fn(self, in_channels, filters, blocks, residual_block_type, stride=2, attention=None):
        block_layer = BasicBlock if residual_block_type == "basic" else BottleneckBlock
        stack = []
        
        base_kwargs = {
            "in_channels": in_channels,
            "stride": stride,
            "attention": attention
        }
        
        if residual_block_type == "basic":
            base_kwargs["out_channels"] = filters
            base_kwargs["base_filters"] = filters 
        else:
            base_kwargs["base_filters"] = filters

        stack.append(block_layer(conv_shortcut=True, **base_kwargs))
        
        multiplier = 4 if residual_block_type == "bottleneck" else 1
        current_in = filters * multiplier
        
        for _ in range(1, blocks):
            next_kwargs = {
                "in_channels": current_in,
                "stride": 1,
                "attention": attention,
                "conv_shortcut": False
            }
            if residual_block_type == "basic":
                next_kwargs["out_channels"] = filters
                next_kwargs["base_filters"] = filters
            else:
                next_kwargs["base_filters"] = filters
                
            stack.append(block_layer(**next_kwargs))
            
        return stack

    def forward(self, x):
        features = self.backbone_model(x)
        return self.logits_head(features)
