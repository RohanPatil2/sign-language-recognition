import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, LayerNorm, trunc_normal_
from torchvision.ops import StochasticDepth

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation attention block with LayerScale"""
    def __init__(self, dim, reduction_ratio=8, init_value=1e-2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction_ratio, dim, bias=False),
            nn.Sigmoid()
        )
        self.ls = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.ls.view(1, c, 1, 1) * y

class SpatialAttention(nn.Module):
    """Coordinated Attention with depth-wise convolution"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act(x)
        return identity + x

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base model with flexible pretraining
        self.backbone = timm.create_model(
            config['model_name'],
            pretrained=config.get('pretrained', True),
            features_only=True,
            out_indices=config.get('out_indices', [2, 3, 4]),
            drop_rate=config.get('drop_rate', 0.2),
            attn_drop_rate=config.get('attn_drop_rate', 0.1),
            norm_layer=partial(LayerNorm, eps=1e-6)
        )
        
        # Feature enhancement modules
        self.attention_layers = nn.ModuleList()
        self.stochastic_depth = StochasticDepth(p=config.get('drop_path_rate', 0.2), mode='row')
        
        # Attention type configuration
        attn_type = config.get('attention_type', 'se')
        for _ in range(config.get('num_attention_layers', 3)):
            if attn_type == 'se':
                self.attention_layers.append(ChannelAttention(self.backbone.num_features))
            elif attn_type == 'spatial':
                self.attention_layers.append(SpatialAttention(self.backbone.num_features))
        
        # Feature pyramid fusion
        self.fpn = nn.ModuleDict({
            'lateral_conv': nn.Conv2d(256, 256, 1),
            'smooth_conv': nn.Conv2d(256, 256, 3, padding=1)
        })
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(1) if config.get('global_pool', True) else nn.Identity()
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, config['embed_dim'])
        ) if config.get('use_projection', False) else None
        
        # Classification head
        self.classifier = nn.Linear(self.backbone.num_features, config['num_classes']) if config.get('num_classes') else None
        
        # Weight initialization
        self._init_weights()
        
        # Freeze backbone if specified
        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # Extract hierarchical features
        features = self.backbone(x)
        
        # Feature pyramid network fusion
        pyramid_features = []
        last_feature = None
        for i, feat in enumerate(reversed(features)):
            if i == 0:
                last_feature = self.fpn['lateral_conv'](feat)
            else:
                lateral = self.fpn['lateral_conv'](feat)
                last_feature = lateral + F.interpolate(last_feature, scale_factor=2)
            pyramid_features.append(self.fpn['smooth_conv'](last_feature))
        
        # Attention processing with stochastic depth
        x = torch.cat(pyramid_features, dim=1)
        for attn in self.attention_layers:
            x = self.stochastic_depth(attn(x)) + x
        
        return self.pool(x).flatten(1)

    def forward(self, x, mode='features'):
        x = self.forward_features(x)
        
        if mode == 'features':
            return x
        elif mode == 'projection' and self.projection_head is not None:
            return F.normalize(self.projection_head(x), p=2, dim=1)
        elif mode == 'logits' and self.classifier is not None:
            return self.classifier(x)
        else:
            raise ValueError(f"Invalid forward mode: {mode}")