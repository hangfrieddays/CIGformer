from .modules import *
import torch.nn.functional as F
from .utils.registry import MODEL_REGISTRY
import torch
import torch.nn as nn
from datasets.utils import data_denormalize


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # Initialize weights of Conv2d or Linear layers using Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                # Initialize bias of Conv2d or Linear layers to zeros
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


@MODEL_REGISTRY.register()
class CIGformer(BaseModel):
    def __init__(self, cfg):
        super(PCGformer, self).__init__()
        model_cfg = cfg.get('modules_configs', dict())
        pan_channel = cfg.get('pan_channel')  # channel of pan
        ms_channel = cfg.get('ms_channel')  # channel of ms
        last_channels = model_cfg.get('last_channels')  # channel of last layer
        base_dim = model_cfg.get('base_dim')  # base dimension of PCGNet for encoder
        win_size = model_cfg.get('win_size')
        guidance = model_cfg.get('guidance')
        self.guidance = guidance

        self.SubstituteModule = SubstituteModule(pan_channel, ms_channel, base_dim)
        self.FusionModule = FusionModule(last_channels)

        self.number_layers = cfg.get('number_PCG_layers')  # numbers of layers in PCGNet

        self.PCG_layers = nn.ModuleDict()

        self.conv_layer_2 = conv1x1(base_dim, base_dim)
        self.conv_layer_3 = conv1x1(base_dim, base_dim * 2)
        if guidance:
            self.conv_pan = conv1x1(pan_channel, base_dim)
            self.conv_ms = conv1x1(ms_channel, base_dim)

        for layer in range(self.number_layers):
            n_feats = model_cfg[f'n_feats_{layer + 1}']

            enc_n_feats = base_dim * 2 ** layer
            dec_n_feats = n_feats if layer != 0 else n_feats * 2

            n_heads = model_cfg[f'n_heads_{layer + 1}']
            head_dim = model_cfg[f'head_dim_{layer + 1}']
            downsample_factor = model_cfg[f'downsample_factor_{layer + 1}']
            guide_enc_feats = enc_n_feats

            self.PCG_layers[f'layer_{layer}'] = nn.ModuleDict({
                'encoder': Encoder(n_feats=enc_n_feats, n_heads=n_heads, head_dim=head_dim, win_size=win_size, downsample_factor=downsample_factor, guidance= guidance),
                'decoder': Decoder(n_feats=dec_n_feats, n_heads=n_heads, head_dim=head_dim, win_size=win_size, downsample_factor=1),
                'guide_module': IntermediateGuidanceModule(n_feats=guide_enc_feats, n_heads=n_heads,
                                               head_dim=enc_n_feats // n_heads, win_size=win_size, n_blocks=1),
            })


    def forward(self, pan, ms, TransferNet):
        """
            input:
                1. pan: (B, 1, H, W)
                2. ms:  (B, 4, H/4, W/4)
            output:
                1. HMS: (B, 4, H, W)
        """
        H, W = ms.shape[2] * 4, ms.shape[3] * 4
        ms_up = F.interpolate(ms, size=(H, W), mode='bicubic', align_corners=False)
        feature_lis = []
        pan_down = F.interpolate(pan, scale_factor=0.25, mode='bilinear', align_corners=False)
        pan_down_up = F.interpolate(pan_down, scale_factor=4, mode='bilinear', align_corners=False)

        if TransferNet != None:
            ms_to_pan = TransferNet(ms_up)
            ms_up -= ms_to_pan

        pan_substitute = pan + ms_to_pan - pan_down_up
        ms_substitute = ms_up - ms_to_pan

        combined_figure = torch.concat((pan, ms_up), dim=1)
        combined_feature = self.SubstituteModule(combined_figure)  # 16

        inter_feature = combined_feature    # 16
        if self.guidance:
            ms_feature = F.relu(self.conv_ms(ms_substitute))    # 16
            pan_feature = F.relu(self.conv_pan(pan_substitute)) # 16
        # encode
        for layer in range(self.number_layers):
            # guide
            if self.guidance:
                inter_feature, pan_feature, ms_feature = self.PCG_layers[f'layer_{layer}']['encoder'](inter_feature, pan_feature, ms_feature)  # 16 -> 32 -> 64
                feat_guide = self.PCG_layers[f'layer_{layer}']['guide_module'](ms_feature, pan_feature, inter_feature)

                feature = feat_guide + inter_feature
            else:
                feature = self.PCG_layers[f'layer_{layer}']['encoder'](inter_feature)
            inter_feature = feature
            feature_lis.append(feature)

        # decode
        # layer 3
        feature = feature_lis[2]  # 64
        feature = pixel_shuffle(feature, 2)  # 16
        feature = self.conv_layer_3(feature)  # 16 -> 32
        feature = F.leaky_relu(feature, 0.2)
        dec_res_layer3 = self.PCG_layers[f'layer_{2}']['decoder'](feature)  # 32

        # layer 2
        feature = feature_lis[1]  # 32
        feature = torch.cat((feature, dec_res_layer3), dim=1)  # 64
        feature = pixel_shuffle(feature, 2)  # 16
        feature = self.conv_layer_2(feature)  # 16 -> 32
        feature = F.leaky_relu(feature, 0.2)
        dec_res_layer2 = self.PCG_layers[f'layer_{1}']['decoder'](feature)  # 32

        # layer 1
        feature = feature_lis[0]  # 16
        feature = torch.cat((feature, dec_res_layer2), dim=1)  # 32
        dec_res_layer1 = self.PCG_layers[f'layer_{0}']['decoder'](feature)  # 32

        # recovery
        result = self.FusionModule(dec_res_layer1) + ms_up  # 4
        return result


# Transfer Network Archtecture
# for the detailed introduction of this module, please see https://github.com/Baixuzx7/P2Sharpen
@MODEL_REGISTRY.register()
class TransferNetwork(nn.Module):
    def __init__(self, cfg):
        super(TransferNetwork, self).__init__()
        self.cfg = cfg
        self.Layer1 = nn.Sequential(
            # Layer1
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=self.cfg['in_channel'], out_channels=16, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=16)
        )
        self.Layer2 = nn.Sequential(
            # Layer2
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=16)
        )
        self.Layer3 = nn.Sequential(
            # Layer3
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=16)
        )
        # Concatanate
        self.Layer4 = nn.Sequential(
            # Layer4
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=16)
        )
        # Concatanate
        self.Layer5 = nn.Sequential(
            # Layer5
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=16 + self.cfg['in_channel'], out_channels=8, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=8)
        )
        self.Layer6 = nn.Sequential(
            # Layer6
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1),  bias=True),
            nn.Tanh()
        )

    def initialize(self):
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                nn.init.trunc_normal_(model.weight, mean=0.0, std=1e-3)
                nn.init.constant_(model.bias, val=0.0)

    def forward(self, x):
        y1 = self.Layer1(x)
        y2 = self.Layer2(y1)
        y3 = self.Layer3(y2)
        x1 = torch.cat([y1, y3], dim=1)
        y4 = self.Layer4(x1)
        x2 = torch.cat([x, y4], dim=1)
        y5 = self.Layer5(x2)
        y6 = self.Layer6(y5)
        return y6



