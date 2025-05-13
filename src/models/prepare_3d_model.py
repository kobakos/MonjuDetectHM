from typing import Optional
from functools import partial
# Copied from Quishen Ha's notebook:
import segmentation_models_pytorch as smp
from torch import nn
import torch.nn.functional as F
import torch
import timm
import timm_3d

def convert_3d(module, relu2gelu=False):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, torch.nn.Conv2d):
        #print('Conv #############################################')
        #print(module.kernel_size, module.stride, module.padding, module.dilation)
        #print('#############################################')
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=[module.stride[0]]*3,#[1, module.stride, module.stride] if isinstance(module.stride, int) else [1, module.stride[0], module.stride[1]],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-3).repeat(1,1, module.kernel_size[0],1,1))

    elif isinstance(module, torch.nn.MaxPool2d):
        #print('MaxPool #############################################')
        #print(module.kernel_size, module.stride, module.padding, module.dilation, module.ceil_mode)
        #print('#############################################')
        module_output = torch.nn.MaxPool3d(
            kernel_size = module.kernel_size,#[module.kernel_size - 1, module.kernel_size, module.kernel_size] if isinstance(module.kernel_size, int) else [1, module.kernel_size[0], module.kernel_size[1]],
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        #print('#############################################')
        #print(module.kernel_size, module.stride, module.padding, module.ceil_mode)
        #print('#############################################')
        module_output = torch.nn.AvgPool3d(
            kernel_size = module.kernel_size,#[1, module.kernel_size, module.kernel_size] if isinstance(module.kernel_size, int) else [1, module.kernel_size[0], module.kernel_size[1]],
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.UpsamplingBilinear2d):
        module_output = torch.nn.Upsample(
            scale_factor=module.scale_factor,
            mode='trilinear',
            align_corners=module.align_corners
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output

def convert_3d_inception(module, relu2gelu=False):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, torch.nn.Conv2d):
        #print('Conv #############################################')
        #print(module.kernel_size, module.stride, module.padding, module.dilation)
        #print('#############################################')
        # if the convolution kernel is in one direction as in incpeption net
        if module.kernel_size [0] != module.kernel_size[1]:
            module_output = torch.nn.Conv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=(1, module.kernel_size[0], module.kernel_size[1]),
                stride=(1, module.stride[0], module.stride[1]),
                padding=(0, module.padding[0], module.padding[1]),
                dilation=(1, module.dilation[0], module.dilation[1]),
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode
            )
            module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-3))
        else:
            module_output = torch.nn.Conv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=[module.stride[0]]*3,#[1, module.stride, module.stride] if isinstance(module.stride, int) else [1, module.stride[0], module.stride[1]],
                padding=module.padding[0],
                dilation=module.dilation[0],
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode
            )
            module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-3).repeat(1,1, module.kernel_size[0],1,1))

    elif isinstance(module, torch.nn.MaxPool2d):
        #print('MaxPool #############################################')
        #print(module.kernel_size, module.stride, module.padding, module.dilation, module.ceil_mode)
        #print('#############################################')
        module_output = torch.nn.MaxPool3d(
            kernel_size = module.kernel_size,#[module.kernel_size - 1, module.kernel_size, module.kernel_size] if isinstance(module.kernel_size, int) else [1, module.kernel_size[0], module.kernel_size[1]],
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        #print('#############################################')
        #print(module.kernel_size, module.stride, module.padding, module.ceil_mode)
        #print('#############################################')
        module_output = torch.nn.AvgPool3d(
            kernel_size = module.kernel_size,#[1, module.kernel_size, module.kernel_size] if isinstance(module.kernel_size, int) else [1, module.kernel_size[0], module.kernel_size[1]],
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.UpsamplingBilinear2d):
        module_output = torch.nn.Upsample(
            scale_factor=module.scale_factor,
            mode='trilinear',
            align_corners=module.align_corners
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d_inception(child)
        )
    del module

    return module_output

from torch.autograd import Function
class GradientReversalLayer(Function):
    @staticmethod
    def forward(context, x, constant):
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        return grad.neg() * context.constant, None
    
import numpy as np
def create_alpha_scheduler(cfg):
    type = cfg["type"]
    if type == "sigmoid":
        gamma = cfg["gamma"]
        total = cfg["total_epoch"]
        def schedule(epoch):
            return 2 / (1 + np.exp(-gamma * epoch / total)) - 1
        return schedule
    else:
        raise ValueError(f'Unknown type: {type}')


class DomainAdversarialHead(nn.Module):
    def __init__(self,
        dann_scheduler_cfg, in_dim = 2048, dims = [256], n_domains = 2):
        super(DomainAdversarialHead, self).__init__()
        dims = [in_dim] + dims + [n_domains]
        dims = zip(dims[:-1], dims[1:])
        self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in dims])
        self.alpha_scheduler = create_alpha_scheduler(dann_scheduler_cfg)
    def forward(self, x, epoch):
        alpha = self.alpha_scheduler(epoch)
        x = GradientReversalLayer.apply(x, alpha)
        for fc in self.fcs:
            x = F.gelu(fc(x))
        return x

def truncate_encoder(encoder, n_blocks):
    layer_names = [
        [],
        ['layer1', 'stages_1'],
        ['layer2', 'stages_2'],
        ['layer3', 'stages_3'],
        ['layer4', 'stages_4'],
        [], # just for indexing convenience
    ]
    encoder_output = []
    for name, child in encoder.named_children():
        if name in layer_names[n_blocks]:
            break
        encoder_output.append(child)
    encoder_output = nn.Sequential(*encoder_output)
    return encoder_output
    
from src.models.decoder import DeepLabHeadV3Plus
class TimmSegModel(nn.Module):
    def __init__(self,
            backbone,
            use_timm_3d=False,
            architecture='unet',
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            n_classes=1,
            n_blocks=4,
            final_activation=None,
            decoder_channels=[256, 128, 64, 32, 16],
            decoder_attention_type: Optional[str] = None,
            dann = False,
            dann_scheduler_cfg = {},
            relu2gelu = False,
            input_upsample = 1.0,
            w_offset_head = False,
            output_stride = None,
            aspp_dilation = [6, 12, 18],
        ):
        super(TimmSegModel, self).__init__()
        self.n_blocks = n_blocks

        if use_timm_3d:
            self.encoder = timm_3d.create_model(
                backbone,
                in_chans=1,
                features_only=True,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                pretrained=pretrained,
                output_stride=output_stride,
            )
        else:
            self.encoder = timm.create_model(
                backbone,
                in_chans=1,
                features_only=True,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                pretrained=pretrained,
                output_stride=output_stride,
            )
            g = self.encoder(torch.rand(1, 1, 128, 128))
            #[print(_.shape) for _ in g]
            if 'inception' in backbone:
                self.encoder = convert_3d_inception(self.encoder)
            else:
                self.encoder = convert_3d(self.encoder)
        g = self.encoder(torch.rand(1, 1, 128, 128, 128))
        pre_scale = input_upsample
        post_scale = 1 / input_upsample
        # because convnexts has stride 4 at the first feature map
        if 'convnext' in backbone or 'inception' in backbone:
            post_scale *= 2
        #if self.n_blocks < len(g):
        #    self.encoder = truncate_encoder(self.encoder, self.n_blocks)
        #[print(_.shape) for _ in g]
        encoder_channels = [1] + [_.shape[1] for _ in g]
        if architecture == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type=decoder_attention_type,
            )
            self.decoder = convert_3d(self.decoder, relu2gelu)
        elif architecture == 'deeplabv3+':
            self.decoder = DeepLabHeadV3Plus(
                in_channels=encoder_channels[n_blocks],
                #out_channels=decoder_channels[n_blocks-1],
                aspp_dilate=aspp_dilation,
                num_classes=n_classes,
                highres_features=decoder_channels[2],
                out_features=decoder_channels[n_blocks-1],
                #output_stride=16,
                low_level_channels = encoder_channels[2],
                #attention_type=decoder_attention_type,
            )
            self.decoder = convert_3d(self.decoder, relu2gelu)
            post_scale *= 4
        else:
            raise ValueError(f'Unknown segtype: {architecture}')
        
        #print('pre_scale: ', pre_scale)
        #print('post_scale: ', post_scale)
        if post_scale != 1:
            self.upsample = partial(F.interpolate, scale_factor=post_scale, mode='trilinear', align_corners=False)
        else:
            self.upsample = lambda x:x
        if pre_scale != 1:
            self.pre_scale = partial(F.interpolate, scale_factor=pre_scale, mode='trilinear', align_corners=False)
        else:
            self.pre_scale = None

        if dann:
            self.dann = DomainAdversarialHead(in_dim = encoder_channels[self.n_blocks], dann_scheduler_cfg = dann_scheduler_cfg, dims=[128], n_domains=1)
        else:
            self.dann = None

        if w_offset_head:
            self.offset_head = nn.Conv3d(decoder_channels[n_blocks-1], 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        else:
            self.offset_head = None

        if final_activation is None:
            self.segmentation_head = nn.Conv3d(decoder_channels[n_blocks-1], n_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        elif final_activation == 'sigmoid':
            self.segmentation_head = nn.Sequential(
                nn.Conv3d(decoder_channels[n_blocks-1], n_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.Sigmoid()
            )
        elif final_activation == 'softmax':
            self.segmentation_head = nn.Sequential(
                nn.Conv3d(decoder_channels[n_blocks-1], n_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f'Unknown activation: {final_activation}')
    def forward(self,x):
        if self.dann is not None:
            x, alpha = x
        if self.pre_scale is not None:
            x = self.pre_scale(x)
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_map = self.segmentation_head(seg_features)
        seg_map = self.upsample(seg_map)
        if self.dann is not None:
            dann_output = self.dann(
                global_features[-1].mean((2, 3, 4)),# pool over spatial dimensions
                alpha)
            return seg_map, dann_output
        if self.offset_head is not None:
            offset_map = self.upsample(self.offset_head(seg_features))
            return seg_map, offset_map
        return seg_map
