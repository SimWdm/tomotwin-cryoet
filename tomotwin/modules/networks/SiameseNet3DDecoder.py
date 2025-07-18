from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomotwin.modules.networks.torchmodel import TorchModel

#%%
# maps number of channels to spatial dimensions in TomoTwin SiameseNet3Damese architecture, this info is needed for the decoder
NUM_CHANS_TO_SPAT_DIM_37 = {
    64: 18,  # example: if inpput spatial fimensions are 37x37x37, features after first encoder block have 64 channels and spatial dimensions of 18x18x18
    128: 14, 
    256: 10,
    512: 6,
    1024: 2,
}



class ConvLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, activation=nn.LeakyReLU()) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_chans, out_chans, eps=1e-05, affine=True),
            activation,
        )
    
    def forward(self, x):
        return self.layers(x)


class SpatialUpsampling(nn.Module):
    def __init__(self, out_size, in_chans: int, out_chans: int, activation=nn.LeakyReLU()) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(size=out_size, mode='trilinear'),
            ConvLayer(in_chans, out_chans, activation=activation),
        )
    
    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_size, in_chans, out_chans):
        super().__init__()
        self.out_size = out_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upsampler = SpatialUpsampling(
            out_size=out_size,
            in_chans=in_chans,
            out_chans=out_chans,
        )
        self.conv_layer0 = ConvLayer(2*out_chans, out_chans)
        self.conv_layer1 = ConvLayer(out_chans, out_chans)
    
    
    def forward(self, x, cat):  
        x = self.upsampler(x)
        x = torch.concat((x, cat), dim=1)
        x = self.conv_layer0(x)
        x = self.conv_layer1(x)
        return x

class SiameseNet3DDecoder(nn.Module):
    def __init__(self, out_size=37, out_chans=1, final_activation=nn.Identity(), encoder_cat_layer_ids=[0, 1, 2, 3]):
        super().__init__()
        self.out_size = out_size
        self.out_chans = out_chans
        self.final_activation = final_activation
        self.encoder_cat_layer_ids = encoder_cat_layer_ids
        self.decoder_blocks = nn.ModuleList()
        
        num_chans_list = [512, 256, 128, 64]
        num_chans_list = [1024] + [num_chans_list[i] for i in encoder_cat_layer_ids]  # 1024 is alays used as this is the decoder input size
        for k in range(len(num_chans_list)-1):
            decoder_block = DecoderBlock(
                out_size=NUM_CHANS_TO_SPAT_DIM_37[num_chans_list[k+1]],
                in_chans=num_chans_list[k],
                out_chans=num_chans_list[k+1],
            )
            self.decoder_blocks.append(decoder_block)

        self.final_upsampler = SpatialUpsampling(
            out_size=out_size,
            in_chans=num_chans_list[-1],
            out_chans=out_chans,
            activation=final_activation,
        )

    def forward(self, x, cats):
        cats = [cat for i, cat in enumerate(cats) if i in self.encoder_cat_layer_ids]
        assert len(cats) == len(self.decoder_blocks), "Number of cat elements (skip connection) must match number of decoder blocks"
        for decoder_block, cat in zip(self.decoder_blocks, cats):
            x = decoder_block(x, cat)
        x = self.final_upsampler(x)
        return x