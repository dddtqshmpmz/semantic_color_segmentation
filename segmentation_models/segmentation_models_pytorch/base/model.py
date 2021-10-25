from math import fabs
import torch
from . import initialization as init
from .mbv2_ca import InvertedResidual,conv_3x3_bn
import torch.nn as nn


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationModelWithColor(torch.nn.Module):
    # def __init__(self) -> None:
    #     super().__init__()

    #     input_channel = 3+7
    #     block = InvertedResidual
    #     layers = []
        
    #     output_channel = input_channel
    #     layers.append(block( input_channel, output_channel, 1, 1))
    #     self.attention_ca =  nn.Sequential(*layers)


    def initialize(self):
        # init.initialize_attention(self.attention_ca)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x, processed_alpha_layers): # processed_alpha_layers
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # x = self.attention_ca(x)

        features = self.encoder(x)
        features[0] = torch.cat([features[0], processed_alpha_layers[:,:,:,:]  ],dim=1)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x,processed_alpha_layers):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x, processed_alpha_layers) # processed_alpha_layers

        return x