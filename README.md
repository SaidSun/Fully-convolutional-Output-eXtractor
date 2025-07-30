# Fully-Convolutional-Output-Extractor(FOX)

## Description

Fully-Convolutional-Output-Extractor is a model for transform 5 channel images (RGBDE) into new 3 channel tensor with preservation of the structural and contrast features of the original objects. It may be useful for systems with multispectral camera to detecting or classifying objects.

The structure of this model is based on U-Net architecture with 5 level encoder and decoder parts. The general steps that were taken to reduce common requirements of RAM:
- each encoder convolution layer divided into a Depthwise and Pointwize convolutions parts;
- skip connection technology on levels were supplemented by Channel Attention level.

To avoid a fully mixturing unique information from input channels basic decoter turning into 3 independent modules with specific layer weights and activation functions.

The architecture of FOX model presented in Figure 1.

<div align="center">
  <img src="https://github.com/SaidSun/Fully-convolutional-Output-eXtractor/raw/main/images/Arch.jpg" alt="Альтернативный текст" width="80%">
  <p><em>Figure 1. FOX architecture</em></p>
</div>

