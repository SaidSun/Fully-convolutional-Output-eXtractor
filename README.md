# Fully-Convolutional-Output-Extractor(FOX)

## Description

Fully-Convolutional-Output-Extractor is a model for transform RGBD(Red Green Blue Depth Edge) images into new 3 channel tensor with preservation of the structural and contrast features of the original objects. It may be useful for systems with multispectral camera to detecting or classifying objects.

The structure of this model is based on U-Net architecture with 5 level encoder and decoder parts. The general steps that were taken to reduce common requirements of RAM:
- each encoder convolution layer divided into a Depthwise and Pointwize convolutions parts;
- skip connection technology on levels were supplemented by Channel Attention level.

To avoid a fully mixturing unique information from input channels basic decoder turning into 3 independent modules with specific layer weights and activation functions.

The architecture of FOX model presented in Figure 1.

<div align="center">
  <img src="https://github.com/SaidSun/Fully-convolutional-Output-eXtractor/raw/main/images/Arch.jpg" alt="Альтернативный текст" width="80%">
  <p><em>Figure 1. FOX architecture</em></p>
</div>

## Training and validating

At present we can't use a perfect metric to understand how good FOX model work. Although precision can be compute through combining with a model that produces the final result (detection, classification etc).
Our model current version trained with YOLOv8 model on dataset, which have 30 object classes distributed as show in Figure 2.

<div align="center">
  <img src="https://github.com/SaidSun/Fully-convolutional-Output-eXtractor/raw/main/images/classes.jpg" alt="Альтернативный текст" width="80%">
  <p><em>Figure 2. Object class distribution</em></p>
</div>

We use F1-score metric to understand how good our model is. The latest measured metric values are indicated in Figure 3.

<div align="center">
  <img src="https://github.com/SaidSun/Fully-convolutional-Output-eXtractor/raw/main/images/metrics.jpg" alt="Альтернативный текст" width="80%">
  <p><em>Figure 3. Metric value comparison</em></p>
</div>

## Result examples

There is some output examples.

<div align="center">
  <img src="https://github.com/SaidSun/Fully-convolutional-Output-eXtractor/raw/main/images/ex1.jpg" alt="Алтернативный текст" width="80%">
</div>

<div align="center">
  <img src="https://github.com/SaidSun/Fully-convolutional-Output-eXtractor/raw/main/images/ex2.jpg" alt="Альтернативный текст" wigth="60%">
</div>
