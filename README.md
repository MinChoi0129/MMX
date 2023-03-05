# Multimodal-XAD
This is the official implementation of Multimodal-XAD.

The codes and dataset will be released when the paper is published.

## Environment setting：
* Python 3.6 or higher
* Pytorch 1.10 or higher

## Dataset
Download the datasets and then extract it in the file of `Data`

To download the nu-A2D dataset, please refers to: http://

The encoder and BEV module of the Multimodal-XAD network is firstly pre-trained by using the nuScenes dataset.

To download the nuScenes data, you may refer to the offical site of nuScenes.

## Pretrained weights：
* Download the pretrained weights and then extract it in the file of `weight`
* The link for pretrained weights is: https: //

## Usage_Evaluation
* Clone this repo.
```
git clone 
```

* Download the dataset and put into the file of `Data`;
nu-A2D for the training of whole model
* Download the pretrained weight and put into the file of `weight`;

* To evluate the network, use the predict.py in the root folder of this project


## Usage_Train
* Clone this repo.
```
git clone 
```

* Download the dataset and put into the file of `Data`;

nuScenes data for the pre-training of the encoder and BEV Module
nu-A2D for the training of whole model

* Download the pretrained weight and put into the file of `weight` (optional);

* To train the network, use the train.py in the root folder of this project

## Citation
If you found this code or dataset are useful in your research, please consider citing
```
...
```
