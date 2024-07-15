# Multimodal-XAD
This is the official implementation of Multimodal-XAD.

## Environment setting：
* Python 3.6 or higher
* Pytorch 1.10 or higher
* torchvision 0.13.1 or higher
* numpy 1.19.0 or higher
* see details in requirment.txt

## Dataset
To download the nu-A2D dataset, please refers to: http://

Download all the compressed file and then extract them in the folder of `project_root/data/trainval/`


## Usage
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
