# Multimodal-XAD
This is the official implementation of Multimodal-XAD.

## Environment setting：
* Python 3.6 or higher
* Pytorch 1.10 or higher
* torchvision 0.13.1 or higher
* numpy 1.19.0 or higher
* Install the NuScenes devkit from https://github.com/nutonomy/nuscenes-devkit
* see details in requirment.txt

## Dataset
To download the nu-A2D dataset, please refers to: http://

Download all the compressed file and then extract them in the folder of `project_root/data/trainval/`


## Usage
* Clone this repo and prepare the environment.
```
git clone https://github.com/lab-sun/Multimodal-XAD.git
```

* Download the dataset, create the foler `data/trainval/` in the project root, and release the dataset into the `/trainval/`.
* To pretrain/train the network, use the pretrain.py/train.py.
* To obatin the prediction results: 1) download the pretrained weight and put into the file of `weight`; 2) use the predict.py.
* The link for the weight is: 

## Citation
If you found this code or dataset are useful in your research, please consider citing
```
...
```
If you have any questions, pleas feel free to contact us!

Contact: yx.sun@cityu.edu.hk; yuchao.feng@connect.polyu.hk

Website: https://yuxiangsun.github.io/
