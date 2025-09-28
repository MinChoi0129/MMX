# Multimodal-XAD
This is the official implementation of Multimodal-XAD.

## Setting：
* Clone this repo and prepare the environment.
```
git clone https://github.com/lab-sun/Multimodal-XAD.git
cd Multimodal-XAD

conda create -n multimodal_xad python=3.11
conda activate multimodal_xad

nvcc -V
conda install -c conda-forge pyquaternion nuscenes-devkit efficientnet_pytorch
nvcc -V
conda install -c conda-forge cuda-version=12.8.* cuda-nvcc=12.8.* cuda-cudart=12.8.*
nvcc -V

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
nvcc -V

python
import torch; print(torch.cuda.is_available()); print(torch.__version__)

```

## Dataset
To download the nu-A2D dataset, please refers to: https://drive.google.com/drive/folders/15zQhXqRjs-KmZyCuxsfGbHiaLdpQbHst?usp=drive_link

Download all the files and then extract them into the folder of `project_root/data/trainval/`


## Usage
* Clone this repo and prepare the environment.
* Download the dataset, create the foler `data/trainval/` in the project root, and release the dataset into the `/trainval/`.
* To pretrain/train the network, use the pre_train.py/train.py.
* To obatin the prediction results: 1) download the weight and put into the folder of `weight`; 2) use the predict.py.
* The link for the weight is: https://drive.google.com/file/d/1CFvBUTZ_EL0c3NT6JJrEJPyLaLMjCAur/view?usp=drive_link

## Citation
If you found this code or dataset are useful in your research, please consider citing
```
...
```
If you have any questions, pleas feel free to contact us!

Contact: yx.sun@cityu.edu.hk; yuchao.feng@connect.polyu.hk

Website: https://yuxiangsun.github.io/
