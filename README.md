# Pix2pix for Aerial-Map dataset

Implementation of pix2pix methods to pass Aerial images to maps, the algorithms that we implemented are:

* Unet [supervised]
* GAN [unsupervised]
* L-GAN [supervised]
* CycleGAN [supervised]
* L-CycleGAN [unsupervised]

## Setup
To install all required dependencies run:
```bash
pip install -r requirements.txt
```

## Running
To start training and get output with:
```bash
python train.py -m=<model> -e=<epochs> -d=<dataset>
```
Possible values for parameter `model` are: `unet`,`gan`,`lgan`,`cyclegan`,and `lcyclegan`.

Possible values for parameter `d` are: `a2m` and `inria`

`epochs` is the number of epoches that you want to run

All the output images are saved in the `/output` directory

You can use tensorboard in the `/logs` directory to visualize the loss 
