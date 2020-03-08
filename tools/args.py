import sys
import argparse



def create_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e','--epochs',type=int, default=50, help='The epochs')
    parser.add_argument('-m','--model', type=str, default="unet", choices=['unet','gan','lgan','cyclegan','lcyclegan'], help='The model you choose')
    parser.add_argument('-l','--load', type=bool, default=False, help = 'Weather you load your checkpoint or not')
    parser.add_argument('-d','--dataset',type=str, default='a2m', choices = ['a2m','inria'], help='Choose the dataset that you want to train')

    args = parser.parse_args()

    return args