import argparse
from random import sample
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import utils
import os

from non_leaking import augment
from tun_model3 import Encoder, Generator, Classifier
from tun_model3 import Model as S2FGAN
from tun_dset2 import CeleDataset, CustomDataset
from tun_train2 import sample_data

def generate(loader, g_ema, device):
    img = next(loader)

    encoder = Encoder(3, 64, 6).to(device)

    with torch.no_grad():
        samples = None
        num =0

        img_latent = encoder((img[0].view(1,3,256,256)))
        k = g_ema(img_latent)
        
        print('k :', k)
        
        sample = torch.cat((img[0], k.view(3,256,256)), -1).unsqueeze(0)
        samples = sample if samples is None else torch.cat((samples,sample),-2)
        num+=1

        utils.save_image(
            samples,
            f"sample/{str(num).zfill(6)}.png",
            nrow= 16,
            normalize=True,
            range=(-1, 1),
            )


    

if __name__=="__main__":
    device = "cuda:0"

    parser = argparse.ArgumentParser()

    parser.add_argument(
                        "--ckpt", 
                        type=str,
                        default=None,
                        help="path to the checkpoints to resume training",
                        )

    parser.add_argument(
                        "--selected_attrs",
                        type = list,
                        nargs="+",
                        help="selected attributes for the CelebAMask-HQ dataset",
                        default=["Smiling", "Male","No_Beard", "Eyeglasses","Young", "Bangs", "Narrow_Eyes", "Pale_Skin", "Big_Lips","Big_Nose","Mustache","Chubby"],
                        )

    parser.add_argument(
                        "--TORCH_HOME", 
                        type=str, 
                        default="None", 
                        help="where to load/save pytorch pretrained models"
                        )

    parser.add_argument(
                        "--ATMDTT",
                        type = list,
                        nargs="+",
                        help="Attributes to manipulate during testing time",
                        default= 
                        [[1,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0]
                         ]
                        )
    
    parser.add_argument(
                        "--model_type", 
                        type = int, 
                        default = 0, 
                        help = "0- S2F-DIS, 1- S2F-DEC"
                        )

    parser.add_argument(
                        "--d_reg_every",
                        type=int,
                        default=16,
                        help="interval of the applying r1 regularization",
                        )
    
    parser.add_argument(
                        "--lr", 
                        type=float, 
                        default=0.002, 
                        help="learning rate"
                        )

    parser.add_argument(
                        "--augment", 
                        type=bool, 
                        default=True, 
                        help="apply discriminator augmentation"
                        )

    parser.add_argument(
                        "--label_path", 
                        type = str, 
                        default = "data/CelebAMask-HQ-attribute-anno.txt", 
                        help = "attributes annotation text file of CelebAMask-HQ"
                        )
    
    parser.add_argument(
                        "--imageZip", 
                        type=str, 
                        default= "data/CelebAMask-HQ-Sample.zip"
                        )
    
    parser.add_argument(
                        "--imagePath",
                        type=str, 
                        default= "CelebAMask-HQ-Sample/CelebA-HQ-img"
                        )

    parser.add_argument(
                        "--NumOfImage", 
                        type=int, 
                        default= 10, 
                        help = "number of images in the zip"
                        )
    
    parser.add_argument(
                        "--img_height", 
                        type=int, 
                        default=256, 
                        help="size of image height"
                        )
    
    parser.add_argument(
                        "--img_width", 
                        type=int, 
                        default=256, 
                        help="size of image width"
                        )

    parser.add_argument(
                        "--img_folder", 
                        type=str,
                        default="test_image", 
                        help="image folder"
                        )

    parser.add_argument(
                        "--img_path",
                        type=str,
                        default="3.jpg"
                        )

    args = parser.parse_args()

    c_dim = len(args.selected_attrs)

    #Set TORCH_HOME to system enviroment.
    if args.TORCH_HOME != "None":
        os.environ['TORCH_HOME'] = args.TORCH_HOME


    #Sanity check of GPU installation
    if not torch.cuda.is_available():
        raise SystemExit("GPU Required")

    model     = S2FGAN(args,c_dim,augment)
    ckpt = torch.load(args.ckpt)
    g_ema = Generator(c_dim).to(device)

    g_ema.load_state_dict(ckpt["g_ema"])

    # dataset_val = CeleDataset(args, False)
    test_data = CustomDataset(args, args.img_path, args.img_folder)
    dataloader_val = torch.utils.data.DataLoader(
        test_data, 
        batch_size=len(args.ATMDTT) + 2,
        num_workers=4
    )

    loader = sample_data(dataloader_val, device)

    
    generate(loader, g_ema, device)