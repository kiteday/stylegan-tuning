import argparse
from random import sample
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import utils
import os

from non_leaking import augment
from tuning_model import Generator
from tuning_model import Model as S2FGAN
from tuning_dataset import CeleDataset
from tuning_train import sample_data

def generate(loader, model):
    img, label = next(loader)

    with torch.no_grad():
        samples = None
        for j,l in zip(img,torch.cat((LABELS,label[-2:]))): 
            l  =  l.view(1,12).repeat(13,1) * SCALE
            k = model(j.view(1,3,256,256),sampled_ratio = l,generate = True) 
            im = torch.cat([x for x in k],-1) 
            sample = torch.cat((j,k.view(3,256,256),im),-1).unsqueeze(0)
            samples = sample if samples is None else torch.cat((samples,sample),-2)
            
        utils.save_image(
            samples,
            f"sample/{str(label).zfill(6)}.png",
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
    model.load_state_dict(ckpt["model"])
    model.to(device)

    g_optim = model.g_optim
    d_optim = model.d_optim

    model = nn.DataParallel(model, [0])

    dataset_val = CeleDataset(args, False)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=len(args.ATMDTT) + 2,
        num_workers=4
    )

    loader = sample_data(dataloader_val, device)

    LABELS = torch.FloatTensor(args.ATMDTT).to(device)
    # SCALE =  torch.FloatTensor([-4.0,-3.0, -2.0,-1.5, -1.0, -0.5,0,0.5, 1.0,1.5,2.0,3.0,4.0]).to(device).view(13,1)

    SCALE =  torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device).view(13,1)

    generate(loader, model)
