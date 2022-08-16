import argparse
# from random import sample
# import numpy as np
import torch
from torch import nn
# from torch.utils import data
from torchvision import utils
# from torchvision import transforms

import os
from io import BytesIO
from PIL import Image
from non_leaking import augment
from tun_model2 import Encoder, Generator, Classifier
from tun_model2 import Model as S2FGAN
from tuning_dataset import CustomDataset
from tun_train2 import accumulate, sample_data

def generate(loader, model):
    img = next(loader)
    # print(img)
    
    with torch.no_grad():
        samples = None
        num =0
           
        print("j :",img[0])

        # k는 generator img, im은 k 이미지 모음
        k = model(img=img[0].view(1,3,256,256),generate = True) 
        # im = im.view(3,256,256)
        # print('j :',img[0].view(1,3,256,256))
        # im = torch.cat([x for x in im],-1) #k.view(3,256,256)와 같음
        print('k :', k)
        # print('im :', im)
        # # k를 출력(256)에 맞게 reshape
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

        

# 이미지 경로를 주면 이미지 읽어들이기
def read_data(path, height = None, width = None):
    with open(path, 'rb') as f:
        data = f.read()
    data_io = BytesIO(data)

    image = Image.open(data_io)

    if height != None and width != None:
        image = image.resize((height,width))
        
    return image

    

if __name__=="__main__":
    device = "cuda:0"

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", 
                        type=str,
                        default=None,
                        help="path to the checkpoints to resume training",
                        )

    parser.add_argument("--selected_attrs",
                        type = list,
                        nargs="+",
                        help="selected attributes for the CelebAMask-HQ dataset",
                        default=["Smiling", "Male","No_Beard", "Eyeglasses","Young", "Bangs", "Narrow_Eyes", "Pale_Skin", "Big_Lips","Big_Nose","Mustache","Chubby"],
                        )

    parser.add_argument("--TORCH_HOME", 
                        type=str, 
                        default="None", 
                        help="where to load/save pytorch pretrained models"
                        )

    parser.add_argument("--ATMDTT",
                        type = list,
                        nargs="+",
                        help="Attributes to manipulate during testing time",
                        default= 
                        [[0,0,0,0,0,0,0,0,0,0,0,0]
                         ]
                        )
    
    parser.add_argument("--model_type", 
                        type = int, 
                        default = 0, 
                        help = "0- S2F-DIS, 1- S2F-DEC"
                        )

    parser.add_argument("--d_reg_every",
                        type=int,
                        default=16,
                        help="interval of the applying r1 regularization",
                        )
    
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.002, 
                        help="learning rate"
                        )

    parser.add_argument("--augment", 
                        type=bool, 
                        default=True, 
                        help="apply discriminator augmentation"
                        )
    
    parser.add_argument("--NumOfImage", 
                        type=int, 
                        default= 10, 
                        help = "number of images in the zip"
                        )
    
    parser.add_argument("--img_height", 
                        type=int, 
                        default=256, 
                        help="size of image height"
                        )
    
    parser.add_argument("--img_width", 
                        type=int, 
                        default=256, 
                        help="size of image width"
                        )

    parser.add_argument(
                        "--input_img_path", 
                        type=str,  
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
    model_ema = S2FGAN(args,c_dim,augment)

    ckpt = torch.load(args.ckpt)
    # print(model)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    model_ema.load_state_dict(ckpt["model_ema"])
    model_ema.to(device)

    accumulate(model_ema, model, 0)
    model_ema.eval()
   
    # g_optim = model.g_optim
    # d_optim = model.d_optim

    model = nn.DataParallel(model, [0])



    img = read_data(args.input_img_path)
 
    test_data = CustomDataset(args, img, args.input_img_path)
    dataloader_val = torch.utils.data.DataLoader(
        test_data, 
        batch_size=len(args.ATMDTT) + 2,
        num_workers=4
    )

    loader = sample_data(dataloader_val, device)

    LABELS = torch.FloatTensor(args.ATMDTT).to(device)
    # SCALE =  torch.FloatTensor([-4.0,-3.0, -2.0,-1.5, -1.0, -0.5,0,0.5, 1.0,1.5,2.0,3.0,4.0]).to(device).view(13,1)

    SCALE =  torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device).view(13,1)

    generate(loader, model)
