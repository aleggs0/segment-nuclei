import numpy as np
import torch
#import torchio
from tqdm import tqdm
from utils import load_checkpoint
import torch.nn as nn
import torch.optim as optim
import torchvision
from model_division import CPnet
from dataloader import DivisionsData
from torch.utils.data import DataLoader
from transforms import val_transform

def save_predictions_as_imgs(
        loader,model,folder="val_pred/", device="cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        if idx >5 :
            continue
        x=x.to(device=device).float()
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #preds = (preds>0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.tif"
        )
        torchvision.utils.save_image(
            y.float(), f"{folder}/y_{idx}.tif"
        )
    model.train()

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = CPnet([1,32,64,128,256], 1, 3, max_pool=True, conv_3D=False) #make sure this matches
    model= nn.DataParallel(model).to(DEVICE)
    #model=model.to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.tar"), model)
    model.eval()
    val_ds = DivisionsData(
        img_dir="data/case_study", #val_img
        div_label_dir="data/val_div_lbl",
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1, #16
        num_workers=2,
        pin_memory = True,
        shuffle=False,
    )
    save_predictions_as_imgs(val_loader,model,folder="data/val_pred/", device=DEVICE)

if __name__=="__main__":
    main()