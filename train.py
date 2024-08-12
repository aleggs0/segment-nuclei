##code adapted from Aladdin Persson
import numpy as np
import torch
#import torchio
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy
)
from transforms import (
    train_transform,
    val_transform
)

from model_division import CPnet

from utils import get_loaders
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128 #128, choosing larger batch since we run on 4 gpus. but 256 gets slow
NUM_EPOCHS = 20 #100
NUM_WORKERS = 4 #4
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/img"
TRAIN_MASK_DIR = "data/div_lbl"
VAL_IMG_DIR = "data/val_img"
VAL_MASK_DIR = "data/val_div_lbl"

def train_loop(dataloader, model, loss_fn, optimizer, scaler):
    running_loss = 0.
    last_loss = 0.
    model.train()
    dataloader_tqdm = tqdm(dataloader)
    for batch_idx, (X, Y) in enumerate(dataloader_tqdm):
        # Compute prediction and loss
        X = X.to(device=DEVICE).float()
        Y = Y.to(device=DEVICE).float()

        #assert X.type == torch.float32
        with torch.autocast(DEVICE):
            pred = model(X)
            loss = loss_fn(pred, Y)
        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update dataloader_tqdm
        dataloader_tqdm.set_postfix(loss=loss.item())

        #reporting
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            last_loss = running_loss / 10
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            running_loss = 0.
    pass   

def loss_fn(Y_pred, Y_lbl): #,mask
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #Y_lbl = (Y_lbl > 2).float()
    return criterion(Y_pred, Y_lbl)

def main():
    print(f"Using {DEVICE} device")
    model = CPnet([1,32,64,128,256], 1, 3, max_pool=True, conv_3D=False)
    model= nn.DataParallel(model).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.tar"), model)

    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.GradScaler(DEVICE)
    
    check_accuracy(val_loader, model, loss_fn, device=DEVICE)
    for iepoch in range(NUM_EPOCHS):
        print(f"Epoch {iepoch+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, scaler)
        
        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,"my_checkpoint.tar")

        check_accuracy(val_loader, model, loss_fn, device=DEVICE)
    print("Done!")

if __name__=="__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time elapsed: {end_time-start_time:.2f} seconds")
    
    """import cProfile
    import pstats
    from pstats import SortKey
    cProfile.run('main()', 'train_profile_stats')
    p = pstats.Stats('train_profile_stats')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(100)"""