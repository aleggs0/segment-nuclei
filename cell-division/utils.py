import torch
import torchvision
from dataloader import DivisionsData
from torch.utils.data import DataLoader
from tqdm import tqdm

def save_checkpoint(state,filename="my_checkpoint.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_divlbl_dir,
        val_dir,
        val_divlbl_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=1,
        pin_memory=True,
):
    train_ds = DivisionsData(
        img_dir=train_dir,
        div_label_dir=train_divlbl_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,        
    )

    val_ds = DivisionsData(
        img_dir=val_dir,
        div_label_dir=val_divlbl_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader,val_loader

def check_accuracy(dataloader,model,loss_fn,device="cuda",p_threshold=0.1):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    running_loss=0.
    model.eval()
    with torch.no_grad():
        dataloader_tqdm = tqdm(dataloader)
        for batch_idx, (X, Y) in enumerate(dataloader_tqdm):
            X = X.to(device=device).float()
            Y = Y.to(device=device).float()
            preds=model(X)
            running_loss += loss_fn(preds, Y)
            preds = torch.sigmoid(preds)
            preds=(preds>p_threshold)
            Y = (Y>p_threshold)
            num_correct+=(preds==Y).sum()
            num_pixels+=torch.numel(preds)
            dice_score += (2*(preds*Y).sum()) / ((preds+Y).sum() + 1e-8)
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(dataloader)}")
    last_loss = running_loss / (batch_idx+1)
    print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
    model.train()
    pass