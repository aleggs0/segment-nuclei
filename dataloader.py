import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset

class DivisionsData(Dataset):
    def __init__(self, img_dir, div_label_dir, transform=None):
        self.img_dir = img_dir
        self.div_label_dir = div_label_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        self.div_labels = os.listdir(div_label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        div_label_path = os.path.join(self.div_label_dir, self.images[idx].replace("_img.tif", "_div.tif"))
        image = tiff.imread(img_path)
        label = tiff.imread(div_label_path)
        assert np.shape(image)==np.shape(label), 'img and label must be same size'
        if self.transform:
            image,label = self.transform(img=image, lbl=label)
        return image, label