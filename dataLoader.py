import pandas as pd
import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class belugaDataset(Dataset):
    def __init__(self, csv_file, im_folder, keep_ratio = False, size = (720, 480)):
        self.csv_file = csv_file
        self.im_folder = im_folder
        self.keep_ratio = keep_ratio
        self.df = pd.read_csv(csv_file)
        self.size = size


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d = dict(self.df.iloc[idx,:])
        img = cv2.imread(os.path.join(self.im_folder, d['image_id']+'.jpg'))
        date = d['date']
        viewpoint = d['viewpoint']
        y = int((d['whale_id'].split('whale'))[-1])
        x_0, y_0, _ = img.shape

        if x_0 >y_0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = cv2.resize(img, self.size)
        if self.keep_ratio:
            r_y, r_x = self.size[0], self.size[1] 
            pass
        
        return {'x':img, 'date':date, 'y':y, 'vp':viewpoint} 


if __name__ == "__main__":
    ds = belugaDataset(csv_file=os.path.join(os.pardir, 'metadata.csv'), im_folder=os.path.join(os.pardir, 'images'))
    print(ds[0])
