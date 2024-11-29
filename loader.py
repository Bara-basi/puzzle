import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 
from config import *
import random
import matplotlib.pyplot as plt
from agent import solvable


class PuzzleDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        img = cv2.imread(row['img_path'])
        if img.shape[0]!= IMAGE_SIZE or img.shape[1]!= IMAGE_SIZE:
            img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
        img = Image.fromarray(img)
        img_width,img_height = img.size
        block_size_w,block_size_h = img_width//BLOCK_NUM,img_height//BLOCK_NUM

        blocks = []
        for i in range(BLOCK_NUM):
            for j in range(BLOCK_NUM):
                block = img.crop((j*block_size_w,i*block_size_h,(j+1)*block_size_w,(i+1)*block_size_h))
                blocks.append(block)
        # 打乱顺序,blocks和positions的顺序一致相同
        index = list(range(BLOCK_NUM*BLOCK_NUM))
        random.shuffle(index)
        while not solvable(index):
            random.shuffle(index)
        blocks = [blocks[i] for i in index]
        if self.transform is not None:
            blocks = [self.transform(block) for block in blocks]
        else:
            blocks = [transforms.ToTensor()(block) for block in blocks]
        blocks = torch.stack(blocks,dim = 0)
        index = torch.tensor(index)
        return blocks,index
    
def get_dataloader(csv_file,batch_size,transform=None,shuffle=True,num_workers=4): # num_workers参数的作用是设置加载数据的进程数，默认为0，表示使用主进程加载数据
    dataset = PuzzleDataset(csv_file,transform)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    train_loader = get_dataloader(
        'imgs/train.csv',
        transform=transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = get_dataloader(
        'imgs/val.csv',
        transform=transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    