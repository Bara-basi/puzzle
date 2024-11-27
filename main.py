import os
import cv2
import torch
import time
import random
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import PuzzleNet
from loader import get_dataloader
from evaluate import evaluate
from evaluate import calculate_consistency_loss
from config import *
from torch.nn.utils import clip_grad_norm_
from agent import PuzzleAgent,PuzzleEnv,solvable
import torchvision.transforms as transforms
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 逆归一化类
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # 逆归一化操作
        return tensor


def train(model,train_loader,test_loader,device):
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) 
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0

        for blocks,positions in train_loader:
            blocks = blocks.to(device)
            positions = positions.to(device)

            optimizer.zero_grad()

            predictions = model(blocks)
            position_loss = criterion(predictions.view(-1, 9), positions.view(-1))  
            consistency_loss = calculate_consistency_loss(predictions)
            loss = position_loss + 0.1*consistency_loss
            
            loss.backward()
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            total_loss += loss.item() * blocks.size(0)
            total_samples += blocks.size(0)
        
        train_loss = total_loss / total_samples

        val_position_loss,val_consistency_loss = evaluate(model,test_loader,device,epoch)
        val_loss = val_position_loss + 0.1*val_consistency_loss
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/puzzle_net_{epoch+1}.pth")
            print("Model saved!")

if __name__ == '__main__':
    # 如果模型不存在，则开始训练，否则，直接调用models/final_model.pth模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD)
    ])
    val_loader = get_dataloader(
        'imgs/val.csv',
        transform=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = PuzzleNet()
    model.to(device)
    print(f"Using device: {device}")
    if not os.path.exists("models/final_model.pth"):
        train_loader = get_dataloader(
            'imgs/train.csv',
            batch_size=BATCH_SIZE,
            shuffle=True,
        )       
        train(model,train_loader,val_loader,device)
    else:
        model.load_state_dict(torch.load("models/final_model.pth")) # 加载模型
        evaluate(model,val_loader,device,0) # 查看模型准确率
        for blocks,position in val_loader:
            model.eval()
            blocks = blocks.to(device)
            predictions = model(blocks)
            predictions = predictions.argmax(dim=2)
            for k in range(predictions.shape[0]):
                state = predictions[k].float().cpu().numpy()+1
                goal = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.float32)
                mark_value = state[8]
                state[8] = 0
                goal[goal==mark_value] = 0
                state = state.reshape(3,3)
                if not solvable(state):
                    print("Not solvable")
                    continue
                agent = PuzzleAgent()
                agent.load_Q('models/Q_label{}.csv'.format(mark_value))
                env = PuzzleEnv(state,goal)
                state_list = agent.solve(env)
                original_img = blocks[k].cpu().numpy() # [9,3,85,85]
                original_img[-1] = np.zeros((3,85,85)) # 填充空白块
                state_img = np.zeros((255,255,3),dtype=np.uint8)

                for i in range(3):
                    for j in range(3):
                        state_img[i*85:i*85+85,j*85:j*85+85] = original_img[i*3+j].transpose(1,2,0)*255
                cv2.imshow('original',state_img)
                for pre,cur in zip(state_list[:-1],state_list[1:]):
                    block = state_img[cur[0]*85:cur[0]*85+85,cur[1]*85:cur[1]*85+85,:]
                    state_img[pre[0]*85:pre[0]*85+85,pre[1]*85:pre[1]*85+85,:] = block
                    state_img[cur[0]*85:cur[0]*85+85,cur[1]*85:cur[1]*85+85,:] = np.zeros((85,85,3),dtype=np.uint8)
                cv2.imshow('result',state_img)
                print("press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
