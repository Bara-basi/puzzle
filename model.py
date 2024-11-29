import torch.nn as nn
import torch
from torchvision import models
from config import *

class PuzzleNet(nn.Module):
    def __init__(self):
        super(PuzzleNet, self).__init__()

        self.feature_extractor = models.__dict__[BACKBONE](pretrained=True)
        self.feature_extractor.fc = nn.Identity() # 将全连接层去掉

        self.pos_embedding = nn.Parameter(torch.randn(9,EMBED_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_TRANSFORMER_LAYERS)

        self.prediction_head = nn.Sequential(
            nn.Linear(EMBED_DIM,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,9)
        )
    
    def forward(self, x):
        # print(x.shape)
        batch_size = x.size(0)
        block_features = []
        for i in range(9):
            block = x[:,i]
            features = self.feature_extractor(block)
            block_features.append(features)
        
        block_features = torch.stack(block_features,dim=1)

        block_features = block_features + self.pos_embedding.unsqueeze(0)

        transformed = self.transformer(block_features)
        outputs = []
        for i in range(9):
            block_prediction = self.prediction_head(transformed[:,i])
            outputs.append(block_prediction)

        return torch.stack(outputs,dim=1)
    
if __name__ == '__main__':
    model = PuzzleNet()
    x = torch.randn(4,9,3,255,255)
    y = model(x)
    print(y.shape)