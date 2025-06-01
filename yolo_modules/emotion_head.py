# yolo_modules/emotion_head.py
import torch
import torch.nn as nn

class EmotionAbnHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_emotion: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc_emotion = nn.Linear(128, num_emotion)
        self.fc_abn = nn.Linear(128, 1)
    
    def forward(self, roi_feats):
        x = roi_feats.mean(dim=[2, 3])
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        emo_logit = self.fc_emotion(x)
        abn_score = self.fc_abn(x).squeeze(1)
        return emo_logit, abn_score