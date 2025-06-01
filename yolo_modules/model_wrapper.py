# yolo_modules/model_wrapper.py
import torch
import torch.nn as nn
import sys
import os

# 添加 YOLOv9 到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
yolov9_path = os.path.join(project_root, 'code', 'yolov9')
sys.path.insert(0, yolov9_path)

from models.yolo import Model as Yolov9  # YOLOv9 官方的导入路径

# 导入自定义模块
from .grl import DomainClassifier
from .gat_head import GATAUHead, FaceRoIExtractor
from .emotion_head import EmotionAbnHead

class Y9_GAT_DA(nn.Module):
    def __init__(self, y9_cfg='yolov9s-face.yaml', gat_adj=None,
                 num_au=32, num_emotion=6):
        super().__init__()
        
        # 构建 YOLOv9 配置文件的完整路径
        cfg_path = os.path.join(yolov9_path, 'models', y9_cfg)
        
        self.detector = Yolov9(cfg=cfg_path, ch=3, nc=1)  # nc=1 for face only
        self.roi_extractor = FaceRoIExtractor(spatial_scale=1/8)
        self.gat_head = GATAUHead(adj_mat=gat_adj, num_au=num_au)
        self.emotion_head = EmotionAbnHead(num_emotion=num_emotion)
        self.domain_cls = DomainClassifier()
    
    @torch.no_grad()
    def _get_face_boxes(self, det_out, conf_thres=0.25):
        all_boxes, all_idx = [], []
        for b, det in enumerate(det_out):
            if det is None or len(det) == 0:
                continue
            det = det[det[:, 4] > conf_thres]
            boxes = det[:, :4]
            all_boxes.append(boxes)
            all_idx.append(torch.full((boxes.size(0),), b, 
                          device=boxes.device, dtype=torch.long))
        if len(all_boxes) == 0:
            return None, None
        return torch.cat(all_boxes, 0), torch.cat(all_idx, 0)
    
    def forward(self, images):
        # 注意：需要修改 YOLOv9 源码以返回中间特征
        det_preds, neck_feats = self.detector(images, return_feat=True)
        boxes, idx = self._get_face_boxes(det_preds)
        if boxes is None:
            return det_preds, None, None, None, None
        
        roi_feats = self.roi_extractor(neck_feats['P3'], boxes, idx)
        au_logit = self.gat_head(roi_feats)
        emo_logit, abn_score = self.emotion_head(roi_feats)
        
        global_feat = neck_feats['P3'].mean([2, 3])
        dom_logit = self.domain_cls(global_feat)
        
        return det_preds, au_logit, emo_logit, abn_score, dom_logit