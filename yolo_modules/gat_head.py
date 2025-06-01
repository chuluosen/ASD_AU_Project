# yolo_modules/gat_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATAUHead(nn.Module):
    def __init__(self, in_channels: int = 256, hidden: int = 256,
                 num_au: int = 32, num_heads: int = 4,
                 adj_mat: torch.Tensor = None):
        super().__init__()
        self.register_buffer("edge_index", self._adj_to_edge_index(adj_mat))
        self.gat = GATConv(in_channels, hidden // num_heads,
                          heads=num_heads, concat=True, dropout=0.2)
        self.classifier = nn.Linear(hidden, num_au)
    
    def _adj_to_edge_index(self, adj):
        src, dst = torch.nonzero(adj, as_tuple=True)
        edge_index = torch.stack([torch.cat([src, dst]),
                                 torch.cat([dst, src])], dim=0)
        return edge_index
    
    def forward(self, roi_feats):
        x = roi_feats.mean(dim=[2, 3])
        x = self.gat(x, self.edge_index)
        logit = self.classifier(F.silu(x))
        return logit

# 添加 RoI 提取器到同一文件
class FaceRoIExtractor(nn.Module):
    def __init__(self, spatial_scale: float, feat_dim: int = 256,
                 output_size: int = 128, sampling_ratio: int = 2):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = spatial_scale
        self.reduction = nn.Conv2d(feat_dim, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, feat_map, bboxes, batch_idx):
        from torchvision.ops import roi_align
        rois = torch.cat([batch_idx.unsqueeze(1), bboxes], dim=1)
        feat_map = self.act(self.bn(self.reduction(feat_map)))
        aligned = roi_align(feat_map, rois,
                           output_size=self.output_size,
                           spatial_scale=self.spatial_scale,
                           sampling_ratio=self.sampling_ratio,
                           aligned=True)
        return aligned