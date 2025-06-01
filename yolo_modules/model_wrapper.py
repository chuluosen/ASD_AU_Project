# yolo_modules/model_wrapper.py
import torch
import torch.nn as nn
import sys
import os
import warnings

# 添加 YOLOv9 到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # yolo_modules 目录
project_root = os.path.dirname(current_dir)  # ASD_AU_Project 目录 (项目根目录)
yolov9_path = os.path.join(project_root, 'code', 'yolov9')  # 指向您的 YOLOv9 submodule
sys.path.insert(0, yolov9_path)  # 将 YOLOv9 submodule 路径添加到模块搜索路径的最前面

# 尝试从YOLOv9的工具中导入LOGGER，如果失败则使用Python标准logging
try:
    from utils.general import LOGGER
except ImportError:
    import logging
    LOGGER = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    LOGGER.warning("YOLOv9's native LOGGER not found, using standard Python logging.")

from models.yolo import Model as Yolov9  # 从已添加到路径的 YOLOv9 submodule 中导入

# 处理相对导入
try:
    # 作为包导入时使用相对导入
    from .grl import DomainClassifier
    from .gat_head import GATAUHead, FaceRoIExtractor
    from .emotion_head import EmotionAbnHead
except ImportError:
    # 直接运行文件时使用绝对导入
    from grl import DomainClassifier
    from gat_head import GATAUHead, FaceRoIExtractor
    from emotion_head import EmotionAbnHead


class Y9_GAT_DA(nn.Module):
    def __init__(self, 
                 y9_cfg='yolov9s-face.yaml', 
                 gat_adj=None,
                 num_au=32, 
                 num_emotion=6,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 p3_stride=8,  # 允许用户指定P3的stride
                 conf_thres=0.25):  # 置信度阈值作为可配置参数
        """
        YOLOv9 + GAT + Domain Adaptation 模型
        
        Args:
            y9_cfg: YOLOv9配置文件名
            gat_adj: GAT邻接矩阵，可以是None
            num_au: AU类别数量
            num_emotion: 情绪类别数量
            device: 运行设备
            p3_stride: P3特征的stride，默认为8
            conf_thres: 人脸检测置信度阈值
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.p3_stride = p3_stride
        
        # 构建 YOLOv9 配置文件的完整路径
        cfg_path = os.path.join(yolov9_path, 'models', y9_cfg)
        
        # 验证配置文件是否存在
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"YOLOv9 config file not found: {cfg_path}")
        
        # 初始化检测器
        self.detector = Yolov9(cfg=cfg_path, ch=3, nc=1)  # nc=1 for face only
        LOGGER.info(f"Loaded YOLOv9 detector from {cfg_path}")
        
        # 初始化其他组件
        self.roi_extractor = FaceRoIExtractor(spatial_scale=1/self.p3_stride)
        
        # 处理GAT邻接矩阵
        if gat_adj is not None:
            if isinstance(gat_adj, torch.Tensor):
                gat_adj = gat_adj.to(self.device)
            else:
                gat_adj = torch.tensor(gat_adj, device=self.device, dtype=torch.float32)
        
        self.gat_head = GATAUHead(adj_mat=gat_adj, num_au=num_au)
        self.emotion_head = EmotionAbnHead(num_emotion=num_emotion)
        self.domain_cls = DomainClassifier()
        
        # 将整个模型移动到指定设备
        self.to(self.device)
        
        LOGGER.info(f"Model initialized on device: {self.device}")
    
    @torch.no_grad()
    def _get_face_boxes(self, det_out, conf_thres=None):
        """
        从YOLOv9的检测输出中提取人脸框。
        
        Args:
            det_out: YOLOv9检测器的输出，列表格式，每个元素对应一张图像
            conf_thres: 置信度阈值，如果为None则使用self.conf_thres
            
        Returns:
            (all_boxes, all_idx) 或 (None, None)
            all_boxes: [Total_N, 4] 的张量，包含所有图像中筛选后的人脸框(xyxy)
            all_idx: [Total_N] 的张量，每个框对应的图像在批处理中的索引
        """
        if conf_thres is None:
            conf_thres = self.conf_thres
            
        all_boxes, all_idx = [], []
        
        # 确保det_out是列表格式
        if not isinstance(det_out, list):
            det_out = [det_out]
        
        for b, det_tensor in enumerate(det_out):
            if det_tensor is None or len(det_tensor) == 0:
                continue
            
            # 确保检测结果在正确的设备上
            if det_tensor.device != self.device:
                det_tensor = det_tensor.to(self.device)
            
            # 验证检测结果的格式
            if det_tensor.dim() != 2 or det_tensor.shape[1] < 5:
                LOGGER.warning(f"Unexpected detection tensor shape: {det_tensor.shape}")
                continue
            
            # 应用置信度筛选
            filtered_det = det_tensor[det_tensor[:, 4] > conf_thres]
            
            if filtered_det.shape[0] == 0:
                continue
                
            boxes = filtered_det[:, :4]  # 前4列是 xyxy 坐标
            all_boxes.append(boxes)
            all_idx.append(torch.full((boxes.size(0),), b, 
                                     device=self.device, dtype=torch.long))
        
        if len(all_boxes) == 0:
            return None, None
            
        return torch.cat(all_boxes, 0), torch.cat(all_idx, 0)
    
    def forward(self, images, return_intermediate=False):
        """
        前向传播
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            return_intermediate: 是否返回中间特征
            
        Returns:
            如果return_intermediate=False:
                (det_preds, au_logit, emo_logit, abn_score, dom_logit)
            如果return_intermediate=True:
                还会额外返回neck_feats字典
        """
        # 输入验证
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(images)}")
        
        if images.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {images.dim()}D")
        
        # 确保输入在正确的设备上
        images = images.to(self.device)
        
        # 运行检测器
        try:
            det_preds, neck_feats = self.detector(images, return_feat=True)
        except Exception as e:
            LOGGER.error(f"Error in YOLOv9 detector forward pass: {e}")
            raise
        
        # 检查neck特征
        if neck_feats is None:
            LOGGER.warning("neck_feats is None. YOLOv9 detector may not support return_feat=True")
            if return_intermediate:
                return det_preds, None, None, None, None, None
            return det_preds, None, None, None, None
            
        if 'P3' not in neck_feats:
            available_keys = list(neck_feats.keys())
            LOGGER.warning(f"P3 feature not found. Available keys: {available_keys}")
            if return_intermediate:
                return det_preds, None, None, None, None, neck_feats
            return det_preds, None, None, None, None
        
        # 提取人脸框
        boxes, idx = self._get_face_boxes(det_preds)
        
        if boxes is None:
            LOGGER.debug("No faces detected or passed confidence threshold.")
            if return_intermediate:
                return det_preds, None, None, None, None, neck_feats
            return det_preds, None, None, None, None
        
        # 使用P3特征进行后续处理
        p3_features = neck_feats['P3']
        
        # 确保特征在正确的设备上
        if p3_features.device != self.device:
            p3_features = p3_features.to(self.device)
        
        try:
            # ROI特征提取
            roi_feats = self.roi_extractor(p3_features, boxes, idx)
            
            # AU预测
            au_logit = self.gat_head(roi_feats)
            
            # 情绪和异常预测
            emo_logit, abn_score = self.emotion_head(roi_feats)
            
            # 域分类（使用全局特征）
            global_feat = p3_features.mean(dim=[2, 3])  # [BatchSize, Channels_P3]
            dom_logit = self.domain_cls(global_feat)
            
        except Exception as e:
            LOGGER.error(f"Error in downstream heads: {e}")
            raise
        
        if return_intermediate:
            return det_preds, au_logit, emo_logit, abn_score, dom_logit, neck_feats
        return det_preds, au_logit, emo_logit, abn_score, dom_logit
    
    def get_param_groups(self, detector_lr_mult=0.1):
        """
        获取参数组用于不同的学习率设置
        
        Args:
            detector_lr_mult: 检测器学习率倍数（相对于其他部分）
            
        Returns:
            参数组列表
        """
        detector_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'detector' in name:
                    detector_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr_mult': 1.0},
            {'params': detector_params, 'lr_mult': detector_lr_mult}
        ]
        
        LOGGER.info(f"Parameter groups created: {len(other_params)} other params, {len(detector_params)} detector params")
        
        return param_groups


# 测试代码
if __name__ == '__main__':
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型实例
    print("Creating model instance...")
    
    # 创建一个简单的邻接矩阵用于测试
    dummy_adj = torch.ones((32, 32))  # 假设的邻接矩阵
    
    try:
        model = Y9_GAT_DA(
            y9_cfg='yolov9s-face.yaml',
            gat_adj=dummy_adj,
            num_au=32,
            num_emotion=6,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Model created successfully on device: {model.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        print("\nTesting forward pass...")
        dummy_images = torch.randn(2, 3, 640, 640)  # batch_size=2, 输入 640x640
        
        with torch.no_grad():
            outputs = model(dummy_images)
            
        print("\nOutput shapes:")
        output_names = ['det_preds', 'au_logit', 'emo_logit', 'abn_score', 'dom_logit']
        for i, (name, out) in enumerate(zip(output_names, outputs)):
            if out is not None:
                if isinstance(out, list):
                    print(f"{name}: list of {len(out)} items")
                    if out:
                        print(f"  First item shape: {out[0].shape if hasattr(out[0], 'shape') else 'N/A'}")
                else:
                    print(f"{name}: {out.shape}")
            else:
                print(f"{name}: None")
                
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()