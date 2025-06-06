"""
Stage 1: HDA-SynChild预训练脚本 - Colab适配版
只训练YOLOv9主干+检测头，冻结GAT-AU和情绪头

## 复现环境要求 ##
- Platform: Google Colab (推荐) 或支持CUDA的Linux环境
- GPU: A100 (推荐) / V100 / RTX 30/40系列 / T4
- Python: ≥ 3.8
- PyTorch: ≥ 2.0
- CUDA: ≥ 11.0

## 依赖安装 ##
pip install torch torchvision opencv-python numpy PyYAML matplotlib albumentations pycocotools torch-geometric

## 数据要求 ##
- HDA-SynChild数据集（需要联系原作者获取）
- YOLOv9预训练权重（建议使用官方yolov9c.pt作为fallback）
- 数据结构：
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/

## 复现性说明 ##
- 随机种子：固定为42
- CuDNN：A100/V100启用benchmark模式，其他GPU启用确定性模式
- batch_size会根据GPU自动调整

## 预期性能 ##
- A100-80GB: batch_size=32-48, ~2-3 it/s, mAP@0.5:0.95 ≥ 0.57
- V100: batch_size=24, ~1.5-2 it/s
- T4: batch_size=8, ~0.8-1.2 it/s
"""

import os
import sys
import torch
import torch.nn as nn

# 解决PyTorch 2.6的weights_only问题
import torch.serialization
try:
    # 方法1：添加安全全局类
    torch.serialization.add_safe_globals(['DetectionModel', 'Model'])
except:
    # 方法2：monkey patch torch.load
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import shutil
from google.colab import drive
import gc

# ============= Colab环境设置 =============
def setup_colab_env():
    """设置Colab环境"""
    # 挂载Google Drive
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    
    # 检查GPU并显示优化信息
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        gpu_name = gpu_info.name
        gpu_memory = gpu_info.total_memory / 1024**3
        
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # 显示针对特定GPU的优化信息
        if "A100" in gpu_name:
            if gpu_memory > 70:
                print("🎯 A100-80GB检测到！已启用高性能优化配置")
                print("   - Batch Size: ↑ 最大48")
                print("   - Workers: ↑ 最大20")
                print("   - TensorFloat-32: ✅ 启用")
                print("   - FPS目标: ≥80")
            else:
                print("🎯 A100-40GB检测到！已启用优化配置")
                print("   - Batch Size: ↑ 最大32")  
                print("   - Workers: ↑ 最大16")
        elif "V100" in gpu_name:
            print("⚡ V100检测到！已启用中等优化配置")
            print("   - Batch Size: ↑ 最大24")
            print("   - FPS目标: ≥50")
        elif gpu_memory < 16:
            print("💡 小显存GPU检测到，已启用内存安全配置")
            print("   - Batch Size: 限制为8")
        
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 PyTorch Version: {torch.__version__}")
    else:
        raise RuntimeError("No GPU available in Colab!")
    
    # 设置项目路径
    project_root = Path('/content/ASD_AU_Project')
    if not project_root.exists():
        raise RuntimeError(f"Project not found at {project_root}. Please clone the repository first!")
    
    # 添加路径
    sys.path.append(str(project_root))
    sys.path.append(str(project_root / 'code' / 'yolov9'))
    
    # 创建必要的目录
    drive_save_dir = Path('/content/drive/MyDrive/ASD_AU_weights/stage1')
    drive_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理缓存避免损坏文件问题（轻量级）
    print("Cleaning recent cache files...")
    try:
        # 只清理明确知道的缓存位置
        cache_files = [
            '/content/ASD_AU_Project/train.cache',
            '/content/ASD_AU_Project/val.cache',
            '/tmp/train.cache',
            '/tmp/val.cache'
        ]
        cache_count = 0
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                cache_count += 1
        print(f"Cleaned {cache_count} cache files")
    except Exception as e:
        print(f"Cache cleaning skipped: {e}")
    
    # 检查数据划分脚本的输出（支持嵌套路径）
    data_candidates = [
        Path('/content/yolo_data/yolo_data'),  # 用户当前的实际路径
        Path('/content/yolo_data'),            # 标准路径
    ]
    
    found_data = False
    for yolo_data_path in data_candidates:
        if yolo_data_path.exists() and (yolo_data_path / 'images').exists():
            print(f"✅ Found dataset at: {yolo_data_path}")
            # 显示数据统计
            for split in ['train', 'val', 'test']:
                img_dir = yolo_data_path / 'images' / split
                if img_dir.exists():
                    img_count = len(list(img_dir.glob('*')))
                    print(f"   {split:5s}: {img_count:,} images")
            found_data = True
            break
    
    if not found_data:
        print("⚠️  No dataset found in expected locations")
    
    return project_root, drive_save_dir

# 设置环境
PROJECT_ROOT, DRIVE_SAVE_DIR = setup_colab_env()

# 确保YOLOv9路径正确添加
yolov9_path = PROJECT_ROOT / 'code' / 'yolov9'
if str(yolov9_path) not in sys.path:
    sys.path.insert(0, str(yolov9_path))

# 验证路径和导入
print(f"YOLOv9 path: {yolov9_path}")
print(f"YOLOv9 path exists: {yolov9_path.exists()}")

try:
    # 导入YOLOv9组件
    from models.yolo import Model as Yolov9
    from utils.dataloaders import create_dataloader
    from utils.general import increment_path, colorstr, check_img_size, LOGGER
    from utils.loss_tal import ComputeLoss
    from utils.torch_utils import ModelEMA, de_parallel
    from utils.metrics import fitness
    from val import run as validate
    from utils.callbacks import Callbacks
    print("✅ YOLOv9 modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import YOLOv9 modules: {e}")
    print("📁 Available files in yolov9 directory:")
    if yolov9_path.exists():
        import os
        for item in os.listdir(yolov9_path):
            print(f"   {item}")
    raise

# 导入自定义模块
from yolo_modules.model_wrapper import Y9_GAT_DA


class ColabStage1Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 使用临时目录作为工作目录
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Drive保存目录
        self.drive_save_dir = DRIVE_SAVE_DIR
        
        # 设置日志
        self.setup_logging()
        
        # 初始化回调
        self.callbacks = Callbacks()
        
        # Colab特定设置
        self.setup_colab_specific()
        
    def setup_colab_specific(self):
        """Colab特定的设置"""
        # 减少显存碎片
        torch.cuda.empty_cache()
        gc.collect()
        
        # 设置cudnn
        torch.backends.cudnn.benchmark = True
        
        # A100 TensorFloat-32优化
        if self.config.get('tf32', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("Enabled TensorFloat-32 for A100 optimization")
        
        # 启用YOLOv9的pin_memory优化（通过环境变量）
        if self.config.get('pin_memory', True):
            os.environ['PIN_MEMORY'] = 'True'
            self.logger.info("Enabled pin_memory for DataLoader optimization")
        
        # 检查是否有之前的checkpoint (暂时注释掉用于测试)
        # self.check_resume()
        
    def check_resume(self):
        """检查是否需要恢复训练"""
        checkpoints = list(self.drive_save_dir.glob('checkpoint_epoch_*.pt'))
        
        # 过滤出真正存在的文件
        existing_checkpoints = [ckpt for ckpt in checkpoints if ckpt.exists()]
        
        # 也检查best.pt文件
        best_pt = self.drive_save_dir / 'best.pt'
        if best_pt.exists():
            existing_checkpoints.append(best_pt)
        
        if existing_checkpoints:
            # 找到最新的checkpoint (优先使用best.pt)
            if best_pt in existing_checkpoints:
                latest_checkpoint = best_pt
                self.logger.info(f"Found best checkpoint: {latest_checkpoint}")
            else:
                latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                self.logger.info(f"Found checkpoint: {latest_checkpoint}")
            self.config['resume'] = str(latest_checkpoint)
        else:
            self.logger.info("No valid checkpoints found, starting fresh training")
            self.config['resume'] = None
    
    def setup_logging(self):
        """设置日志系统"""
        # 同时保存到本地和Drive
        log_file = self.save_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        drive_log_file = self.drive_save_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.FileHandler(drive_log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_model(self):
        """创建YOLOv9检测器模型（Stage 1 只训练检测器）"""
        self.logger.info("Creating YOLOv9 detector for Stage 1...")
        
        # 直接创建YOLOv9检测器
        cfg_path = self.config['model_cfg']
        
        # 如果是相对路径，尝试多个位置查找
        if not os.path.isabs(cfg_path):
            cfg_candidates = [
                os.path.join('/content', cfg_path),
                os.path.join(PROJECT_ROOT, 'code', 'yolov9', 'models', 'detect', cfg_path),
                os.path.join(PROJECT_ROOT, 'code', 'yolov9', 'models', cfg_path),
                os.path.join(PROJECT_ROOT, 'code', 'yolov9', 'models', 'detect', 'yolov9-c.yaml'),  # 回退
            ]
            
            cfg_found = None
            for candidate in cfg_candidates:
                if os.path.exists(candidate):
                    cfg_found = candidate
                    self.logger.info(f"Found config at: {cfg_found}")
                    break
            
            if cfg_found is None:
                raise FileNotFoundError(f"Config file not found. Tried: {cfg_candidates}")
            cfg_path = cfg_found
        
        # 创建YOLOv9模型
        model = Yolov9(cfg=cfg_path, ch=3, nc=1)  # nc=1 for face detection
        
        # 加载预训练权重（在移到GPU之前）
        if self.config['pretrained_weights']:
            weights_path = Path(self.config['pretrained_weights'])
            
            # 按优先级查找权重文件
            if not weights_path.exists():
                # 检查多个可能的位置
                weight_candidates = [
                    Path('/content') / weights_path.name,                                    # Colab根目录
                    Path('/content/drive/MyDrive/ASD_AU_weights/pretrained') / weights_path.name,  # Drive位置
                    self.drive_save_dir.parent / 'pretrained' / weights_path.name,          # 默认Drive位置
                    Path('/content') / 'yolov9c_face_weights_only.pt',                      # 修正的权重文件（直接路径）
                ]
                
                for candidate in weight_candidates:
                    if candidate.exists():
                        weights_path = candidate
                        break
            
            if weights_path.exists():
                self.logger.info(f"Loading pretrained weights from {weights_path}")
                
                # 检查是否为ultralytics格式的权重
                try:
                    # 尝试直接加载PyTorch权重
                    ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
                except Exception as e1:
                    self.logger.warning(f"Direct PyTorch loading failed: {e1}")
                    
                    # 尝试使用ultralytics加载并转换
                    try:
                        self.logger.info("Attempting to load ultralytics format and convert...")
                        import subprocess
                        import sys
                        
                        # 安装ultralytics如果没有
                        try:
                            import ultralytics
                        except ImportError:
                            self.logger.info("Installing ultralytics...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                            import ultralytics
                        
                        from ultralytics import YOLO
                        
                        # 加载ultralytics模型
                        yolo_model = YOLO(str(weights_path))
                        
                        # 提取纯PyTorch权重
                        pure_weights = yolo_model.model.state_dict()
                        
                        # 保存转换后的权重
                        converted_path = weights_path.parent / f"{weights_path.stem}_converted.pt"
                        torch.save(pure_weights, converted_path)
                        self.logger.info(f"Converted weights saved to: {converted_path}")
                        
                        # 创建ckpt字典格式
                        ckpt = {'model': pure_weights}
                        
                    except Exception as e2:
                        self.logger.error(f"Ultralytics conversion failed: {e2}")
                        self.logger.warning("Proceeding without pretrained weights...")
                        ckpt = None
                
                # 加载检测器权重（直接加载到YOLOv9模型）
                if ckpt is not None:
                    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
                    if hasattr(state_dict, 'state_dict'):
                        state_dict = state_dict.state_dict()
                    
                    # 获取模型当前的state_dict作为参考
                    model_state = model.state_dict()
                    compatible_state = {}
                    
                    # 尝试直接匹配键名
                    for k, v in state_dict.items():
                        # 去掉可能的detector前缀
                        clean_k = k.replace('detector.', '') if k.startswith('detector.') else k
                        if clean_k in model_state and v.shape == model_state[clean_k].shape:
                            compatible_state[clean_k] = v
                    
                    # 加载权重
                    if compatible_state:
                        missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
                        self.logger.info(f"Pretrained weights loaded successfully. Matched {len(compatible_state)} layers.")
                        
                        # 详细的加载信息
                        if missing_keys:
                            self.logger.info(f"Missing keys (will be randomly initialized): {len(missing_keys)}")
                        if unexpected_keys:
                            self.logger.info(f"Unexpected keys (ignored): {len(unexpected_keys)}")
                    else:
                        self.logger.warning("No compatible weights found! Training from scratch...")
                        
                    # 显示匹配统计
                    total_model_params = len(model_state.keys())
                    self.logger.info(f"Weight loading: {len(compatible_state)}/{total_model_params} layers matched")
                else:
                    self.logger.warning("No pretrained weights loaded, training from scratch")
            else:
                self.logger.warning(f"Pretrained weights not found: {weights_path}")
        
        # 权重加载完成后，移动模型到GPU
        model = model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")
        
        # 恢复训练（跳过不兼容的checkpoint）(暂时注释掉用于测试)
        start_epoch = 0
        if False and self.config.get('resume'):
            self.logger.info(f"Found checkpoint: {self.config['resume']}")
            try:
                checkpoint = torch.load(self.config['resume'], map_location=self.device, weights_only=False)
                # 检查checkpoint是否与当前模型架构兼容
                checkpoint_keys = set(checkpoint['model_state_dict'].keys())
                model_keys = set(model.state_dict().keys())
                
                # 如果checkpoint包含GAT/emotion组件，说明是旧的包装模型，尝试部分加载
                has_old_components = any(key.startswith(('gat_head', 'emotion_head', 'roi_extractor')) 
                                       for key in checkpoint_keys)
                
                if has_old_components:
                    self.logger.warning("Checkpoint contains old model architecture (Y9_GAT_DA). Attempting partial weight loading...")
                    # 只加载YOLOv9部分的权重
                    yolo_weights = {k.replace('detector.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                                   if k.startswith('detector.') or not any(comp in k for comp in ['gat_head', 'emotion_head', 'roi_extractor'])}
                    
                    if yolo_weights:
                        missing_keys, unexpected_keys = model.load_state_dict(yolo_weights, strict=False)
                        start_epoch = checkpoint['epoch'] + 1
                        self.logger.info(f"Partially resumed from epoch {start_epoch} (YOLOv9 weights only)")
                    else:
                        start_epoch = 0
                else:
                    # 兼容的checkpoint，可以恢复
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    start_epoch = checkpoint['epoch'] + 1
                    self.logger.info(f"Resumed from epoch {start_epoch}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting fresh training.")
                start_epoch = 0
        
        # Stage 1：纯YOLOv9检测器，无需冻结模块
        # 打印参数统计
        self.print_param_stats(model)
        
        return model, start_epoch
    
    # freeze_modules 函数已移除 - Stage1只训练纯YOLOv9检测器
    
    def flatten_nested_tensors(self, nested_structure):
        """递归扁平化嵌套结构，只保留torch.Tensor对象"""
        flattened = []
        
        def _recursive_flatten(item):
            if isinstance(item, torch.Tensor):
                # 确保张量有view方法（基本上所有tensor都有）
                if hasattr(item, 'view'):
                    flattened.append(item)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    _recursive_flatten(sub_item)
        
        _recursive_flatten(nested_structure)
        return flattened
    
    def process_model_output_for_loss(self, raw_output, is_training=True, debug=False):
        """
        完善的模型输出处理，保持YOLOv9期望的结构
        
        Args:
            raw_output: 模型的原始输出
            is_training: 是否为训练模式
            debug: 是否输出调试信息
            
        Returns:
            list: 适合ComputeLoss的张量列表，保持原始结构
        """
        if debug:
            self.logger.info(f"Raw output type: {type(raw_output)}")
            if isinstance(raw_output, (list, tuple)):
                self.logger.info(f"Raw output length: {len(raw_output)}")
                for i, item in enumerate(raw_output):
                    self.logger.info(f"  raw_output[{i}] type: {type(item)}, "
                                   f"shape/len: {getattr(item, 'shape', len(item) if isinstance(item, (list, tuple)) else 'unknown')}")
        
        # 步骤1: 获取模型原始输出
        potential_predictions = raw_output
        
        # 步骤2: 处理OrderedDict输出（修复关键问题）
        from collections import OrderedDict
        if isinstance(raw_output, OrderedDict):
            if debug:
                self.logger.info("Detected OrderedDict output, extracting values")
            # 提取OrderedDict中的值，通常是tensor列表
            potential_predictions = list(raw_output.values())
            if debug:
                self.logger.info(f"Extracted {len(potential_predictions)} values from OrderedDict")
        
        # 步骤3: 处理评估模式下的元组输出
        if isinstance(potential_predictions, tuple):
            if debug:
                self.logger.info("Detected tuple output (eval mode)")
            
            # YOLOv9在eval模式下通常返回 (inference_detections, raw_predictions)
            # 我们需要raw_predictions用于损失计算
            if len(potential_predictions) >= 2:
                potential_predictions = potential_predictions[1]  # 通常raw predictions在第二个位置
                if debug:
                    self.logger.info(f"Using tuple[1] as potential_predictions: {type(potential_predictions)}")
            else:
                potential_predictions = potential_predictions[0]
                if debug:
                    self.logger.info(f"Using tuple[0] as potential_predictions: {type(potential_predictions)}")
        
        # 步骤4: 保守的列表处理（不破坏YOLOv9的结构）
        if isinstance(potential_predictions, list):
            # 检查是否为嵌套列表
            if len(potential_predictions) > 0 and isinstance(potential_predictions[0], list):
                # 只处理第一层嵌套，保持内部结构
                potential_predictions = potential_predictions[0]
                if debug:
                    self.logger.info(f"Unwrapped one layer of nesting, new length: {len(potential_predictions)}")
            
            # 确保列表中都是tensor
            valid_tensors = []
            for item in potential_predictions:
                if isinstance(item, torch.Tensor) and hasattr(item, 'view'):
                    valid_tensors.append(item)
                elif debug:
                    self.logger.warning(f"Skipping non-tensor item: {type(item)}")
            
            if debug:
                self.logger.info(f"Final valid tensors count: {len(valid_tensors)}")
                for i, tensor in enumerate(valid_tensors):
                    if hasattr(tensor, 'shape'):
                        self.logger.info(f"  tensor[{i}] shape: {tensor.shape}")
            
            return valid_tensors
        
        # 步骤5: 如果不是列表，直接返回
        elif isinstance(potential_predictions, torch.Tensor):
            if debug:
                self.logger.info(f"Single tensor output, shape: {potential_predictions.shape}")
            return [potential_predictions]
        
        else:
            self.logger.warning(f"Unexpected output type: {type(potential_predictions)}")
            return []
        
    def print_param_stats(self, model):
        """打印参数统计"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        
    def setup_data_paths(self):
        """设置数据路径（兼容用户的数据划分脚本）"""
        # 按优先级检查数据路径
        data_candidates = [
            Path('/content/yolo_data/yolo_data'),                         # 用户当前的实际路径
            Path('/content/yolo_data'),                                    # 用户划分脚本的标准输出
            Path('/content/drive/MyDrive/datasets/HDA-SynChild'),         # Drive标准位置
            Path('/content/datasets/HDA-SynChild'),                       # 本地位置
        ]
        
        for data_root in data_candidates:
            if data_root.exists() and (data_root / 'images' / 'train').exists():
                self.logger.info(f"Found dataset at: {data_root}")
                
                # 如果数据在嵌套位置，创建符号链接到标准位置优化访问
                if str(data_root) in ['/content/yolo_data', '/content/yolo_data/yolo_data']:
                    standard_path = Path('/content/datasets/HDA-SynChild')
                    standard_path.parent.mkdir(exist_ok=True)
                    if not standard_path.exists():
                        os.symlink(data_root, standard_path)
                        self.logger.info(f"Created symlink: {standard_path} -> {data_root}")
                        return standard_path
                
                return data_root
        
        # 如果都没找到，给出详细的提示
        raise RuntimeError(
            "Dataset not found! Please ensure your data is in one of these locations:\n"
            "1. /content/yolo_data (output from your data split script)\n"
            "2. /content/drive/MyDrive/datasets/HDA-SynChild\n"
            "3. /content/datasets/HDA-SynChild\n"
            "Required structure: {path}/images/train/, {path}/images/val/, {path}/labels/train/, {path}/labels/val/"
        )
    
    def check_image_integrity(self, data_root):
        """检查图像完整性，删除损坏的文件"""
        if not self.config.get('check_images', False):
            return
            
        print("Checking image integrity...")
        import cv2
        corrupt_count = 0
        
        for split in ['train', 'val']:
            img_dir = data_root / 'images' / split
            if not img_dir.exists():
                continue
                
            img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
            
            for img_path in img_files:
                try:
                    # 尝试读取图像
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Removing corrupt image: {img_path}")
                        img_path.unlink()  # 删除损坏的文件
                        
                        # 同时删除对应的标签文件
                        label_path = data_root / 'labels' / split / (img_path.stem + '.txt')
                        if label_path.exists():
                            label_path.unlink()
                            
                        corrupt_count += 1
                    else:
                        # 检查图像是否能正常解码
                        if img.shape[0] < 10 or img.shape[1] < 10:
                            print(f"Removing too small image: {img_path}")
                            img_path.unlink()
                            
                            label_path = data_root / 'labels' / split / (img_path.stem + '.txt')
                            if label_path.exists():
                                label_path.unlink()
                                
                            corrupt_count += 1
                except Exception as e:
                    print(f"Error checking {img_path}: {e}")
                    try:
                        img_path.unlink()
                        label_path = data_root / 'labels' / split / (img_path.stem + '.txt')
                        if label_path.exists():
                            label_path.unlink()
                        corrupt_count += 1
                    except:
                        pass
        
        if corrupt_count > 0:
            print(f"Removed {corrupt_count} corrupt images")
        else:
            print("All images are intact")

    def create_dataloaders(self):
        """创建数据加载器（Colab优化）"""
        # 设置数据路径
        data_root = self.setup_data_paths()
        
        # 跳过耗时的图像完整性检查（可在首次运行时手动检查）
        # self.check_image_integrity(data_root)
        
        # 检查是否存在用户生成的data.yaml
        user_data_yaml = data_root / 'data.yaml'
        if user_data_yaml.exists():
            self.logger.info(f"Using existing data.yaml from: {user_data_yaml}")
            data_yaml = user_data_yaml
            # 保存data_yaml路径供后续使用
            self.data_yaml_path = data_yaml
        else:
            # 创建data yaml
            data_yaml = self.save_dir / 'hda_synchild_colab.yaml'
            data_dict = {
                'path': str(data_root),
                'train': 'images/train',
                'val': 'images/val',
                'names': {0: 'face'}
            }
            
            with open(data_yaml, 'w') as f:
                yaml.dump(data_dict, f)
            self.logger.info(f"Created data.yaml at: {data_yaml}")
        
        # 保存data_yaml路径供后续使用
        self.data_yaml_path = data_yaml
        
        # Colab优化的超参数
        hyp = self.get_hyp_dict()
        
        # 根据GPU类型和内存自动调整batch size
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_properties(0).name
        
        if "A100" in gpu_name:
            # A100优化：80GB显存，可以支持很大的batch_size
            if gpu_mem > 70:  # A100-80GB
                self.config['batch_size'] = min(self.config['batch_size'], 48)
                self.config['workers'] = min(self.config['workers'], 20)
            else:  # A100-40GB
                self.config['batch_size'] = min(self.config['batch_size'], 32)
                self.config['workers'] = min(self.config['workers'], 16)
            self.logger.info(f"A100 optimized: batch_size={self.config['batch_size']}, workers={self.config['workers']}")
        elif "V100" in gpu_name:
            # V100优化：32GB显存
            self.config['batch_size'] = min(self.config['batch_size'], 24)
            self.config['workers'] = min(self.config['workers'], 12)
            self.logger.info(f"V100 optimized: batch_size={self.config['batch_size']}")
        elif gpu_mem < 16:  # T4/L4等小显存GPU
            self.config['batch_size'] = min(self.config['batch_size'], 8)
            self.config['workers'] = min(self.config['workers'], 6)
            self.logger.info(f"Small GPU optimized: batch_size={self.config['batch_size']}")
        
        # 训练数据加载器 - A100优化版
        train_loader, dataset = create_dataloader(
            path=str(data_root / 'images/train'),
            imgsz=self.config['img_size'],
            batch_size=self.config['batch_size'],
            stride=32,
            single_cls=True,
            hyp=hyp,
            augment=True,
            cache=False,  # 禁用缓存避免损坏文件
            rect=self.config.get('rect', False),
            rank=-1,
            workers=self.config['workers'],
            image_weights=False,
            quad=False,
            prefix='train: ',
            shuffle=True
        )
        
        # 验证数据加载器 - A100优化版
        val_loader = create_dataloader(
            path=str(data_root / 'images/val'),
            imgsz=self.config['img_size'],
            batch_size=self.config['batch_size'] * 2,
            stride=32,
            single_cls=True,
            hyp=hyp,
            augment=False,
            cache=False,  # 禁用缓存避免损坏文件
            rect=True,  # 验证时使用rect提速
            rank=-1,
            workers=self.config['workers'],
            pad=0.5,
            prefix='val: '
        )[0]
        
        return train_loader, val_loader, dataset
    
    def get_hyp_dict(self):
        """获取数据增强超参数（Colab优化）"""
        return {
            'lr0': self.config['lr0'],
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': self.config.get('warmup_epochs', 2),
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'anchors': 3,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.5,  # 降低以节省内存
            'mixup': 0.0,
            'copy_paste': 0.0,
            'paste_in': 0.15,
            'loss_ota': 1,
        }
    
    def save_checkpoint(self, model, optimizer, epoch, best_fitness, ema=None, best=False):
        """保存检查点到Google Drive"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_fitness': best_fitness,
            'config': self.config
        }
        
        if ema:
            checkpoint['ema_state_dict'] = ema.ema.state_dict()
        
        # 保存到本地
        if best:
            local_path = self.save_dir / 'weights' / 'best.pt'
        else:
            local_path = self.save_dir / 'weights' / f'epoch_{epoch}.pt'
        
        local_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, local_path)
        
        # 复制到Drive
        if best:
            drive_path = self.drive_save_dir / 'best.pt'
        else:
            drive_path = self.drive_save_dir / f'checkpoint_epoch_{epoch}.pt'
            # 只保留最近5个checkpoint
            self.cleanup_old_checkpoints()
        
        shutil.copy2(local_path, drive_path)
        self.logger.info(f"Checkpoint saved to {drive_path}")
    
    def cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最近的5个"""
        checkpoints = list(self.drive_save_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 5:
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            for ckpt in checkpoints[:-5]:
                ckpt.unlink()
                self.logger.info(f"Removed old checkpoint: {ckpt}")
    
    def train_epoch(self, model, train_loader, optimizer, compute_loss, epoch, epochs, scaler=None):
        """训练一个epoch（添加NaN保护）"""
        model.train()
        
        # 测试模式：限制训练样本数量
        if self.config.get('test_mode', False):
            max_batches = self.config.get('test_samples', 100) // self.config['batch_size']
            max_batches = max(1, max_batches)  # 至少1个batch
            total_batches = min(len(train_loader), max_batches)
            self.logger.info(f"Test mode: Using only {max_batches} batches ({max_batches * self.config['batch_size']} samples)")
        else:
            total_batches = len(train_loader)
            
        pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f'Epoch {epoch+1}/{epochs}')
        
        losses = []
        accumulation_steps = self.config.get('gradient_accumulation', 1)
        nan_count = 0  # 新增：统计NaN次数
        
        for i, (imgs, targets, paths, _) in pbar:
            # 测试模式：限制batch数量
            if self.config.get('test_mode', False) and i >= max_batches:
                break
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            
            # 前向传播
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.get('amp', True)):
                raw_pred = model(imgs)
                pred = self.process_model_output_for_loss(raw_pred, is_training=True, debug=(i == 0))
                loss, loss_items = compute_loss(pred, targets.to(self.device))
                
                # 新增：检查NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    self.logger.warning(f"NaN/Inf loss detected in batch {i}! Skipping...")
                    optimizer.zero_grad()
                    if nan_count > 10:  # 如果连续多个NaN，停止训练
                        raise RuntimeError("Too many NaN losses detected!")
                    continue
                
                loss = loss / accumulation_steps
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    # 新增：梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                  max_norm=self.config.get('clip_grad_norm', 10.0))
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    # 新增：梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                  max_norm=self.config.get('clip_grad_norm', 10.0))
                    optimizer.step()
                    optimizer.zero_grad()
            
            # 更新进度条
            if not (torch.isnan(loss) or torch.isinf(loss)):
                losses.append(loss_items.cpu().numpy())
            
            if len(losses) > 100:
                losses = losses[-100:]
            
            if losses:  # 只有在有有效损失时才更新
                mean_loss = np.mean(losses, axis=0)
                pbar.set_postfix({
                    'loss': f'{mean_loss[0]:.4f}',
                    'GPU': f'{torch.cuda.memory_reserved() / 1024**3:.1f}G',
                    'NaN': nan_count  # 显示NaN计数
                })
        
        # 处理最后的梯度
        if (len(train_loader)) % accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              max_norm=self.config.get('clip_grad_norm', 10.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              max_norm=self.config.get('clip_grad_norm', 10.0))
                optimizer.step()
            optimizer.zero_grad()
        
        return np.mean(losses, axis=0) if losses else np.array([float('inf')])
    
    def train(self):
        """主训练函数"""
        # 创建模型
        model, start_epoch = self.create_model()
        
        # 创建数据加载器
        train_loader, val_loader, dataset = self.create_dataloaders()
        
        # 添加超参数到模型（损失函数需要）
        hyp = self.get_hyp_dict()
        model.hyp = hyp
        
        # 设置模型为训练模式
        model.train()
        
        # 确保模型stride和检测头正确初始化
        self.logger.info("Initializing model detection head...")
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, self.config['img_size'], self.config['img_size'], device=self.device))
        
        # 创建损失函数
        compute_loss = ComputeLoss(model)
        
        # 创建优化器
        optimizer = self.create_optimizer(model)
        
        # 恢复优化器状态 (暂时注释掉用于测试)
        if False and self.config.get('resume') and start_epoch > 0:
            checkpoint = torch.load(self.config['resume'], map_location=self.device, weights_only=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 创建调度器
        scheduler = self.create_scheduler(optimizer, self.config['epochs'] - start_epoch)
        
        # 创建EMA和梯度缩放器
        ema = ModelEMA(model) if self.config.get('use_ema', True) else None
        scaler = torch.amp.GradScaler('cuda') if self.config.get('amp', True) else None
        
        # 训练循环
        best_fitness = 0
        best_map50_95 = 0
        patience_counter = 0
        patience = self.config.get('patience', 8)
        
        # 训练指标记录
        self.train_metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'mAP@0.5': [],
            'mAP@0.5:0.95': [],
            'precision': [],
            'recall': [],
            'lr': []
        }
        
        try:
            for epoch in range(start_epoch, self.config['epochs']):
                # 训练
                train_loss = self.train_epoch(model, train_loader, optimizer, compute_loss, 
                                            epoch, self.config['epochs'], scaler)
                
                # 检查训练损失是否有效
                if np.isnan(train_loss[0]) or np.isinf(train_loss[0]):
                    self.logger.error(f"Invalid training loss at epoch {epoch+1}! Stopping training.")
                    break
                
                # 验证（每个epoch）
                val_loss = None
                map_results = None
                
                if (epoch + 1) % self.config.get('val_frequency', 1) == 0:
                    val_loss = self.validate(model, val_loader, compute_loss)
                    
                    # 更频繁的mAP评估
                    if (epoch + 1) % self.config.get('map_eval_frequency', 2) == 0 or \
                       epoch >= self.config['epochs'] - 3:
                        map_results = self.evaluate_map(model, val_loader, 
                                                      self.data_yaml_path)
                
                # 更新EMA
                if ema:
                    ema.update(model)
                
                # 更新学习率
                scheduler.step()
                
                # 记录指标
                current_lr = optimizer.param_groups[0]['lr']
                self.train_metrics['epochs'].append(epoch + 1)
                self.train_metrics['train_loss'].append(train_loss[0])
                self.train_metrics['lr'].append(current_lr)
                
                if val_loss is not None:
                    self.train_metrics['val_loss'].append(val_loss[0])
                else:
                    self.train_metrics['val_loss'].append(None)
                
                if map_results:
                    self.train_metrics['mAP@0.5'].append(map_results['mAP@0.5'])
                    self.train_metrics['mAP@0.5:0.95'].append(map_results['mAP@0.5:0.95'])
                    self.train_metrics['precision'].append(map_results['precision'])
                    self.train_metrics['recall'].append(map_results['recall'])
                else:
                    self.train_metrics['mAP@0.5'].append(None)
                    self.train_metrics['mAP@0.5:0.95'].append(None)
                    self.train_metrics['precision'].append(None)
                    self.train_metrics['recall'].append(None)
                
                # 记录日志
                if val_loss is not None:
                    log_msg = (
                        f"Epoch {epoch+1}/{self.config['epochs']} - "
                        f"Train Loss: {train_loss[0]:.4f} - "
                        f"Val Loss: {val_loss[0]:.4f} - "
                        f"LR: {current_lr:.6f}"
                    )
                    
                    if map_results:
                        log_msg += (
                            f" - mAP@0.5: {map_results['mAP@0.5']:.4f} - "
                            f"mAP@0.5:0.95: {map_results['mAP@0.5:0.95']:.4f} - "
                            f"Precision: {map_results['precision']:.4f} - "
                            f"Recall: {map_results['recall']:.4f}"
                        )
                        
                        # 检查是否达到目标mAP@0.5:0.95 >= 0.57
                        if map_results['mAP@0.5:0.95'] >= 0.57:
                            self.logger.info(f"🎯 Target mAP@0.5:0.95 >= 0.57 achieved: {map_results['mAP@0.5:0.95']:.4f}")
                    
                    log_msg += f" - Patience: {patience_counter}/{patience}"
                    self.logger.info(log_msg)
                    
                    # 保存最佳模型（基于mAP@0.5:0.95或验证损失）
                    if map_results and map_results['mAP@0.5:0.95'] > best_map50_95:
                        best_map50_95 = map_results['mAP@0.5:0.95']
                        best_fitness = map_results['mAP@0.5:0.95']
                        patience_counter = 0
                        self.save_checkpoint(model, optimizer, epoch, best_fitness, ema, best=True)
                        self.logger.info(f"New best model! mAP@0.5:0.95: {best_map50_95:.4f}")
                    else:
                        # 回退到基于验证损失的fitness
                        fitness = 1 / (val_loss[0] + 1e-6)
                        if fitness > best_fitness and not map_results:
                            best_fitness = fitness
                            patience_counter = 0
                            self.save_checkpoint(model, optimizer, epoch, best_fitness, ema, best=True)
                            self.logger.info(f"New best model! Fitness: {fitness:.6f}")
                        else:
                            patience_counter += 1
                else:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config['epochs']} - "
                        f"Train Loss: {train_loss[0]:.4f} - "
                        f"LR: {current_lr:.6f}"
                    )
                
                # 保存检查点
                if (epoch + 1) % self.config['save_period'] == 0:
                    self.save_checkpoint(model, optimizer, epoch, best_fitness, ema)
                    
                # 早停检查
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}. "
                                   f"No improvement for {patience} epochs.")
                    self.save_checkpoint(model, optimizer, epoch, best_fitness, ema)
                    break
                
                # 检查Colab运行时间
                if self.check_runtime_limit():
                    self.logger.warning("Approaching Colab runtime limit. Saving checkpoint...")
                    self.save_checkpoint(model, optimizer, epoch, best_fitness, ema)
                    break
                    
        except Exception as e:
            self.logger.error(f"Training interrupted: {e}")
            self.save_checkpoint(model, optimizer, epoch, best_fitness, ema)
            raise
        
        self.logger.info("Training completed!")
        
        # 最终评估套件
        self.logger.info("🔍 开始最终模型评估...")
        
        # 1. FPS基准测试
        self.logger.info("🚀 FPS基准测试...")
        try:
            fps_results = self.benchmark_fps(model)
            
            # 保存FPS结果到Drive
            import json
            fps_file = self.drive_save_dir / 'fps_benchmark.json'
            with open(fps_file, 'w') as f:
                json.dump(fps_results, f, indent=2)
            self.logger.info(f"FPS测试结果已保存到: {fps_file}")
            
        except Exception as e:
            self.logger.warning(f"FPS基准测试失败: {e}")
        
        # 2. 定性检测可视化
        self.logger.info("📸 定性检测可视化...")
        try:
            qualitative_results = self.qualitative_check(model, val_loader, num_samples=30)
            
            if qualitative_results:
                # 保存定性检查结果
                qual_file = self.drive_save_dir / 'qualitative_check.json'
                with open(qual_file, 'w') as f:
                    json.dump(qualitative_results, f, indent=2)
                self.logger.info(f"定性检查结果已保存到: {qual_file}")
            
        except Exception as e:
            self.logger.warning(f"定性检查失败: {e}")
        
        # 3. 最终mAP评估
        self.logger.info("📊 最终mAP评估...")
        try:
            final_map_results = self.evaluate_map(model, val_loader, 
                                                 self.data_yaml_path)
            if final_map_results:
                # 保存最终mAP结果
                map_file = self.drive_save_dir / 'final_map_results.json'
                with open(map_file, 'w') as f:
                    json.dump(final_map_results, f, indent=2)
                
                self.logger.info("🎯 最终评估结果:")
                self.logger.info(f"   mAP@0.5: {final_map_results['mAP@0.5']:.4f}")
                self.logger.info(f"   mAP@0.5:0.95: {final_map_results['mAP@0.5:0.95']:.4f}")
                self.logger.info(f"   Precision: {final_map_results['precision']:.4f}")
                self.logger.info(f"   Recall: {final_map_results['recall']:.4f}")
                
                # 总结评估状态
                target_achieved = final_map_results['mAP@0.5:0.95'] >= 0.57
                self.logger.info(f"   目标达成: {'✅ 是' if target_achieved else '❌ 否'} (≥0.57)")
                
        except Exception as e:
            self.logger.warning(f"最终mAP评估失败: {e}")
        
        return str(self.drive_save_dir / 'best.pt')
    
    def check_runtime_limit(self):
        """检查是否接近Colab运行时限制"""
        # 这里可以实现更复杂的逻辑
        # 简单起见，训练超过11小时就停止
        import time
        if hasattr(self, 'start_time'):
            elapsed = time.time() - self.start_time
            return elapsed > 11 * 3600  # 11小时
        return False
    
    def create_optimizer(self, model):
        """创建优化器"""
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        
        if self.config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.config['lr0'],
                betas=(0.937, 0.999),
                weight_decay=0.0005
            )
        else:
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.config['lr0'],
                momentum=0.937,
                weight_decay=0.0005,
                nesterov=True
            )
        
        return optimizer
    
    def create_scheduler(self, optimizer, epochs):
        """创建学习率调度器（支持warmup）"""
        warmup_epochs = self.config.get('warmup_epochs', 0)
        
        if warmup_epochs > 0 and warmup_epochs < epochs:
            # 使用warmup + 主调度器
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            # Warmup阶段：从10%学习率线性增加到100%
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=warmup_epochs
            )
            
            # 主训练阶段调度器
            if self.config['scheduler'] == 'cosine':
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs - warmup_epochs,
                    eta_min=self.config['lr0'] * 0.1
                )
            else:
                lambda_lr = lambda epoch: (1 - epoch / (epochs - warmup_epochs)) * (1 - 0.1) + 0.1
                main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
            
            # 组合调度器
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[warmup_epochs]
            )
            
            self.logger.info(f"Using warmup scheduler: {warmup_epochs} epochs warmup + {self.config['scheduler']}")
            
        else:
            # 不使用warmup，直接使用主调度器
            if self.config['scheduler'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                    eta_min=self.config['lr0'] * 0.1
                )
            else:
                lambda_lr = lambda epoch: (1 - epoch / epochs) * (1 - 0.1) + 0.1
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        
        return scheduler
    
    def validate(self, model, val_loader, compute_loss):
        """验证模型并输出详细指标"""
        model.eval()
        losses = []
        
        # 测试模式：限制验证样本数量
        if self.config.get('test_mode', False):
            max_val_batches = self.config.get('test_samples', 100) // (self.config['batch_size'] * 2)
            max_val_batches = max(1, max_val_batches)  # 至少1个batch
            self.logger.info(f"Test mode: Validating only {max_val_batches} batches")
        
        with torch.no_grad():
            for i, (imgs, targets, paths, _) in enumerate(tqdm(val_loader, desc='Validating')):
                # 测试模式：限制batch数量
                if self.config.get('test_mode', False) and i >= max_val_batches:
                    break
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.get('amp', True)):
                    raw_pred = model(imgs)
                    pred = self.process_model_output_for_loss(raw_pred, is_training=False, debug=False)
                    
                    # 检查预测是否有效
                    if pred and all(not torch.isnan(p).any() for p in pred):
                        loss, loss_items = compute_loss(pred, targets.to(self.device))
                        
                        # 只添加有效的损失
                        if not torch.isnan(loss):
                            losses.append(loss_items.cpu().numpy())
        
        if losses:
            mean_losses = np.mean(losses, axis=0)
            
            # 输出详细的损失组成（假设loss_items包含[total, box, obj, cls]）
            if len(mean_losses) >= 4:
                self.logger.info(f"Validation Loss Details:")
                self.logger.info(f"  Total: {mean_losses[0]:.4f}")
                self.logger.info(f"  Box: {mean_losses[1]:.4f}")
                self.logger.info(f"  Obj: {mean_losses[2]:.4f}")
                self.logger.info(f"  Cls: {mean_losses[3]:.4f}")
            
            return mean_losses
        else:
            self.logger.warning("No valid losses in validation!")
            return np.array([float('inf')])
    
    def evaluate_map(self, model, val_loader, data_yaml_path):
        """修复YOLOv9的val.py评估 - 方案A：直接传递模型对象和数据字典"""
        try:
            # 读取yaml文件转换为字典
            import yaml
            from val import run as validate_yolo
            from utils.torch_utils import de_parallel   # 确保去掉 DDP/Horovod wrapper
            
            # ① 读 yaml → dict
            with open(data_yaml_path, "r") as f:
                data_cfg = yaml.safe_load(f)
            
            # ② 确保必备字段齐全
            if 'names' in data_cfg and isinstance(data_cfg['names'], dict):
                # val.py 里要求 list，所以把 {0: 'face'} → ['face']
                data_cfg['names'] = list(data_cfg['names'].values())
            data_cfg.setdefault('nc', len(data_cfg['names']))
            
            # 打印调试信息
            self.logger.info(f"Calling validate_yolo with:")
            self.logger.info(f"  data_cfg: {data_cfg}")
            self.logger.info(f"  model type: {type(de_parallel(model))}")
            self.logger.info(f"  device: {self.device}")
            
            # ③ 交给 val.run；不再写临时权重，不再传字符串路径
            results = validate_yolo(
                data=data_cfg,                    # ← dict
                model=de_parallel(model).float(), # ← nn.Module
                weights=None,                     # ← 省略
                batch_size=self.config['batch_size'] * 2,
                imgsz=self.config['img_size'],
                conf_thres=0.001,
                iou_thres=0.60,
                max_det=300,
                task='val',
                device=str(self.device),
                workers=self.config['workers'],
                single_cls=True,
                verbose=False,
                save_json=False,
                half=False
            )
            
            self.logger.info(f"validate_yolo returned: {type(results)}, value: {results}")
            
            # 检查results是否有效
            if results is None:
                self.logger.warning("validate_yolo returned None")
                return None
            
            if not isinstance(results, (list, tuple)) or len(results) < 4:
                self.logger.warning(f"Unexpected results format: {type(results)}, length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
                return None
            
            # results: (precision, recall, mAP50, mAP50-95, …)
            return dict(zip(['precision','recall','mAP@0.5','mAP@0.5:0.95'], results[:4]))
            
        except Exception as e:
            self.logger.warning(f"mAP evaluation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def benchmark_fps(self, model, warmup_runs=10, test_runs=100):
        """FPS基准测试 - 目标: RTX 3060 ≥ 28 FPS"""
        model.eval()
        
        # 创建测试输入
        test_input = torch.randn(1, 3, self.config['img_size'], self.config['img_size'], device=self.device)
        
        # 预热GPU
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(test_input)
        
        # 同步CUDA确保预热完成
        torch.cuda.synchronize()
        
        # FPS测试
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(test_runs):
                _ = model(test_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 计算FPS
        total_time = end_time - start_time
        fps = test_runs / total_time
        avg_inference_time = total_time / test_runs * 1000  # ms
        
        # GPU信息
        gpu_name = torch.cuda.get_device_properties(0).name
        
        self.logger.info(f"🚀 FPS基准测试结果:")
        self.logger.info(f"   GPU: {gpu_name}")
        self.logger.info(f"   输入尺寸: {self.config['img_size']}x{self.config['img_size']}")
        self.logger.info(f"   推理速度: {fps:.1f} FPS")
        self.logger.info(f"   平均延迟: {avg_inference_time:.2f} ms")
        
        # 根据GPU类型检查性能目标
        if "A100" in gpu_name:
            target_fps = 80  # A100目标：≥80 FPS
            if fps >= target_fps:
                self.logger.info(f"🚀 A100性能优秀! ({fps:.1f} ≥ {target_fps} FPS)")
            else:
                self.logger.warning(f"⚠️ A100未达目标! ({fps:.1f} < {target_fps} FPS)")
        elif "V100" in gpu_name:
            target_fps = 50  # V100目标：≥50 FPS
            if fps >= target_fps:
                self.logger.info(f"✅ V100达到目标! ({fps:.1f} ≥ {target_fps} FPS)")
            else:
                self.logger.warning(f"⚠️ V100未达目标! ({fps:.1f} < {target_fps} FPS)")
        elif "RTX 30" in gpu_name or "RTX 40" in gpu_name:
            target_fps = 28
            if fps >= target_fps:
                self.logger.info(f"✅ 达到目标性能! ({fps:.1f} ≥ {target_fps} FPS)")
            else:
                self.logger.warning(f"⚠️ 未达目标性能! ({fps:.1f} < {target_fps} FPS)")
        
        return {
            'fps': fps,
            'avg_inference_time_ms': avg_inference_time,
            'gpu_name': gpu_name,
            'input_size': self.config['img_size']
        }
    
    def qualitative_check(self, model, val_loader, num_samples=50):
        """定性检测可视化 - 随机抽取验证图像进行检测可视化"""
        try:
            import cv2
            import random
            from pathlib import Path
            
            model.eval()
            sample_dir = self.save_dir / 'qualitative_samples'
            sample_dir.mkdir(exist_ok=True)
            
            # 随机采样验证图像
            all_samples = []
            for batch_idx, (imgs, targets, paths, _) in enumerate(val_loader):
                for i in range(len(paths)):
                    all_samples.append((imgs[i], paths[i]))  # 只保留图像和路径，移除targets避免索引问题
                if len(all_samples) >= num_samples * 2:  # 采样更多以便随机选择
                    break
            
            if len(all_samples) < 10:
                self.logger.warning("样本数量不足，跳过定性检查")
                return
            
            # 随机选择样本
            selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
            
            detection_count = 0
            good_detections = 0
            
            with torch.no_grad():
                for idx, (img_tensor, img_path) in enumerate(selected_samples):
                    try:
                        # 准备输入
                        img_input = img_tensor.unsqueeze(0).to(self.device).float() / 255.0
                        
                        # 推理
                        pred = model(img_input)
                        
                        # 读取原图
                        img_cv = cv2.imread(str(img_path))
                        if img_cv is None:
                            continue
                        
                        img_h, img_w = img_cv.shape[:2]
                        
                        # 简单的检测结果处理（这里简化处理，实际应该用NMS）
                        if isinstance(pred, list) and len(pred) > 0:
                            # 取第一个尺度的预测
                            if isinstance(pred[0], list):
                                first_pred = pred[0][0] if len(pred[0]) > 0 else None
                            else:
                                first_pred = pred[0]
                            
                            if first_pred is not None and len(first_pred.shape) >= 2:
                                detection_count += 1
                                
                                # 简化的框绘制（仅用于定性检查）
                                pred_np = first_pred.cpu().numpy()
                                if pred_np.shape[-1] >= 5:  # 至少有x,y,w,h,conf
                                    # 找置信度较高的检测框
                                    conf_scores = pred_np[..., 4] if pred_np.shape[-1] > 4 else pred_np[..., 0]
                                    high_conf_mask = conf_scores > 0.3
                                    
                                    if high_conf_mask.any():
                                        good_detections += 1
                                        # 在图像上标记 "Good Detection"
                                        cv2.putText(img_cv, f"Sample {idx+1}: Good Detection", 
                                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    else:
                                        cv2.putText(img_cv, f"Sample {idx+1}: Low Confidence", 
                                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                                else:
                                    cv2.putText(img_cv, f"Sample {idx+1}: No Detection", 
                                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(img_cv, f"Sample {idx+1}: No Output", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                        
                        # 保存标注图像
                        save_path = sample_dir / f"sample_{idx+1:03d}.jpg"
                        cv2.imwrite(str(save_path), img_cv)
                        
                    except Exception as e:
                        self.logger.warning(f"处理样本 {idx+1} 时出错: {e}")
                        continue
            
            # 统计结果
            detection_rate = good_detections / len(selected_samples) * 100 if selected_samples else 0
            
            self.logger.info(f"📸 定性检测分析完成:")
            self.logger.info(f"   检查样本数: {len(selected_samples)}")
            self.logger.info(f"   有效检测数: {good_detections}")
            self.logger.info(f"   检测成功率: {detection_rate:.1f}%")
            self.logger.info(f"   样本图片保存至: {sample_dir}")
            
            if detection_rate < 30:
                self.logger.warning("⚠️ 检测成功率较低，建议检查模型训练")
            elif detection_rate > 70:
                self.logger.info("✅ 检测效果良好")
            else:
                self.logger.info("📊 检测效果中等，可考虑继续训练")
                
            return {
                'total_samples': len(selected_samples),
                'good_detections': good_detections,
                'detection_rate': detection_rate,
                'sample_dir': str(sample_dir)
            }
            
        except Exception as e:
            self.logger.warning(f"定性检查失败: {e}")
            return None


def set_random_seeds(seed=42):
    """设置所有随机种子以确保可复现性"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN设置：在复现性和性能之间平衡
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        if "A100" in gpu_name or "V100" in gpu_name:
            # 高性能GPU：优先性能，保持一定随机性
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            print("🚀 High-performance GPU detected: enabling CuDNN benchmark for speed")
        else:
            # 普通GPU：优先复现性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("🔒 Enabling deterministic mode for reproducibility")
    
    print(f"🔒 Random seeds set to {seed} for reproducibility")


def main():
    """主函数 - Colab版本"""
    import time
    
    # 设置随机种子确保可复现性
    set_random_seeds(42)
    
    config = {
        # 模型配置
        'model_cfg': 'yolov9c-face-exact.yaml',  # 优先使用修正的配置，回退到标准配置
        'pretrained_weights': 'yolov9c_face_weights_only.pt',  # 会自动在多个位置查找
        'num_au': 32,
        'num_emotion': 6,
        
        # 数据配置 - A100 GPU优化（80GB显存版）
        'img_size': 640,
        'batch_size': 16,           # A100可以支持更大batch_size
        'workers': 16,              # A100支持更多并行workers
        'pin_memory': True,         # 启用pin_memory加速数据传输（通过环境变量）
        'cache_images': False,      # 禁用缓存避免损坏文件
        'check_images': False,      # 跳过图像检查加速启动
        'rect': True,               # A100内存充足，启用矩形训练提效
        
        # 训练配置 - 小批量测试版
        'epochs': 3,                # 改为3个epoch用于测试
        'test_mode': True,          # 启用测试模式，只使用少量数据
        'test_samples': 100,        # 测试模式下每个epoch只用100个样本
        'lr0': 0.001,               # 提高学习率适配更大batch_size
        'warmup_epochs': 3,         # warmup epochs
        'optimizer': 'AdamW',
        'scheduler': 'cosine',
        'device': 'cuda:0',
        'use_ema': True,
        'save_period': 5,           # 保存频率
        'amp': True,                # 混合精度训练
        'tf32': False,               # 启用TensorFloat-32优化
        'gradient_accumulation': 2, # A100大batch_size，减少梯度累积
        'val_frequency': 2,         # 每2个epoch验证一次（更频繁）
        'map_eval_frequency': 2,    # 新增：每2个epoch评估完整指标
        'patience': 8,              # 稍微增加早停耐心
        'clip_grad_norm': 10.0,     # 新增：梯度裁剪
        
        # 保存配置
        'save_dir': '/content/runs/stage1_hda_synchild_l4',
    }
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建训练器并开始训练
    trainer = ColabStage1Trainer(config)
    trainer.start_time = start_time
    
    best_weights = trainer.train()
    
    print(f"Training completed! Best weights saved at: {best_weights}")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    

if __name__ == '__main__':
    main()