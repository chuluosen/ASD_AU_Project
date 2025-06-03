"""
Stage 1: HDA-SynChild预训练脚本 - Colab适配版
只训练YOLOv9主干+检测头，冻结GAT-AU和情绪头
"""

import os
import sys
import torch
import torch.nn as nn
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
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"GPU Memory: {gpu_info.total_memory / 1024**3:.1f} GB")
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
        
        # 检查是否有之前的checkpoint
        self.check_resume()
        
    def check_resume(self):
        """检查是否需要恢复训练"""
        checkpoint_pattern = self.drive_save_dir / 'checkpoint_epoch_*.pt'
        checkpoints = list(self.drive_save_dir.glob('checkpoint_epoch_*.pt'))
        
        if checkpoints:
            # 找到最新的checkpoint
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            self.logger.info(f"Found checkpoint: {latest_checkpoint}")
            self.config['resume'] = str(latest_checkpoint)
        else:
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
        
        # 加载预训练权重
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
        
        # 恢复训练（跳过不兼容的checkpoint）
        start_epoch = 0
        if self.config.get('resume'):
            self.logger.info(f"Found checkpoint: {self.config['resume']}")
            try:
                checkpoint = torch.load(self.config['resume'], map_location=self.device)
                # 检查checkpoint是否与当前模型架构兼容
                checkpoint_keys = set(checkpoint['model_state_dict'].keys())
                model_keys = set(model.state_dict().keys())
                
                # 如果checkpoint包含GAT/emotion组件，说明是旧的包装模型，跳过
                has_old_components = any(key.startswith(('gat_head', 'emotion_head', 'roi_extractor')) 
                                       for key in checkpoint_keys)
                
                if has_old_components:
                    self.logger.warning("Checkpoint contains old model architecture (Y9_GAT_DA). Starting fresh training with new YOLOv9-only architecture.")
                    start_epoch = 0
                else:
                    # 兼容的checkpoint，可以恢复
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    start_epoch = checkpoint['epoch'] + 1
                    self.logger.info(f"Resumed from epoch {start_epoch}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting fresh training.")
                start_epoch = 0
        
        # Stage 1不需要冻结模块（只有检测器）
        # 打印参数统计
        self.print_param_stats(model)
        
        return model, start_epoch
    
    def freeze_modules(self, model):
        """冻结GAT-AU头和情绪头的参数"""
        frozen_modules = ['gat_head', 'emotion_head', 'roi_extractor']
        
        for name, param in model.named_parameters():
            if any(module in name for module in frozen_modules):
                param.requires_grad = False
                
        self.logger.info(f"Frozen modules: {frozen_modules}")
        
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
    
    def create_dataloaders(self):
        """创建数据加载器（Colab优化）"""
        # 设置数据路径
        data_root = self.setup_data_paths()
        
        # 检查是否存在用户生成的data.yaml
        user_data_yaml = data_root / 'data.yaml'
        if user_data_yaml.exists():
            self.logger.info(f"Using existing data.yaml from: {user_data_yaml}")
            data_yaml = user_data_yaml
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
        
        
        # Colab优化的超参数
        hyp = self.get_hyp_dict()
        
        # 根据GPU内存调整batch size
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem < 16:  # T4 GPU (15GB)
            self.config['batch_size'] = min(self.config['batch_size'], 8)
            self.logger.info(f"Adjusted batch size to {self.config['batch_size']} for GPU memory")
        
        # 训练数据加载器
        train_loader, dataset = create_dataloader(
            path=data_root / 'images/train',
            imgsz=self.config['img_size'],
            batch_size=self.config['batch_size'],
            stride=32,
            single_cls=True,
            hyp=hyp,
            augment=True,
            cache='ram' if gpu_mem > 20 else False,  # 只在大内存时缓存
            rect=False,
            rank=-1,
            workers=2,  # Colab CPU限制
            image_weights=False,
            quad=False,
            prefix='train: ',
            shuffle=True
        )
        
        # 验证数据加载器
        val_loader = create_dataloader(
            path=data_root / 'images/val',
            imgsz=self.config['img_size'],
            batch_size=self.config['batch_size'] * 2,
            stride=32,
            single_cls=True,
            hyp=hyp,
            augment=False,
            cache='ram' if gpu_mem > 20 else False,
            rect=True,
            rank=-1,
            workers=2,
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
            'warmup_epochs': 3.0,
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
    
    def train_epoch(self, model, train_loader, optimizer, compute_loss, epoch, epochs):
        """训练一个epoch（内存优化）"""
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        
        losses = []
        accumulation_steps = self.config.get('gradient_accumulation', 1)
        
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            
            # 前向传播
            with torch.amp.autocast('cuda', enabled=self.config.get('amp', True)):
                # 直接使用YOLOv9模型
                pred = model(imgs)
                
                # 第一个batch时打印调试信息
                if i == 0:
                    self.logger.info(f"Raw model output type: {type(pred)}")
                    self.logger.info(f"Model training mode: {model.training}")
                    
                loss, loss_items = compute_loss(pred, targets.to(self.device))
                loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新进度条
            losses.append(loss_items.cpu().numpy())
            if len(losses) > 100:
                losses = losses[-100:]  # 限制内存使用
            
            mean_loss = np.mean(losses, axis=0)
            pbar.set_postfix({
                'loss': f'{mean_loss[0]:.4f}',
                'GPU': f'{torch.cuda.memory_reserved() / 1024**3:.1f}G'
            })
            
            # 定期清理显存
            if i % 100 == 0:
                torch.cuda.empty_cache()
        
        return np.mean(losses, axis=0)
    
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
        
        # 恢复优化器状态
        if self.config.get('resume') and start_epoch > 0:
            checkpoint = torch.load(self.config['resume'], map_location=self.device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 创建调度器
        scheduler = self.create_scheduler(optimizer, self.config['epochs'] - start_epoch)
        
        # 创建EMA
        ema = ModelEMA(model) if self.config.get('use_ema', True) else None
        
        # 训练循环
        best_fitness = 0
        patience_counter = 0
        patience = self.config.get('patience', 8)
        
        try:
            for epoch in range(start_epoch, self.config['epochs']):
                # 训练
                train_loss = self.train_epoch(model, train_loader, optimizer, compute_loss, 
                                            epoch, self.config['epochs'])
                
                # 验证
                val_loss = self.validate(model, val_loader, compute_loss)
                
                # 更新EMA
                if ema:
                    ema.update(model)
                
                # 更新学习率
                scheduler.step()
                
                # 记录日志
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['epochs']} - "
                    f"Train Loss: {train_loss[0]:.4f} - "
                    f"Val Loss: {val_loss[0]:.4f} - "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} - "
                    f"Patience: {patience_counter}/{patience}"
                )
                
                # 保存检查点
                if (epoch + 1) % self.config['save_period'] == 0:
                    self.save_checkpoint(model, optimizer, epoch, best_fitness, ema)
                
                # 保存最佳模型和早停检查
                fitness = 1 / (val_loss[0] + 1e-6)
                if fitness > best_fitness:
                    best_fitness = fitness
                    patience_counter = 0
                    self.save_checkpoint(model, optimizer, epoch, best_fitness, ema, best=True)
                    self.logger.info(f"New best model! Fitness: {fitness:.6f}")
                else:
                    patience_counter += 1
                    
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
        """创建学习率调度器"""
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
        """验证模型"""
        model.eval()
        losses = []
        
        with torch.no_grad():
            for imgs, targets, paths, _ in tqdm(val_loader, desc='Validating'):
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                
                with torch.amp.autocast('cuda', enabled=self.config.get('amp', True)):
                    pred = model(imgs)
                    loss, loss_items = compute_loss(pred, targets.to(self.device))
                
                losses.append(loss_items.cpu().numpy())
        
        return np.mean(losses, axis=0)


def main():
    """主函数 - Colab版本"""
    import time
    
    config = {
        # 模型配置
        'model_cfg': 'yolov9c-face-exact.yaml',  # 优先使用修正的配置，回退到标准配置
        'pretrained_weights': 'yolov9c_face_weights_only.pt',  # 会自动在多个位置查找
        'num_au': 32,
        'num_emotion': 6,
        
        # 数据配置
        'img_size': 640,
        'batch_size': 8,  # Colab T4 GPU优化
        'workers': 2,     # Colab CPU限制
        'cache_images': False,  # 节省内存
        
        # 训练配置
        'epochs': 40,
        'lr0': 0.002,
        'optimizer': 'AdamW',
        'scheduler': 'cosine',
        'device': 'cuda:0',
        'use_ema': True,
        'save_period': 5,  # 更频繁保存
        'amp': True,  # 混合精度训练
        'gradient_accumulation': 2,  # 梯度累积
        'patience': 8,  # 早停机制：8个epoch验证损失不下降就停止
        
        # 保存配置
        'save_dir': '/content/runs/stage1_hda_synchild',
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