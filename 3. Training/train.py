#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                           precision_recall_fscore_support, recall_score)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on effective number of samples."""
    
    def __init__(self, samples_per_class: List[int], beta: float = 0.9999, 
                 gamma: float = 2.0, loss_type: str = "focal"):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.weights = self.weights.to(inputs.device)
        
        if self.loss_type == "focal":
            return FocalLoss(alpha=self.weights, gamma=self.gamma)(inputs, targets)
        elif self.loss_type == "ce":
            return F.cross_entropy(inputs, targets, weight=self.weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class BalancedBatchSampler:
    """Custom sampler that ensures balanced batches."""
    
    def __init__(self, labels: List[int], batch_size: int, classes_per_batch: int = None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(np.unique(labels))
        self.classes_per_batch = classes_per_batch or min(self.num_classes, batch_size)
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // self.classes_per_batch
        
    def __iter__(self):
        # Shuffle indices within each class
        for class_indices in self.class_indices.values():
            random.shuffle(class_indices)
        
        # Create class iterators
        class_iters = {}
        for class_id, indices in self.class_indices.items():
            class_iters[class_id] = iter(indices * (len(indices) // len(indices) + 2))
        
        # Generate balanced batches
        while True:
            batch = []
            classes_to_sample = random.sample(list(self.class_indices.keys()), 
                                            self.classes_per_batch)
            
            try:
                for class_id in classes_to_sample:
                    for _ in range(self.samples_per_class):
                        batch.append(next(class_iters[class_id]))
                
                # Fill remaining slots randomly
                while len(batch) < self.batch_size:
                    random_class = random.choice(classes_to_sample)
                    batch.append(next(class_iters[random_class]))
                
                random.shuffle(batch)
                yield batch
                
            except StopIteration:
                break
    
    def __len__(self):
        return min(len(indices) for indices in self.class_indices.values()) * self.num_classes // self.batch_size


class InteractionAttention(nn.Module):
    """Enhanced Inter-hand attention with class-aware features."""
    
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.channels_per_head = channels // num_heads
        self.qkv_left = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.qkv_right = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        
        # Enhanced feature processing
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        
        # Enhanced feature processing
        x_enhanced = self.feature_enhance(x)
        
        # Split into left and right hands
        x_left = x_enhanced[:, :, :, :21]
        x_right = x_enhanced[:, :, :, 21:]
        
        # Generate QKV for both hands
        QKV_left = self.qkv_left(x_left)
        Q_left, K_left, V_left = torch.chunk(QKV_left, 3, dim=1)
        
        QKV_right = self.qkv_right(x_right)
        Q_right, K_right, V_right = torch.chunk(QKV_right, 3, dim=1)
        
        # Reshape for multi-head attention
        Q_left = Q_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        K_left = K_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        V_left = V_left.view(N, self.num_heads, self.channels_per_head, T, 21)
        Q_right = Q_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        K_right = K_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        V_right = V_right.view(N, self.num_heads, self.channels_per_head, T, 21)
        
        # Cross-attention computations
        d_k = Q_right.size(2)
        
        # Right-to-left attention
        attn_right_to_left = torch.matmul(
            Q_right.view(N, self.num_heads, T * 21, -1),
            K_left.view(N, self.num_heads, T * 21, -1).transpose(-2, -1)
        ) / (d_k ** 0.5)
        attn_right_to_left = F.softmax(attn_right_to_left, dim=-1)
        out_right_to_left = torch.matmul(
            attn_right_to_left, 
            V_left.view(N, self.num_heads, T * 21, -1)
        ).view(N, self.num_heads, T, 21, -1).permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)
        
        # Left-to-right attention
        attn_left_to_right = torch.matmul(
            Q_left.view(N, self.num_heads, T * 21, -1),
            K_right.view(N, self.num_heads, T * 21, -1).transpose(-2, -1)
        ) / (d_k ** 0.5)
        attn_left_to_right = F.softmax(attn_left_to_right, dim=-1)
        out_left_to_right = torch.matmul(
            attn_left_to_right,
            V_right.view(N, self.num_heads, T * 21, -1)
        ).view(N, self.num_heads, T, 21, -1).permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, 21)
        
        # Concatenate and process
        out = torch.cat([out_left_to_right, out_right_to_left], dim=3)
        out = out.view(N, C, T, V)
        
        # Enhanced residual connection
        residual = out
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = out + self.residual(residual + x)
        
        return out


class GraphBase:
    """Enhanced graph structure with better connectivity."""
    
    def __init__(self, hop_size: int):
        self.hop_size = hop_size
        self.num_node = 42  # 21 keypoints per hand
        self.get_edge()
        self.hop_dis = self.get_hop_distance()
        self.get_adjacency()
    
    def get_edge(self):
        """Define enhanced hand skeleton connections."""
        self_link = [(i, i) for i in range(self.num_node)]
        
        # MediaPipe hand connections (0-indexed)
        hand_connections = [
            (0,1),(1,2),(2,3),(3,4),    # Thumb
            (0,5),(5,6),(6,7),(7,8),    # Index
            (0,9),(9,10),(10,11),(11,12),   # Middle
            (0,13),(13,14),(14,15),(15,16), # Ring
            (0,17),(17,18),(18,19),(19,20)  # Pinky
        ]
        
        # Apply to both hands
        neighbor_link = []
        for offset in [0, 21]:  # Left hand (0-20), Right hand (21-41)
            for i, j in hand_connections:
                neighbor_link.extend([(i+offset, j+offset), (j+offset, i+offset)])
        
        # Inter-hand connections (corresponding fingers)
        inter_hand = []
        for i in range(21):
            inter_hand.extend([(i, i+21), (i+21, i)])
        
        self.edge = self_link + neighbor_link + inter_hand
    
    def get_hop_distance(self) -> np.ndarray:
        """Calculate hop distances with better connectivity."""
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        
        for d in range(self.hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis
    
    def get_adjacency(self):
        """Build normalized adjacency matrices."""
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A
    
    @staticmethod
    def normalize_digraph(A: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix."""
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        return np.dot(A, Dn)


class SpatialGraphConvolution(nn.Module):
    """Enhanced spatial graph convolution with residual connections."""
    
    def __init__(self, in_channels: int, out_channels: int, s_kernel_size: int):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * s_kernel_size,
            kernel_size=1
        )
        
        # Add residual connection if dimensions match
        self.residual = None
        if in_channels == out_channels:
            self.residual = nn.Identity()
        elif in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual is None else self.residual(x)
        
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        # Add residual if dimensions match
        if residual.shape == x.shape:
            x = x + residual
        
        return x.contiguous()


class STGCBlock(nn.Module):
    """Enhanced STGC block with better regularization."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 t_kernel_size: int, A_size: Tuple, dropout: float = 0.3):
        super().__init__()
        
        self.sgc = SpatialGraphConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            s_kernel_size=A_size[0]
        )
        
        self.M = nn.Parameter(torch.ones(A_size))
        
        # Enhanced temporal convolution with residual
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels, out_channels,
                (t_kernel_size, 1), (stride, 1),
                ((t_kernel_size - 1) // 2, 0)
            ),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection for temporal dimension
        self.temporal_residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1), (stride, 1)),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 else nn.Identity()
        
        self.attn = InteractionAttention(out_channels, dropout=dropout)
        self.final_relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Spatial-temporal convolution
        out = self.sgc(x, A * self.M)
        
        # Temporal convolution with residual
        temporal_out = self.tgc(out)
        temporal_residual = self.temporal_residual(out)
        out = self.final_relu(temporal_out + temporal_residual)
        
        # Attention mechanism
        out = self.attn(out) + out
        
        return out


class EnhancedSTGCN(nn.Module):
    """Enhanced STGCN with better handling of class imbalance."""
    
    def __init__(self, num_classes: int, in_channels: int, 
                 t_kernel_size: int = 9, hop_size: int = 2,
                 dropout: float = 0.3, use_class_attention: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_class_attention = use_class_attention
        
        # Build enhanced graph adjacency matrix
        graph = GraphBase(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        
        # Input normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        
        # Progressive channel expansion with better regularization
        self.stgc1 = STGCBlock(in_channels, 64, 1, t_kernel_size, A_size, dropout)
        self.stgc2 = STGCBlock(64, 64, 1, t_kernel_size, A_size, dropout)
        self.stgc3 = STGCBlock(64, 128, 1, t_kernel_size, A_size, dropout)
        self.stgc4 = STGCBlock(128, 128, 2, t_kernel_size, A_size, dropout)
        self.stgc5 = STGCBlock(128, 256, 1, t_kernel_size, A_size, dropout)
        self.stgc6 = STGCBlock(256, 256, 1, t_kernel_size, A_size, dropout)
        
        # Enhanced classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Class-aware attention mechanism
        if use_class_attention:
            self.class_attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
                nn.Softmax(dim=1)
            )
        
        # Final classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with better strategy for imbalanced data."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    # Initialize bias for imbalanced classes
                    if m.out_features == self.num_classes:
                        # Bias initialization for class imbalance
                        nn.init.constant_(m.bias, -2.0)  # Start with low confidence
                    else:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        
        # Input normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        # Progressive feature extraction
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        
        # Global feature extraction
        x = self.global_pool(x)  # (N, 256, 1, 1)
        x = x.view(N, -1)  # (N, 256)
        
        # Class-aware attention (optional)
        if self.use_class_attention:
            attention_weights = self.class_attention(x)
            x = x * attention_weights.sum(dim=1, keepdim=True)  # Weighted features
        
        # Final classification
        x = self.classifier(x)
        
        return x


class PSKUSDataset(Dataset):
    """Enhanced dataset with better data augmentation for minority classes."""
    
    def __init__(self, h5_path: str, items: List, use_z: bool = False,
                 xy_norm: str = "minus1_1", z_norm: str = "zscore",
                 augment_minority: bool = True, minority_threshold: int = 5000):
        self.h5_path = h5_path
        self.items = items
        self.use_z = use_z
        self.xy_norm = xy_norm
        self.z_norm = z_norm
        self.augment_minority = augment_minority
        self.minority_threshold = minority_threshold
        self.h5_file = None
        
        # Identify minority classes for augmentation
        if augment_minority:
            self._identify_minority_classes()
    
    def _identify_minority_classes(self):
        """Identify minority classes that need augmentation."""
        class_counts = Counter([item[-1] for item in self.items])
        self.minority_classes = {
            cls for cls, count in class_counts.items() 
            if count < self.minority_threshold and cls != 0  # Don't augment class 0
        }
        logger.info(f"Minority classes for augmentation: {self.minority_classes}")
    
    def __len__(self) -> int:
        return len(self.items)
    
    def _augment_keypoints(self, xyz: np.ndarray, class_label: int) -> np.ndarray:
        """Apply data augmentation for minority classes."""
        if not self.augment_minority or class_label not in self.minority_classes:
            return xyz
        
        # Random transformations for minority classes
        augmented = xyz.copy()
        
        # 1. Random temporal jittering
        if random.random() < 0.3:
            shift = random.randint(-2, 2)
            if shift != 0:
                if shift > 0:
                    augmented[shift:] = augmented[:-shift]
                    augmented[:shift] = augmented[shift:shift+1]  # Repeat first frames
                else:
                    augmented[:shift] = augmented[-shift:]
                    augmented[shift:] = augmented[shift-1:shift]  # Repeat last frames
        
        # 2. Random spatial scaling (slight)
        if random.random() < 0.4:
            scale_factor = random.uniform(0.95, 1.05)
            augmented[..., :2] *= scale_factor
        
        # 3. Random noise injection
        if random.random() < 0.2:
            noise_std = 0.01
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented += noise
        
        # 4. Random temporal masking
        if random.random() < 0.1:
            mask_length = random.randint(1, 3)
            start_idx = random.randint(0, max(0, augmented.shape[0] - mask_length))
            # Replace with interpolation
            if start_idx > 0 and start_idx + mask_length < augmented.shape[0]:
                for i in range(mask_length):
                    alpha = (i + 1) / (mask_length + 1)
                    augmented[start_idx + i] = (
                        (1 - alpha) * augmented[start_idx - 1] + 
                        alpha * augmented[start_idx + mask_length]
                    )
        
        return augmented
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        
        vid, t0, t_win, W, H, fps, dsid, camid, y = self.items[idx]
        g = self.h5_file["videos"][vid]
        T = g["data"].shape[0]
        t1 = min(t0 + t_win, T)
        
        xyz = g["data"][t0:t1]  # (t, 42, 3)
        
        # Pad if necessary
        if xyz.shape[0] < t_win:
            pad = np.zeros((t_win - xyz.shape[0], 42, xyz.shape[2]), dtype=xyz.dtype)
            xyz = np.concatenate([xyz, pad], axis=0)
        
        # Apply augmentation for minority classes
        xyz = self._augment_keypoints(xyz, y)
        
        # Normalize coordinates
        if self.xy_norm == "minus1_1":
            xyz[..., 0] = 2.0 * (xyz[..., 0] / (W + 1e-6)) - 1.0
            xyz[..., 1] = 2.0 * (xyz[..., 1] / (H + 1e-6)) - 1.0
        
        if self.use_z and xyz.shape[-1] >= 3 and self.z_norm == "zscore":
            z = xyz[..., 2]
            m, s = z.mean(), z.std() or 1.0
            xyz[..., 2] = (z - m) / s
        
        # Select channels and transpose to (C, T, V)
        C = 3 if (self.use_z and xyz.shape[-1] >= 3) else 2
        X = np.transpose(xyz[..., :C], (2, 0, 1))
        
        return torch.from_numpy(X.astype(np.float32)), torch.tensor(y, dtype=torch.long)
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_sampler(labels: List[int], strategy: str = "oversample") -> WeightedRandomSampler:
    """Create balanced sampler with different strategies."""
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    if strategy == "oversample":
        # Oversample minority classes to match majority
        max_count = max(class_counts.values())
        weights = [max_count / class_counts[label] for label in labels]
    elif strategy == "undersample":
        # Weight to undersample majority classes
        min_count = min(class_counts.values())
        weights = [min_count / class_counts[label] for label in labels]
    elif strategy == "balanced":
        # Balanced weighting
        weights = [total_samples / (num_classes * class_counts[label]) for label in labels]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def load_data(h5_path: str, t_win: int, stride: int, val_fraction: float, 
              test_datasets: List[str], seed: int) -> Tuple:
    """Load and split dataset - placeholder implementation."""
    logger.info(f"Loading data from {h5_path}")
    
    # Mock implementation - replace with actual window indexing logic
    np.random.seed(seed)
    
    # Simulate realistic class distribution based on the image
    class_probs = [0.85, 0.025, 0.043, 0.020, 0.023, 0.026, 0.013]  # Based on the chart
    
    def generate_items(n_samples: int, prefix: str):
        items = []
        for i in range(n_samples):
            class_label = np.random.choice(7, p=class_probs)
            items.append((f"{prefix}_vid_{i}", 0, t_win, 320, 240, 30, 
                         "DataSet1", "cam1", class_label))
        return items
    
    train_items = generate_items(10000, "train")
    val_items = generate_items(2000, "val") 
    test_items = generate_items(1000, "test")
    
    # Log class distribution
    train_labels = [item[-1] for item in train_items]
    train_counts = Counter(train_labels)
    logger.info(f"Train class distribution: {dict(train_counts)}")
    
    return train_items, val_items, test_items


def build_model(num_classes: int, in_channels: int, device: torch.device,
                model_config: Dict) -> nn.Module:
    """Build enhanced model with better architecture."""
    model = EnhancedSTGCN(
        num_classes=num_classes,
        in_channels=in_channels,
        t_kernel_size=model_config.get('t_kernel_size', 9),
        hop_size=model_config.get('hop_size', 2),
        dropout=model_config.get('dropout', 0.3),
        use_class_attention=model_config.get('use_class_attention', True)
    )
    model = model.to(device)
    return model


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   epoch: int, log_interval: int) -> Dict[str, float]:
    """Enhanced training with better monitoring."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Per-class accuracy tracking
        for i in range(len(target)):
            label = target[i].item()
            class_total[label] += 1
            if pred[i] == target[i]:
                class_correct[label] += 1
        
        if batch_idx % log_interval == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                       f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {
        cls: class_correct[cls] / max(1, class_total[cls]) 
        for cls in class_total.keys()
    }
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'time': epoch_time,
        'class_accuracies': class_accuracies
    }


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
            device: torch.device, num_classes: int) -> Dict[str, Union[float, Dict]]:
    """Enhanced evaluation with detailed metrics."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Per-class tracking
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i] == target[i]:
                    class_correct[label] += 1
    
    avg_loss = test_loss / len(dataloader)
    accuracy = correct / total
    
    # Calculate comprehensive metrics
    labels_eval = list(range(1, num_classes)) if num_classes > 2 else list(range(num_classes))
    
    # F1 scores
    f1_macro = f1_score(all_targets, all_preds, labels=labels_eval, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, labels=labels_eval, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, labels=labels_eval, average=None, zero_division=0)
    
    # Precision and Recall
    precision, recall, _, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_eval, average=None, zero_division=0
    )
    
    # Per-class accuracies
    class_accuracies = {
        cls: class_correct[cls] / max(1, class_total[cls]) 
        for cls in range(num_classes)
    }
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': dict(zip(labels_eval, f1_per_class)),
        'precision_per_class': dict(zip(labels_eval, precision)),
        'recall_per_class': dict(zip(labels_eval, recall)),
        'class_accuracies': class_accuracies,
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist()
    }


def plot_training_curves(metrics_history: List[Dict], save_dir: str):
    """Plot comprehensive training curves."""
    epochs = [m['epoch'] for m in metrics_history]
    train_losses = [m['train_loss'] for m in metrics_history]
    val_losses = [m['val_loss'] for m in metrics_history]
    train_accs = [m['train_acc'] for m in metrics_history]
    val_accs = [m['val_acc'] for m in metrics_history]
    val_f1s = [m['val_f1_macro'] for m in metrics_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, train_accs, label='Train Acc', color='blue')
    axes[0, 1].plot(epochs, val_accs, label='Val Acc', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(epochs, val_f1s, label='Val F1 (Macro)', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score (Macro)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    lrs = [m.get('lr', 0) for m in metrics_history]
    axes[1, 1].plot(epochs, lrs, label='Learning Rate', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_dir: str):
    """Plot confusion matrix."""
    cm = np.array(cm)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Enhanced LAIA-Net Training with Class Balancing')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--t_win', type=int, default=64,
                       help='Temporal window size')
    parser.add_argument('--stride', type=int, default=8,
                       help='Stride for sliding window')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                       help='Validation split fraction')
    parser.add_argument('--test_datasets', type=str, nargs='*', default=[],
                       help='Dataset IDs to use for testing')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='enhanced_stgcn',
                       choices=['enhanced_stgcn'], help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=2,
                       help='Number of input channels (2 for XY, 3 for XYZ)')
    parser.add_argument('--use_z', action='store_true',
                       help='Use Z coordinate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--use_class_attention', action='store_true', default=True,
                       help='Use class attention mechanism')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'], help='Learning rate scheduler')
    
    # Class balancing parameters
    parser.add_argument('--loss_type', type=str, default='class_balanced',
                       choices=['ce', 'focal', 'class_balanced'], 
                       help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--cb_beta', type=float, default=0.9999,
                       help='Class-balanced loss beta parameter')
    parser.add_argument('--sampling_strategy', type=str, default='balanced',
                       choices=['none', 'oversample', 'undersample', 'balanced'],
                       help='Sampling strategy for class balancing')
    parser.add_argument('--augment_minority', action='store_true', default=True,
                       help='Apply data augmentation to minority classes')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log interval for training')
    parser.add_argument('--save_dir', type=str, default='./outputs_enhanced',
                       help='Directory to save outputs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='laia-net-enhanced',
                       help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Wandb run name')
    
    # Plotting parameters
    parser.add_argument('--plot_local', action='store_true', default=True,
                       help='Save plots locally')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    if args.plot_local:
        os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Wandb not available. Install with: pip install wandb")
            args.use_wandb = False
        else:
            try:
                wandb.init(
                    project=args.project_name,
                    name=args.run_name,
                    config=vars(args),
                    tags=['class-imbalance', 'enhanced-stgcn']
                )
                logger.info("Wandb initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb.")
                args.use_wandb = False
    
    # Adjust input channels based on use_z
    if args.use_z:
        args.in_channels = 3
    
    # Load data
    train_items, val_items, test_items = load_data(
        args.data_path, args.t_win, args.stride, args.val_fraction,
        args.test_datasets, args.seed
    )
    
    # Create datasets
    train_dataset = PSKUSDataset(
        args.data_path, train_items, use_z=args.use_z,
        augment_minority=args.augment_minority
    )
    val_dataset = PSKUSDataset(args.data_path, val_items, use_z=args.use_z, augment_minority=False)
    test_dataset = PSKUSDataset(args.data_path, test_items, use_z=args.use_z, augment_minority=False)
    
    # Create balanced sampler if requested
    train_labels = [item[-1] for item in train_items]
    if args.sampling_strategy != 'none':
        sampler = create_balanced_sampler(train_labels, args.sampling_strategy)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Build model
    model_config = {
        'dropout': args.dropout,
        'use_class_attention': args.use_class_attention
    }
    model = build_model(args.num_classes, args.in_channels, device, model_config)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create loss function based on class distribution
    class_counts = Counter(train_labels)
    samples_per_class = [class_counts[i] for i in range(args.num_classes)]
    
    if args.loss_type == 'focal':
        # Calculate class weights for focal loss
        total = sum(samples_per_class)
        weights = [total / (args.num_classes * count) for count in samples_per_class]
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma)
    elif args.loss_type == 'class_balanced':
        criterion = ClassBalancedLoss(
            samples_per_class=samples_per_class,
            beta=args.cb_beta,
            gamma=args.focal_gamma,
            loss_type='focal'
        )
    else:  # Standard cross-entropy
        weights = torch.tensor([sum(samples_per_class) / (args.num_classes * count) 
                               for count in samples_per_class], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    logger.info(f"Using {args.loss_type} loss with class distribution: {dict(class_counts)}")
    
    # Optimizer with different learning rates for different parts
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': args.lr * 2},
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training loop with early stopping
    best_val_f1 = 0.0
    early_stopping_counter = 0
    metrics_history = []
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.log_interval
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['f1_macro'])
        else:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'train_time': train_metrics['time'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            'lr': current_lr
        }
        
        metrics_history.append(epoch_metrics)
        
        # Detailed logging
        logger.info(f'Epoch {epoch:03d} | '
                   f'Train: {train_metrics["loss"]:.4f}/{train_metrics["accuracy"]:.3f} | '
                   f'Val: {val_metrics["loss"]:.4f}/{val_metrics["accuracy"]:.3f} | '
                   f'F1: {val_metrics["f1_macro"]:.4f} | LR: {current_lr:.2e}')
        
        # Log per-class metrics
        logger.info(f'Val per-class F1: {val_metrics["f1_per_class"]}')
        
        # Wandb logging
        if args.use_wandb:
            log_dict = epoch_metrics.copy()
            log_dict.update({f'val_f1_class_{k}': v for k, v in val_metrics['f1_per_class'].items()})
            log_dict.update({f'val_acc_class_{k}': v for k, v in val_metrics['class_accuracies'].items()})
            wandb.log(log_dict)
        
        # Save best model and early stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            early_stopping_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'best_model.pt'))
            
            logger.info(f'New best model saved with F1: {best_val_f1:.4f}')
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch} epochs')
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, args.num_classes)
    
    logger.info(f'Test Results:')
    logger.info(f'  Loss: {test_metrics["loss"]:.4f}')
    logger.info(f'  Accuracy: {test_metrics["accuracy"]:.3f}')
    logger.info(f'  F1 (Macro): {test_metrics["f1_macro"]:.4f}')
    logger.info(f'  F1 (Weighted): {test_metrics["f1_weighted"]:.4f}')
    logger.info(f'  Per-class F1: {test_metrics["f1_per_class"]}')
    logger.info(f'  Per-class Accuracy: {test_metrics["class_accuracies"]}')
    
    # Save comprehensive results
    final_metrics = {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'total_epochs': epoch,
        'early_stopped': early_stopping_counter >= args.early_stopping_patience,
        'class_distribution': dict(Counter(train_labels)),
        'history': metrics_history,
        'args': vars(args)
    }
    
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Generate plots
    if args.plot_local:
        plot_training_curves(metrics_history, os.path.join(args.save_dir, 'plots'))
        plot_confusion_matrix(
            test_metrics['confusion_matrix'], 
            [f'Class {i}' for i in range(args.num_classes)],
            os.path.join(args.save_dir, 'plots')
        )
    
    # Final wandb logging
    if args.use_wandb:
        wandb.log({
            "test_f1_macro": test_metrics['f1_macro'],
            "test_f1_weighted": test_metrics['f1_weighted'], 
            "test_accuracy": test_metrics['accuracy'],
            "final_epoch": epoch
        })
        
        # Upload plots if available
        if args.plot_local:
            wandb.log({
                "training_curves": wandb.Image(os.path.join(args.save_dir, 'plots', 'training_curves.png')),
                "confusion_matrix": wandb.Image(os.path.join(args.save_dir, 'plots', 'confusion_matrix.png'))
            })
        
        wandb.finish()
    
    logger.info(f"Training completed. Results saved to {args.save_dir}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"Final test F1: {test_metrics['f1_macro']:.4f}")


if __name__ == '__main__':
    main()