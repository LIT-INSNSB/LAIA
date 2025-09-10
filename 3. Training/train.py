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
                 dropout: float = 0.3, use_class_attention: bool = True,
                 return_features: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_class_attention = use_class_attention
        self.return_features = return_features
        
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
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 256),
                nn.Sigmoid()
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
            gate = self.class_attention(x)   # (N,256) ∈ [0,1]
            x = x * gate
        
        # Return features if requested (for two-stage training)
        if self.return_features:
            return x
        
        # Final classification
        x = self.classifier(x)
        
        return x


# ============================================================================
# TWO-STAGE MODELS AND UTILITIES
# ============================================================================

class BinaryActivityModel(nn.Module):
    """Detector binario: actividad (1) vs fondo (0)"""
    
    def __init__(self, backbone_model, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone_model
        
        # Congelar backbone si se desea
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Obtener dimensión del backbone (asumiendo que tiene 256 features)
        backbone_dim = 256
        
        # Cabeza binaria
        self.binary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Salida binaria
        )
        
        # Modificar backbone para devolver features
        self.backbone.return_features = True
    
    def forward(self, x):
        features = self.backbone(x)  # (N, 256)
        logits = self.binary_head(features)  # (N, 1)
        return logits.squeeze(-1)  # (N,)

class LogitAdjustedClassifier(nn.Module):
    """Clasificador con ajuste de logits por prior de clase"""
    
    def __init__(self, in_features, num_classes, class_frequencies, tau=1.0):
        super().__init__()
        self.tau = tau
        
        # Cabeza estándar
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Registro del ajuste de logits (log prior)
        if class_frequencies is not None:
            log_prior = torch.log(class_frequencies / class_frequencies.max())
            self.register_buffer('logit_adjustment', log_prior)
        else:
            self.register_buffer('logit_adjustment', torch.zeros(num_classes))
    
    def forward(self, x, apply_adjustment=True):
        logits = self.classifier(x)
        
        if apply_adjustment and self.training:
            # Solo aplicar durante entrenamiento
            adjusted_logits = logits + self.tau * self.logit_adjustment
            return adjusted_logits
        
        return logits

class MultiClassModel(nn.Module):
    """Clasificador 6-clases con logit adjustment"""
    
    def __init__(self, backbone_model, class_frequencies=None, tau=1.0, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone_model
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Dimensión del backbone
        backbone_dim = 256
        
        # Cabeza multiclase
        self.multiclass_head = LogitAdjustedClassifier(
            backbone_dim, 6, class_frequencies, tau  # 6 clases (1-6 remapeadas a 0-5)
        )
        
        # Modificar backbone
        self.backbone.return_features = True
    
    def forward(self, x, apply_adjustment=True):
        features = self.backbone(x)
        return self.multiclass_head(features, apply_adjustment)

class TwoStageInference:
    """Pipeline de inferencia para el modelo de 2 etapas"""
    
    def __init__(self, binary_model, multiclass_model, threshold=0.5, device='cuda'):
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.threshold = threshold
        self.device = device
        
        self.binary_model.eval()
        self.multiclass_model.eval()
    
    def predict(self, x):
        """Predicción de 2 etapas"""
        batch_size = x.size(0)
        x = x.to(self.device)
        
        # Etapa 1: Detección de actividad
        with torch.no_grad():
            binary_logits = self.binary_model(x)
            activity_probs = torch.sigmoid(binary_logits)
            is_activity = activity_probs >= self.threshold
        
        # Inicializar predicciones como fondo (0)
        final_preds = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Etapa 2: Clasificación para samples con actividad
        if is_activity.any():
            activity_indices = torch.where(is_activity)[0]
            activity_samples = x[activity_indices]
            
            with torch.no_grad():
                multiclass_logits = self.multiclass_model(activity_samples, apply_adjustment=False)
                multiclass_preds = multiclass_logits.argmax(dim=1) + 1  # 0-5 → 1-6
            
            final_preds[activity_indices] = multiclass_preds
        
        return final_preds.cpu()

# ============================================================================
# VALIDATION STRATIFICATION AND MONITORING UTILITIES
# ============================================================================

def create_balanced_validation(val_items, target_bg_ratio=0.3, seed=42):
    """Crea validación balanceada limitando clase 0 al target_bg_ratio"""
    rng = np.random.RandomState(seed)
    
    # Separar por clase
    bg_items = [it for it in val_items if it[-1] == 0]
    pos_items = [it for it in val_items if it[-1] > 0]
    
    logger.info(f"Val original: {len(bg_items)} bg, {len(pos_items)} positive")
    
    # Calcular cuántos bg necesitamos
    bg_needed = int(target_bg_ratio * len(pos_items) / (1 - target_bg_ratio))
    bg_needed = min(bg_needed, len(bg_items))
    
    # Muestrear bg items
    if bg_needed < len(bg_items):
        selected_bg = rng.choice(len(bg_items), size=bg_needed, replace=False)
        bg_selected = [bg_items[i] for i in selected_bg]
    else:
        bg_selected = bg_items
    
    # Combinar y mezclar
    val_bal_items = pos_items + bg_selected
    rng.shuffle(val_bal_items)
    
    final_bg_ratio = len(bg_selected) / len(val_bal_items)
    logger.info(f"Val balanced: {len(bg_selected)} bg, {len(pos_items)} pos "
                f"(bg_ratio={final_bg_ratio:.2f})")
    
    return val_bal_items, val_items

def convert_labels_to_binary(labels):
    """Convierte labels [0,1,2,3,4,5,6] → [0,1,1,1,1,1,1]"""
    return (np.array(labels) > 0).astype(np.int64)

def filter_to_positive_classes(items, labels, class_mapping=None):
    """Filtra items para mantener solo clases 1-6 y las remapea a 0-5"""
    if class_mapping is None:
        class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    
    filtered_items = []
    remapped_labels = []
    
    for item, label in zip(items, labels):
        if label > 0:
            filtered_items.append(item)
            remapped_labels.append(class_mapping.get(label, label-1))
    
    logger.info(f"Filtered {len(items)} → {len(filtered_items)} positive samples")
    logger.info(f"Label distribution: {Counter(remapped_labels)}")
    
    return filtered_items, remapped_labels

class DetailedMetricsTracker:
    """Sistema de tracking de métricas detallado"""
    
    def __init__(self, num_classes, save_dir, patience_zero_recall=4, dominance_threshold=0.7):
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.patience_zero_recall = patience_zero_recall
        self.dominance_threshold = dominance_threshold
        
        # Historial
        self.recall_history = defaultdict(list)  # {class: [recall_by_epoch]}
        self.pred_distribution_history = []  # [distribution_by_epoch]
        self.zero_recall_epochs = defaultdict(int)  # {class: consecutive_epochs}
        
        # Alertas
        self.should_stop_collapse = False
        self.collapse_reason = ""
    
    def update(self, epoch, all_targets, all_preds, val_metrics):
        """Actualizar métricas y revisar condiciones de parada"""
        
        # 1. Tracking de recall por clase
        for cls in range(1, self.num_classes):  # Solo clases 1-6
            cls_recall = val_metrics.get('class_recalls', {}).get(cls, 0.0)
            self.recall_history[cls].append(cls_recall)
            
            # Contador de épocas con recall = 0
            if cls_recall == 0.0:
                self.zero_recall_epochs[cls] += 1
            else:
                self.zero_recall_epochs[cls] = 0
            
            # Alerta si una clase está muerta por mucho tiempo
            if self.zero_recall_epochs[cls] >= self.patience_zero_recall:
                self.should_stop_collapse = True
                self.collapse_reason = f"Clase {cls} con recall=0 por {self.zero_recall_epochs[cls]} épocas"
        
        # 2. Distribución de predicciones
        pred_counts = Counter(all_preds)
        total_preds = len(all_preds)
        pred_distribution = {cls: pred_counts.get(cls, 0) / total_preds for cls in range(self.num_classes)}
        self.pred_distribution_history.append(pred_distribution)
        
        # 3. Detectar dominancia de una sola clase
        max_pred_ratio = max(pred_distribution.values())
        dominant_class = max(pred_distribution, key=pred_distribution.get)
        
        if max_pred_ratio > self.dominance_threshold and epoch > 3:
            # Revisar si domina por varias épocas
            recent_epochs = min(3, len(self.pred_distribution_history))
            dominance_count = sum(
                1 for hist in self.pred_distribution_history[-recent_epochs:]
                if max(hist.values()) > self.dominance_threshold and 
                   max(hist, key=hist.get) == dominant_class
            )
            
            if dominance_count >= 3:
                self.should_stop_collapse = True
                self.collapse_reason = f"Clase {dominant_class} domina >{self.dominance_threshold:.0%} por {dominance_count} épocas"
        
        # 4. Log detallado
        logger.info(f"=== EPOCH {epoch} DETAILED METRICS ===")
        logger.info(f"Recall por clase: {dict(val_metrics.get('class_recalls', {}))}")
        logger.info(f"Distribución predicciones: {dict(pred_distribution)}")
        logger.info(f"Épocas con recall=0: {dict(self.zero_recall_epochs)}")
        
        if max_pred_ratio > 0.5:
            logger.warning(f"⚠️  Clase {dominant_class} domina {max_pred_ratio:.1%} de predicciones")
        
        # 5. Graficar cada 5 épocas
        if epoch % 5 == 0:
            self._plot_distributions(epoch)
        
        return self.should_stop_collapse, self.collapse_reason
    
    def _plot_distributions(self, epoch):
        """Graficar evolución de distribuciones de predicción"""
        try:
            if len(self.pred_distribution_history) < 2:
                return
                
            epochs_range = range(1, len(self.pred_distribution_history) + 1)
            
            plt.figure(figsize=(12, 8))
            
            # Plot distribución por clase
            for cls in range(self.num_classes):
                class_ratios = [hist.get(cls, 0) for hist in self.pred_distribution_history]
                plt.plot(epochs_range, class_ratios, label=f'Clase {cls}', marker='o', linewidth=2)
            
            plt.axhline(y=self.dominance_threshold, color='red', linestyle='--', 
                       label=f'Umbral dominancia ({self.dominance_threshold:.0%})')
            
            plt.xlabel('Época')
            plt.ylabel('Proporción de Predicciones')
            plt.title(f'Evolución Distribución de Predicciones - Época {epoch}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Guardar
            os.makedirs(os.path.join(self.save_dir, 'monitoring'), exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, 'monitoring', f'pred_dist_epoch_{epoch}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot recall por clase
            plt.figure(figsize=(12, 6))
            for cls in range(1, self.num_classes):  # Solo clases 1-6
                if cls in self.recall_history:
                    plt.plot(epochs_range, self.recall_history[cls], 
                           label=f'Clase {cls}', marker='o', linewidth=2)
            
            plt.xlabel('Época')
            plt.ylabel('Recall')
            plt.title(f'Evolución Recall por Clase - Época {epoch}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.save_dir, 'monitoring', f'recall_evolution_epoch_{epoch}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting distributions: {e}")

class PSKUSDataset(Dataset):
    def __init__(self, h5_path: str, items: List, use_z: bool = False,
                 xy_norm: str = "minus1_1", z_norm: str = "zscore",
                 augment_minority: bool = True, minority_threshold: int = 5000,
                 cache_mode: str = 'none', max_cache_gb: float = 2.0):
        self.h5_path = h5_path
        self.items = items
        self.use_z = use_z
        self.xy_norm = xy_norm
        self.z_norm = z_norm
        self.augment_minority = augment_minority
        self.minority_threshold = minority_threshold

        # --- cache en RAM ---
        self.cache_mode = cache_mode  # 'none' | 'lazy' | 'all'
        self.max_cache_bytes = int(max_cache_gb * (1024**3))
        self._ram: Dict[str, np.ndarray] = {}
        self._ram_bytes = 0

        self.h5_file = None

        if augment_minority:
            self._identify_minority_classes()

        # Precarga total (si se pide)
        if self.cache_mode == 'all':
            with h5py.File(self.h5_path, "r") as f:
                vg = f["videos"]
                for vid in vg.keys():
                    arr = vg[vid]["data"][:]  # (T, 42, C)
                    self._ram[vid] = arr
                    self._ram_bytes += arr.nbytes

    def _ensure_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def _maybe_cache(self, vid: str, arr: np.ndarray):
        if self.cache_mode == 'none':
            return
        if self.cache_mode == 'lazy':
            # cache bajo demanda con límite blando
            if self._ram_bytes + arr.nbytes <= self.max_cache_bytes:
                self._ram[vid] = arr
                self._ram_bytes += arr.nbytes

    def _get_slice(self, vid: str, t0: int, t1: int) -> np.ndarray:
        # Prioriza RAM
        if vid in self._ram:
            return self._ram[vid][t0:t1]

        # Si no está en RAM, lee de HDF5
        self._ensure_open()
        g = self.h5_file["videos"][vid]
        if self.cache_mode in ('lazy', 'all'):
            # cachea todo el vídeo si cabe (en lazy) o ya lo pediste en all
            full = g["data"][:]
            self._maybe_cache(vid, full)
            return full[t0:t1]
        else:
            return g["data"][t0:t1]
            
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
        # --- NUEVO: ya no abrimos aquí el HDF5; lo hace _get_slice si hace falta ---
        vid, t0, t_win, W, H, fps, dsid, camid, y = self.items[idx]
        t1 = t0 + t_win

        # Usa RAM si está cacheado; si no, lee de HDF5 y opcionalmente cachea (lazy/all)
        xyz = self._get_slice(vid, t0, t1)  # (t', 42, C)

        # Pad si hace falta (cuando estamos cerca del final del vídeo)
        if xyz.shape[0] < t_win:
            pad = np.zeros((t_win - xyz.shape[0], 42, xyz.shape[2]), dtype=xyz.dtype)
            xyz = np.concatenate([xyz, pad], axis=0)

        # Augment solo para clases minoritarias (igual que antes)
        xyz = self._augment_keypoints(xyz, y)

        # Normalización (igual que antes)
        if self.xy_norm == "minus1_1":
            xyz[..., 0] = 2.0 * (xyz[..., 0] / (W + 1e-6)) - 1.0
            xyz[..., 1] = 2.0 * (xyz[..., 1] / (H + 1e-6)) - 1.0

        if self.use_z and xyz.shape[-1] >= 3 and self.z_norm == "zscore":
            z = xyz[..., 2]
            m, s = z.mean(), z.std() or 1.0
            xyz[..., 2] = (z - m) / s

        # Selección de canales y (C, T, V) (igual que antes)
        C = 3 if (self.use_z and xyz.shape[-1] >= 3) else 2
        X = np.transpose(xyz[..., :C], (2, 0, 1))  # (C, T, V=42)

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

def to_int(x, default=0):
    """Convert to int safely."""
    try: 
        return int(x)
    except Exception:
        try: 
            return int(float(x))
        except Exception: 
            return default

def window_label_movement_strict(g, t0, t1, require_isw2=True, ignore_transitions=True):
    """Get label for window [t0:t1] with strict filtering."""
    mov = g["movement"][t0:t1]
    if (mov < 0).any():  # Filter out -1 values
        return None
    
    if require_isw2 and "is_washing" in g:
        isw = g["is_washing"][t0:t1]
        if not (isw == 2).all():
            return None
    
    if ignore_transitions and "transition" in g:
        trn = g["transition"][t0:t1]
        if (trn != 0).any():
            return None
    
    # Map 7->0 and get majority vote
    mov = np.where(mov == 7, 0, mov)
    vals, cnts = np.unique(mov, return_counts=True)
    return int(vals[np.argmax(cnts)])

def make_window_index(h5_path: str, videos_df: pd.DataFrame, t_win: int, stride: int,
                      require_isw2: bool = True, ignore_transitions: bool = True):
    """Create window index from real HDF5 data."""
    items = []
    with h5py.File(h5_path, "r") as f:
        vg = f["videos"]
        for _, row in videos_df.iterrows():
            vid = row["video_id"]
            if vid not in vg:
                continue
                
            g = vg[vid]
            T = g["data"].shape[0]
            W, H, fps = row["W"], row["H"], row["fps"]
            
            for t0 in range(0, max(1, T - t_win + 1), stride):
                t1 = min(t0 + t_win, T)
                y = window_label_movement_strict(
                    g, t0, t1, 
                    require_isw2=require_isw2,
                    ignore_transitions=ignore_transitions
                )
                if y is None:
                    continue
                    
                items.append((vid, t0, t_win, W, H, fps,
                             row["dataset_id"], row["camera_id"], y))
    return items

def scan_h5_meta(h5_path: str) -> pd.DataFrame:
    """Scan HDF5 file to get video metadata."""
    rows = []
    with h5py.File(h5_path, "r") as f:
        vg = f["videos"]
        for vid in vg.keys():
            g = vg[vid]
            T = int(g["data"].shape[0])
            attrs = dict(g.attrs)
            rows.append({
                "video_id": vid,
                "T": T,
                "dataset_id": attrs.get("dataset_id", ""),
                "camera_id": attrs.get("camera_id", ""),
                "fps": to_int(attrs.get("fps", 30)),
                "W": to_int(attrs.get("width", 320)),
                "H": to_int(attrs.get("height", 240)),
            })
    return pd.DataFrame(rows)

def load_data(h5_path: str, t_win: int, stride: int, val_fraction: float, 
              test_datasets: List[str], seed: int) -> Tuple:
    """Load and split dataset using REAL HDF5 data."""
    logger.info(f"Loading data from {h5_path}")
    
    # Scan HDF5 file for metadata
    meta_df = scan_h5_meta(h5_path)
    logger.info(f"Found {len(meta_df)} videos")
    logger.info(f"Datasets: {sorted(meta_df['dataset_id'].unique())}")
    
    # Split by datasets
    all_datasets = sorted(meta_df["dataset_id"].dropna().unique().tolist())
    if not test_datasets:
        test_datasets = [all_datasets[-1]] if all_datasets else []
    
    logger.info(f"Test datasets: {test_datasets}")
    
    is_test = meta_df["dataset_id"].isin(test_datasets)
    test_df = meta_df[is_test].copy()
    trainval = meta_df[~is_test].copy()
    
    # Split train/val
    rng = np.random.RandomState(seed)
    if len(trainval) > 0:
        m = len(trainval)
        idx = np.arange(m)
        rng.shuffle(idx)
        cut = int(round((1.0 - val_fraction) * m))
        tr_df = trainval.iloc[idx[:cut]].copy()
        va_df = trainval.iloc[idx[cut:]].copy()
    else:
        tr_df = trainval.copy()
        va_df = trainval.copy()
    
    logger.info(f"Split: Train={len(tr_df)}, Val={len(va_df)}, Test={len(test_df)}")
    
    # Create window indices
    logger.info("Creating window indices...")
    train_items = make_window_index(h5_path, tr_df, t_win, stride, True, True)
    val_items = make_window_index(h5_path, va_df, t_win, stride, True, True)
    test_items = make_window_index(h5_path, test_df, t_win, stride, False, False)
    
    logger.info(f"Windows: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
    
    # Log class distribution
    if train_items:
        train_labels = [item[-1] for item in train_items]
        train_counts = Counter(train_labels)
        logger.info(f"Train class distribution: {dict(train_counts)}")
    
    return train_items, val_items, test_items

def downsample_bg(items, max_ratio=0.2, seed=42):
    """
    Limita la PROPORCIÓN FINAL de clase 0 (bg) en el split de train.
    Si max_ratio=0.2, tras el muestreo el fondo será ≤20% aprox.
    """
    if max_ratio <= 0:
        return items

    rng = np.random.RandomState(seed)

    pos = [it for it in items if it[-1] != 0]
    bg  = [it for it in items if it[-1] == 0]

    # cuántos bg caben para lograr ratio final <= max_ratio
    # target_bg / (len(pos) + target_bg) <= max_ratio  =>  target_bg <= max_ratio/(1-max_ratio) * len(pos)
    target_bg = int(np.floor((max_ratio / max(1e-8, 1.0 - max_ratio)) * len(pos)))
    target_bg = min(target_bg, len(bg))

    if target_bg < len(bg):
        sel_idx = rng.choice(len(bg), size=target_bg, replace=False)
        bg = [bg[i] for i in sel_idx]

    new_items = pos + bg
    rng.shuffle(new_items)
    return new_items

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    enabled_amp: bool = False) -> Dict[str, float]:

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(dataloader):
        data   = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=enabled_amp):
            output = model(data)
            loss = criterion(output, target)

        if scaler is not None and enabled_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        for i in range(len(target)):
            label = target[i].item()
            class_total[label] += 1
            if pred[i] == target[i]:
                class_correct[label] += 1

        if batch_idx % log_interval == 0:
            logger.info(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}'
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

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
            data   = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
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
    
    # Class recalls for ALL classes (including 0 for monitoring)
    class_recalls = {}
    for cls in range(num_classes):
        if cls in class_total and class_total[cls] > 0:
            class_recalls[cls] = class_correct[cls] / class_total[cls]
        else:
            class_recalls[cls] = 0.0
    
    arr_preds = np.array(all_preds)
    arr_targets = np.array(all_targets)
    mask_pos = arr_targets != 0
    acc_no_bg = float((arr_preds[mask_pos] == arr_targets[mask_pos]).mean()) if mask_pos.any() else 0.0
    bg_ratio = float((arr_targets == 0).mean())
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': dict(zip(labels_eval, f1_per_class)),
        'precision_per_class': dict(zip(labels_eval, precision)),
        'recall_per_class': dict(zip(labels_eval, recall)),
        'class_recalls': class_recalls,  # Nueva métrica para monitoreo
        'class_accuracies': class_accuracies,
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
        'acc_no_bg': acc_no_bg,
        'bg_ratio': bg_ratio,
        'all_targets': all_targets,  # Para monitoreo detallado
        'all_preds': all_preds       # Para monitoreo detallado
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
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--augment_minority', action='store_true', default=False,
                       help='Apply data augmentation to minority classes')
    parser.add_argument('--bg_weight', type=float, default=0.1,
                    help='Peso relativo para la clase 0 en la pérdida')
    parser.add_argument('--bg_max_ratio', type=float, default=-1.0,
                        help='Si >0, porcentaje máximo de clase 0 en train (downsampling)')
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
    parser.add_argument('--project_name', type=str, default='LAIA-net',
                       help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Wandb run name')
    
    # Plotting parameters
    parser.add_argument('--plot_local', action='store_true', default=True,
                       help='Save plots locally')
    
    # --- Rendimiento / CUDA / AMP ---
    parser.add_argument('--amp', action='store_true',
                        help='Usar mixed precision (autocast + GradScaler) en CUDA')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='pin_memory en DataLoader (solo útil con CUDA)')
    parser.add_argument('--persistent_workers', action='store_true', default=True,
                        help='Mantener workers vivos entre epochs (requiere num_workers>0)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Batches prefetch por worker (requiere num_workers>0)')

    # --- Cache en RAM del HDF5 ---
    parser.add_argument('--cache_mode', type=str, default='none',
                        choices=['none', 'lazy', 'all'],
                        help='Cache por vídeo en RAM: none (sin cache), lazy (bajo demanda), all (precargar todo)')
    parser.add_argument('--max_cache_gb', type=float, default=2.0,
                        help='Límite aproximado de RAM para cache lazy (no estricto)')
    
        
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    # Set seed
    set_seed(args.seed)
# ------------------------------------------------------------------
# Ajustes de DataLoader y cache  ▸  poner después de args.parse_args()
# ------------------------------------------------------------------
    pin_memory = bool(args.pin_memory and device.type == 'cuda')
    use_persistent = bool(args.persistent_workers and args.num_workers > 0)

    loader_kwargs = dict(
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        pin_memory      = pin_memory,
        persistent_workers = use_persistent,
        drop_last       = True          # solo para train; val/test lo quitamos luego
    )

    # prefetch_factor solo existe cuando num_workers > 0 y torch ≥1.7
    if args.num_workers > 0:
        try:
            loader_kwargs['prefetch_factor'] = max(2, int(args.prefetch_factor))
        except TypeError:
            pass  # versión de torch sin este parámetro

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
                    entity='c-vasquezr',
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
    if args.bg_max_ratio > 0:
        old_n = len(train_items)
        train_items = downsample_bg(train_items, max_ratio=args.bg_max_ratio, seed=args.seed)
        logger.info(f"Downsample bg: {old_n} -> {len(train_items)}")   

    for name, items in [('Train', train_items), ('Val', val_items), ('Test', test_items)]:
        cnt = Counter([it[-1] for it in items])
        tot = sum(cnt.values()) or 1
        bgp = 100.0 * cnt.get(0, 0) / tot
        logger.info(f'{name} counts: {dict(cnt)} | bg%={bgp:.1f}') 
    # ---------------------------------------------------------
    # ➊  Datasets con soporte de cache en RAM
    # ---------------------------------------------------------
    train_dataset = PSKUSDataset(
        args.data_path, train_items, use_z=args.use_z,
        augment_minority=args.augment_minority,
        cache_mode=args.cache_mode,            # <-- NUEVO
        max_cache_gb=args.max_cache_gb         # <-- NUEVO
    )
    val_dataset = PSKUSDataset(
        args.data_path, val_items, use_z=args.use_z, augment_minority=False,
        cache_mode='lazy' if args.cache_mode == 'all' else args.cache_mode,
        max_cache_gb=args.max_cache_gb
    )
    test_dataset = PSKUSDataset(
        args.data_path, test_items, use_z=args.use_z, augment_minority=False,
        cache_mode='lazy' if args.cache_mode == 'all' else args.cache_mode,
        max_cache_gb=args.max_cache_gb
    )

    # ---------------------------------------------------------
    # ➋  Sampler / Shuffle
    # ---------------------------------------------------------
    train_labels = [it[-1] for it in train_items]
    if args.sampling_strategy != 'none':
        sampler       = create_balanced_sampler(train_labels, args.sampling_strategy)
        shuffle_train = False
    else:
        sampler       = None
        shuffle_train = True

    # ---------------------------------------------------------
    # ➌  DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        shuffle = shuffle_train,
        sampler = sampler,
        **loader_kwargs
    )

    # Para val / test no usamos drop_last
    val_loader  = DataLoader(
        val_dataset,
        shuffle = False,
        drop_last = False,
        **{k: v for k, v in loader_kwargs.items() if k != 'drop_last'}
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle = False,
        drop_last = False,
        **{k: v for k, v in loader_kwargs.items() if k != 'drop_last'}
    )

    # Build model
    model_config = {
        'dropout': args.dropout,
        'use_class_attention': args.use_class_attention
    }
    model = build_model(args.num_classes, args.in_channels, device, model_config)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create loss function with de-emphasis on class 0
    class_counts = Counter(train_labels)
    samples_per_class = [class_counts.get(i, 1) for i in range(args.num_classes)]

    # Usa un peso bajo para la clase 0 (bg). Si no agregas el flag (1a),
    # puedes dejar 0.1 fijo aquí.
    bg_weight = getattr(args, 'bg_weight', 0.1)
    base_alpha = torch.tensor([bg_weight] + [1.0]*(args.num_classes-1),
                            dtype=torch.float32, device=device)

    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=base_alpha, gamma=args.focal_gamma)

    elif args.loss_type == 'class_balanced':
        cb = ClassBalancedLoss(samples_per_class=samples_per_class,
                            beta=args.cb_beta, gamma=args.focal_gamma,
                            loss_type='focal')
        # combina pesos CB con el castigo a la clase 0
        cb.weights = (cb.weights.to(device) * base_alpha).float()
        criterion = cb
    else:
        criterion = nn.CrossEntropyLoss(weight=base_alpha)

    logger.info(f"Using {args.loss_type} loss with class distribution: {dict(class_counts)} "
                f"| bg_weight={float(base_alpha[0]):.3f}")
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
    
    enabled_amp = args.amp and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=enabled_amp)

    # Training loop with early stopping
    best_val_f1 = 0.0
    early_stopping_counter = 0
    metrics_history = []
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        #train_metrics = train_one_epoch(
        #    model, train_loader, criterion, optimizer, device, epoch, args.log_interval
        #)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.log_interval,
            scaler=scaler, enabled_amp=enabled_amp
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
        # logger.info(f'Epoch {epoch:03d} | '
        #            f'Train: {train_metrics["loss"]:.4f}/{train_metrics["accuracy"]:.3f} | '
        #            f'Val: {val_metrics["loss"]:.4f}/{val_metrics["accuracy"]:.3f} | '
        #            f'F1: {val_metrics["f1_macro"]:.4f} | LR: {current_lr:.2e}')
        
        logger.info(
            f'Epoch {epoch:03d} | '
            f'Train: {train_metrics["loss"]:.4f}/{train_metrics["accuracy"]:.3f} | '
            f'Val: {val_metrics["loss"]:.4f}/{val_metrics["accuracy"]:.3f} | '
            f'F1: {val_metrics["f1_macro"]:.4f} | '
            f'Acc(no-bg): {val_metrics.get("acc_no_bg",0):.3f} | '
            f'bg_ratio={val_metrics.get("bg_ratio",0):.2f} | '
            f'LR: {current_lr:.2e}'
        )
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


# ============================================================================
# TWO-STAGE TRAINING FUNCTIONS
# ============================================================================

def find_optimal_threshold(model, val_loader, device, metric='f1'):
    """Encuentra el umbral óptimo para el clasificador binario"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target_bin = convert_labels_to_binary(target.cpu().numpy())
            
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(target_bin)
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(all_targets, preds, average='binary', zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic (sensitivity + specificity - 1)
            tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.3f} ({metric}={best_score:.3f})")
    return best_threshold, best_score

def train_binary_stage(train_items, val_bal_items, args, device):
    """Entrenar etapa 1: detector binario"""
    logger.info("=== INICIANDO ETAPA 1: DETECTOR BINARIO ===")
    
    # Convertir labels a binario
    train_labels_bin = convert_labels_to_binary([it[-1] for it in train_items])
    val_labels_bin = convert_labels_to_binary([it[-1] for it in val_bal_items])
    
    # Datasets
    train_dataset = PSKUSDataset(args.data_path, train_items, use_z=args.use_z, 
                                augment_minority=False)  # Sin augment en binario
    val_dataset = PSKUSDataset(args.data_path, val_bal_items, use_z=args.use_z, 
                              augment_minority=False)
    
    # Sampler balanceado para binario
    pos_weight = (len(train_labels_bin) - sum(train_labels_bin)) / max(sum(train_labels_bin), 1)
    logger.info(f"Binary classification - pos_weight: {pos_weight:.2f}")
    
    # Modelo binario (crear nuevo backbone)
    model_config = {'dropout': args.dropout, 'use_class_attention': False}
    base_model = build_model(args.num_classes, args.in_channels, device, model_config)
    binary_model = BinaryActivityModel(base_model, freeze_backbone=False)
    binary_model = binary_model.to(device)
    
    # Loss y optimizador
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(binary_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6)
    
    # DataLoaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True if device.type == 'cuda' else False
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    best_f1 = 0.0
    patience = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))
    
    for epoch in range(1, 13):  # Máx 12 epochs
        # Entrenar
        binary_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target_bin = convert_labels_to_binary(target.cpu().numpy())
            target_bin = torch.tensor(target_bin, dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and args.amp)):
                logits = binary_model(data)
                loss = criterion(logits, target_bin)
            
            if args.amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == target_bin).sum().item()
            total += target_bin.size(0)
            
            if batch_idx % args.log_interval == 0:
                logger.info(f'Binary Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f}')
        
        train_acc = correct / total
        
        # Validar y encontrar umbral óptimo
        threshold, f1_bin = find_optimal_threshold(binary_model, val_loader, device, 'f1')
        
        logger.info(f'Binary Epoch {epoch}: train_loss={train_loss/len(train_loader):.3f}, '
                   f'train_acc={train_acc:.3f}, val_f1={f1_bin:.3f}, threshold={threshold:.3f}')
        
        scheduler.step(f1_bin)
        
        if f1_bin > best_f1:
            best_f1 = f1_bin
            patience = 0
            torch.save({
                'model': binary_model.state_dict(),
                'threshold': threshold,
                'f1': f1_bin,
                'epoch': epoch
            }, os.path.join(args.save_dir, 'binary_model.pt'))
            logger.info(f"✅ Nuevo mejor modelo binario: F1={f1_bin:.3f}")
        else:
            patience += 1
            if patience >= 8:
                logger.info("Early stopping en etapa binaria")
                break
    
    # Cargar mejor modelo
    checkpoint = torch.load(os.path.join(args.save_dir, 'binary_model.pt'))
    binary_model.load_state_dict(checkpoint['model'])
    optimal_threshold = checkpoint['threshold']
    
    logger.info(f"=== ETAPA 1 COMPLETADA ===")
    logger.info(f"Mejor F1 binario: {checkpoint['f1']:.3f}")
    logger.info(f"Umbral óptimo: {optimal_threshold:.3f}")
    
    return binary_model, optimal_threshold

def train_multiclass_stage(train_items, val_bal_items, args, device):
    """Entrenar etapa 2: clasificador 6-clases"""
    logger.info("=== INICIANDO ETAPA 2: CLASIFICADOR MULTICLASE ===")
    
    # Filtrar solo clases positivas y remapear
    train_items_pos, train_labels_pos = filter_to_positive_classes(
        train_items, [it[-1] for it in train_items]
    )
    val_items_pos, val_labels_pos = filter_to_positive_classes(
        val_bal_items, [it[-1] for it in val_bal_items]
    )
    
    if len(train_items_pos) == 0:
        raise ValueError("No hay samples positivos para entrenar multiclase!")
    
    # Datasets
    train_dataset = PSKUSDataset(args.data_path, train_items_pos, use_z=args.use_z, 
                                augment_minority=True)  # Con augment para minorías
    val_dataset = PSKUSDataset(args.data_path, val_items_pos, use_z=args.use_z, 
                              augment_minority=False)
    
    # Calcular frecuencias para logit adjustment
    class_counts = Counter(train_labels_pos)
    class_freqs = torch.tensor([class_counts.get(i, 1) for i in range(6)], dtype=torch.float32)
    
    logger.info(f"Multiclass distribution: {dict(class_counts)}")
    logger.info(f"Class frequencies: {class_freqs}")
    
    # Modelo
    model_config = {'dropout': args.dropout, 'use_class_attention': args.use_class_attention}
    base_model = build_model(7, args.in_channels, device, model_config)  # Temporal
    multiclass_model = MultiClassModel(base_model, class_freqs, tau=getattr(args, 'tau', 1.0), 
                                      freeze_backbone=False)
    multiclass_model = multiclass_model.to(device)
    
    # Loss con pesos inversos
    weights = 1.0 / (class_freqs + 1e-8)  # Evitar división por 0
    weights = weights / weights.sum() * len(weights)  # Normalizar
    criterion = FocalLoss(alpha=weights.to(device), gamma=2.0)
    
    # Sampler balanceado
    sampler = create_balanced_sampler(train_labels_pos, strategy="balanced")
    
    # Optimizador
    optimizer = torch.optim.AdamW(multiclass_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8)
    
    # DataLoaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True if device.type == 'cuda' else False
    }
    
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    # Sistema de monitoreo detallado
    metrics_tracker = DetailedMetricsTracker(num_classes=7, save_dir=args.save_dir, 
                                            patience_zero_recall=4, dominance_threshold=0.7)
    
    best_f1 = 0.0
    patience = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))
    
    for epoch in range(1, 21):  # Máx 20 epochs
        # Entrenar
        multiclass_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target_orig) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            
            # Convertir targets originales (1-6) a 0-5 para multiclase
            target = torch.tensor([t-1 for t in target_orig], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and args.amp)):
                logits = multiclass_model(data, apply_adjustment=True)
                loss = criterion(logits, target)
            
            if args.amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % args.log_interval == 0:
                logger.info(f'Multiclass Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f}')
        
        train_acc = correct / total
        
        # Evaluar
        multiclass_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target_orig in val_loader:
                data = data.to(device, non_blocking=True)
                target = torch.tensor([t-1 for t in target_orig], dtype=torch.long, device=device)
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and args.amp)):
                    logits = multiclass_model(data, apply_adjustment=False)
                    loss = criterion(logits, target)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_acc = val_correct / val_total
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # Calcular class recalls para monitoreo
        class_recalls = {}
        for cls in range(6):
            mask = np.array(all_targets) == cls
            if mask.sum() > 0:
                class_recalls[cls+1] = (np.array(all_preds)[mask] == cls).mean()  # +1 para clases 1-6
            else:
                class_recalls[cls+1] = 0.0
        
        val_metrics = {
            'class_recalls': class_recalls,
            'f1_macro': f1_macro
        }
        
        logger.info(f'Multiclass Epoch {epoch}: train_loss={train_loss/len(train_loader):.3f}, '
                   f'train_acc={train_acc:.3f}, val_loss={val_loss/len(val_loader):.3f}, '
                   f'val_acc={val_acc:.3f}, val_f1_macro={f1_macro:.3f}')
        
        # Log per-class F1
        f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
        logger.info(f'F1 per class: {dict(enumerate(f1_per_class))}')
        
        # Sistema de monitoreo detallado
        should_stop, collapse_reason = metrics_tracker.update(epoch, all_targets, all_preds, val_metrics)
        
        if should_stop:
            logger.warning(f"🛑 DETENIENDO ENTRENAMIENTO: {collapse_reason}")
            break
        
        scheduler.step(f1_macro)
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            patience = 0
            torch.save({
                'model': multiclass_model.state_dict(),
                'f1_macro': f1_macro,
                'epoch': epoch,
                'class_freqs': class_freqs
            }, os.path.join(args.save_dir, 'multiclass_model.pt'))
            logger.info(f"✅ Nuevo mejor modelo multiclase: F1_macro={f1_macro:.3f}")
        else:
            patience += 1
            if patience >= 10:
                logger.info("Early stopping en etapa multiclase")
                break
    
    # Cargar mejor modelo
    checkpoint = torch.load(os.path.join(args.save_dir, 'multiclass_model.pt'))
    multiclass_model.load_state_dict(checkpoint['model'])
    
    logger.info(f"=== ETAPA 2 COMPLETADA ===")
    logger.info(f"Mejor F1 macro: {checkpoint['f1_macro']:.3f}")
    
    return multiclass_model

def evaluate_two_stage_pipeline(binary_model, multiclass_model, threshold, test_loader, device, args):
    """Evaluar el pipeline completo de 2 etapas"""
    logger.info("=== EVALUANDO PIPELINE DE 2 ETAPAS ===")
    
    pipeline = TwoStageInference(binary_model, multiclass_model, threshold, device)
    
    all_preds = []
    all_targets = []
    
    for data, target in test_loader:
        data = data.to(device, non_blocking=True)
        preds = pipeline.predict(data)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    # Métricas completas
    f1_macro = f1_score(all_targets, all_preds, labels=list(range(1, args.num_classes)), 
                       average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, labels=list(range(1, args.num_classes)), 
                          average='weighted', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(all_targets, all_preds, labels=list(range(1, args.num_classes)), 
                           average=None, zero_division=0)
    
    # Accuracy general y sin fondo
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    
    mask_pos = np.array(all_targets) > 0
    if mask_pos.sum() > 0:
        acc_no_bg = (np.array(all_preds)[mask_pos] == np.array(all_targets)[mask_pos]).mean()
    else:
        acc_no_bg = 0.0
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    results = {
        'accuracy': accuracy,
        'acc_no_bg': acc_no_bg,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': dict(enumerate(f1_per_class, 1)),
        'confusion_matrix': conf_matrix,
        'all_targets': all_targets,
        'all_preds': all_preds
    }
    
    logger.info(f"Two-stage pipeline results:")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Acc (no bg): {acc_no_bg:.3f}")
    logger.info(f"  F1 Macro: {f1_macro:.3f}")
    logger.info(f"  F1 per class: {results['f1_per_class']}")
    
    return results

# ============================================================================
# MAIN FUNCTION WITH TWO-STAGE PIPELINE
# ============================================================================

def main():
    """Main function with two-stage training pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced STGCN with Two-Stage Training')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--t_win', type=int, default=64, help='Temporal window size')
    parser.add_argument('--stride', type=int, default=8, help='Stride for sliding window')
    parser.add_argument('--val_fraction', type=float, default=0.18, help='Validation fraction')
    parser.add_argument('--test_datasets', type=str, nargs='*', default=[], help='Test dataset IDs')
    
    # Model arguments  
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=2, help='Input channels (2 for XY, 3 for XYZ)')
    parser.add_argument('--use_z', action='store_true', help='Use Z coordinate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_class_attention', action='store_true', help='Use class attention mechanism')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine', 'step'], default='plateau', help='LR scheduler')
    
    # Two-stage specific arguments
    parser.add_argument('--two_stage', action='store_true', help='Use two-stage training pipeline')
    parser.add_argument('--val_bg_ratio', type=float, default=0.3, help='Background ratio in balanced validation')
    parser.add_argument('--tau', type=float, default=1.0, help='Logit adjustment temperature')
    
    # Technical arguments
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    # Monitoring arguments  
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='./exp/two_stage', help='Save directory')
    parser.add_argument('--plot_local', action='store_true', help='Generate local plots')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--project_name', type=str, default='stgcn-gesture', help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    
    # Legacy arguments (for compatibility)
    parser.add_argument('--bg_max_ratio', type=float, default=0, help='Max background ratio (0=disabled)')
    parser.add_argument('--sampling_strategy', type=str, default='none', help='Sampling strategy')
    parser.add_argument('--loss_type', type=str, default='focal', help='Loss type')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--cb_beta', type=float, default=0.9999, help='Class balanced beta')
    parser.add_argument('--bg_weight', type=float, default=0.1, help='Background weight')
    parser.add_argument('--augment_minority', action='store_true', help='Augment minority classes')
    parser.add_argument('--cache_mode', type=str, default='none', help='Cache mode')
    parser.add_argument('--max_cache_gb', type=float, default=2.0, help='Max cache GB')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pin memory')
    parser.add_argument('--persistent_workers', type=int, default=1, help='Persistent workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'monitoring'), exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Wandb not available. Install with: pip install wandb")
            args.use_wandb = False
        else:
            try:
                wandb.init(
                    project=args.project_name,
                    entity='c-vasquezr',
                    name=args.run_name,
                    config=vars(args),
                    tags=['two-stage', 'enhanced-stgcn', 'class-imbalance']
                )
                logger.info("Wandb initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb.")
                args.use_wandb = False
    
    # Load data
    logger.info("Loading and preparing data...")
    train_items, val_items, test_items = load_data(
        args.data_path, args.t_win, args.stride, args.val_fraction,
        args.test_datasets, args.seed
    )
    
    # Print original distribution
    for name, items in [('Train', train_items), ('Val', val_items), ('Test', test_items)]:
        cnt = Counter([it[-1] for it in items])
        tot = sum(cnt.values()) or 1
        bgp = 100.0 * cnt.get(0, 0) / tot
        logger.info(f'{name} counts: {dict(cnt)} | bg%={bgp:.1f}')
    
    # Create balanced validation set
    val_bal_items, val_original_items = create_balanced_validation(
        val_items, target_bg_ratio=args.val_bg_ratio, seed=args.seed
    )
    
    # Adjust input channels
    if args.use_z:
        args.in_channels = 3
    
    if args.two_stage:
        logger.info("🚀 INICIANDO ENTRENAMIENTO DE 2 ETAPAS")
        
        # Stage 1: Binary classification
        binary_model, optimal_threshold = train_binary_stage(train_items, val_bal_items, args, device)
        
        # Stage 2: Multiclass classification
        multiclass_model = train_multiclass_stage(train_items, val_bal_items, args, device)
        
        # Test evaluation with complete pipeline
        test_dataset = PSKUSDataset(args.data_path, test_items, use_z=args.use_z, augment_minority=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers)
        
        test_results = evaluate_two_stage_pipeline(binary_model, multiclass_model, optimal_threshold, 
                                                  test_loader, device, args)
        
        # Save results
        final_results = {
            'pipeline_type': 'two_stage',
            'optimal_threshold': optimal_threshold,
            'test_results': test_results,
            'args': vars(args)
        }
        
        with open(os.path.join(args.save_dir, 'two_stage_results.json'), 'w') as f:
            json.dump({k: v for k, v in final_results.items() if k != 'test_results'}, f, indent=2)
        
        # Save confusion matrix plot
        if args.plot_local:
            plot_confusion_matrix(
                test_results['confusion_matrix'],
                [f'Class {i}' for i in range(args.num_classes)],
                os.path.join(args.save_dir, 'plots')
            )
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                "test_f1_macro_two_stage": test_results['f1_macro'],
                "test_accuracy_two_stage": test_results['accuracy'],
                "test_acc_no_bg_two_stage": test_results['acc_no_bg'],
                "optimal_threshold": optimal_threshold
            })
            
            if args.plot_local:
                wandb.log({
                    "confusion_matrix_two_stage": wandb.Image(os.path.join(args.save_dir, 'plots', 'confusion_matrix.png'))
                })
            
            wandb.finish()
        
        logger.info("🎉 ENTRENAMIENTO DE 2 ETAPAS COMPLETADO")
        logger.info(f"Resultados finales guardados en: {args.save_dir}")
        logger.info(f"F1 Macro en Test: {test_results['f1_macro']:.4f}")
        
    else:
        logger.info("Using single-stage training (original pipeline)")
        # Aquí iría el pipeline original si no se usa two_stage
        # Por ahora solo mostramos un mensaje
        logger.warning("Single-stage pipeline not implemented in this version. Use --two_stage flag.")

if __name__ == '__main__':
    main()