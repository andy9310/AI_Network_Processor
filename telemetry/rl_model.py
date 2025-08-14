"""
RL模型載入和推理模組
基於PPO訓練的強化學習模型
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import random

# Try to import stable_baselines3, but don't fail if not available
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    print("⚠️ stable_baselines3 not available. Using mock model only.")

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only define these classes if stable_baselines3 is available
if STABLE_BASELINES3_AVAILABLE:
    class NetworkFeatureExtractor(nn.Module):
        """網路特徵提取器，與訓練時保持一致"""
        def __init__(self, observation_space, features_dim=256):
            super().__init__()
            # Input shape: (n_links, 7)
            n_links = observation_space.shape[0]
            feature_dim = observation_space.shape[1]
            self.features_dim = features_dim
            
            # Process each link with a small network first
            self.link_encoder = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
            
            # Simple attention mechanism (no LSTM)
            self.attention = nn.Sequential(
                nn.Linear(32, 1),
                nn.Softmax(dim=1)
            )
            
            # Global network features
            self.global_net = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, features_dim)
            )
        
        def forward(self, observations):
            batch_size = observations.shape[0]
            n_links = observations.shape[1]
            
            # Reshape to process each link
            flat_obs = observations.view(-1, observations.shape[2])
            
            # Process each link
            link_features = self.link_encoder(flat_obs)
            link_features = link_features.view(batch_size, n_links, -1)
            
            # Apply attention (no LSTM)
            attention_weights = self.attention(link_features)
            weighted_features = link_features * attention_weights
            
            # Sum across links (weighted by attention)
            global_features = weighted_features.sum(dim=1)
            
            return self.global_net(global_features)

    class EnhancedNetworkPolicy(ActorCriticPolicy):
        """增強的網路策略，與訓練時保持一致"""
        def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                *args,
                **kwargs,
                features_extractor_class=NetworkFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256),
            )
else:
    # Placeholder classes when stable_baselines3 is not available
    class NetworkFeatureExtractor:
        """Placeholder for NetworkFeatureExtractor when stable_baselines3 is not available"""
        pass
    
    class EnhancedNetworkPolicy:
        """Placeholder for EnhancedNetworkPolicy when stable_baselines3 is not available"""
        pass

class MockRLModel:
    """模擬RL模型，用於測試階段"""
    
    def __init__(self):
        self.links = [
            "S1-S2", "S1-S3", "S1-S4", "S1-S9",
            "S2-S1", "S2-S4", "S2-S9",
            "S3-S1", "S3-S4", "S3-S9",
            "S4-S1", "S4-S2", "S4-S3", "S4-S5", "S4-S6", "S4-S7",
            "S4-S8", "S4-S9", "S4-S10", "S4-S11", "S4-S15",
            "S5-S4", "S5-S9",
            "S6-S4", "S6-S15",
            "S7-S4", "S7-S9",
            "S8-S4", "S8-S9",
            "S9-S1", "S9-S2", "S9-S3", "S9-S4", "S9-S5",
            "S9-S7", "S9-S8", "S9-S10", "S9-S15",
            "S10-S4", "S10-S9", "S10-S12", "S10-S13",
            "S10-S14", "S10-S15", "S10-S16", "S10-S17",
            "S11-S4", "S11-S15",
            "S12-S10", "S12-S15",
            "S13-S10", "S13-S15",
            "S14-S10", "S14-S15",
            "S15-S4", "S15-S6", "S15-S9", "S15-S10", "S15-S11",
            "S15-S12", "S15-S13", "S15-S14", "S15-S16", "S15-S17",
            "S16-S10", "S16-S15",
            "S17-S10", "S17-S15",
        ]
        
        # 定義一些預設的關閉策略
        self.closing_strategies = [
            # 策略1: 關閉低流量連結
            ["S1-S9", "S2-S9", "S3-S9", "S5-S9", "S7-S9", "S8-S9"],
            # 策略2: 關閉邊緣連結
            ["S1-S2", "S1-S3", "S16-S10", "S17-S10", "S11-S4", "S12-S10"],
            # 策略3: 關閉冗餘連結
            ["S4-S5", "S4-S6", "S4-S7", "S4-S8", "S9-S7", "S9-S8"],
            # 策略4: 關閉高延遲連結
            ["S1-S4", "S2-S4", "S3-S4", "S10-S4", "S11-S4"],
            # 策略5: 隨機選擇
            ["S1-S2", "S3-S4", "S5-S9", "S10-S12", "S15-S16"]
        ]
        
        logger.info("🤖 Mock RL Model initialized for testing")
    
    def predict(self, observation, deterministic=True):
        """模擬模型預測，返回隨機的閾值"""
        # 生成隨機的3維動作 (閾值)
        action = np.random.uniform(0.0, 1.0, 3)
        return action, None

class RLModelManager:
    """
    RL模型管理器
    負責載入訓練好的PPO模型、處理輸入數據、進行推理
    """
    
    def __init__(self, model_path: str = None, device: str = "cpu", use_mock: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        # Update this path to match your trained model filename
        self.model_path = model_path or "models/python_model_20250810_161530"  # Path to extracted model
        self.use_mock = use_mock  # 使用模擬模型進行測試
        
        # 歷史數據用於計算變化率
        self.buffer_history = {}
        self.util_history = {}
        self.history_length = 3
        
        # 網路拓撲信息 - 使用訓練時的確切hardcoded links (雙向的)
        # 基於訓練時使用的hardcoded_links生成雙向連結
        hardcoded_links = [
            (0, 1),  # S1-S2
            (0, 2),  # S1-S3
            (0, 3),  # S1-S4
            (0, 8),  # S1-S9
            (1, 3),  # S2-S4
            (1, 8),  # S2-S9
            (2, 3),  # S3-S4
            (2, 8),  # S3-S9
            (3, 4),  # S4-S5
            (3, 5),  # S4-S6
            (3, 6),  # S4-S7
            (3, 7),  # S4-S8
            (3, 8),  # S4-S9
            (3, 9),  # S4-S10
            (3, 10), # S4-S11
            (3, 14), # S4-S15
            (4, 8),  # S5-S9
            (5, 14), # S6-S15
            (6, 8),  # S7-S9
            (7, 8),  # S8-S9
            (8, 9),  # S9-S10
            (8, 14), # S9-S15
            (9, 14), # S10-S15
            (9, 15), # S10-S16
            (9, 16)  # S10-S17
        ]
        
        # 將hardcoded_links轉換為雙向的link名稱 (總共50個)
        self.links = []
        for u, v in hardcoded_links:
            # 添加雙向連結
            self.links.append(f"S{u+1}-S{v+1}")
            self.links.append(f"S{v+1}-S{u+1}")
        
        logger.info(f"🔧 Using exactly {len(self.links)} hardcoded training links")
        
        # 節點度數計算（靜態特徵）
        self.node_degrees = self._calculate_node_degrees()
        
        # 載入模型
        self.load_model()
        
    def _calculate_node_degrees(self):
        """計算每個link的節點度數"""
        import networkx as nx
        
        # 創建網路圖
        graph = nx.Graph()
        for i, link in enumerate(self.links):
            u, v = link.split("-")
            graph.add_edge(u, v, link_idx=i)
        
        # 計算每個link的平均節點度數
        node_degrees = np.zeros(len(self.links))
        for i, link in enumerate(self.links):
            u, v = link.split("-")
            node_degrees[i] = (graph.degree[u] + graph.degree[v]) / 2.0
            
        return node_degrees
        
    def load_model(self):
        """載入訓練好的PPO模型或使用模擬模型"""
        if self.use_mock:
            logger.info("🤖 Using Mock RL Model for testing")
            self.model = MockRLModel()
            return
        
        # Check if stable_baselines3 is available
        if not STABLE_BASELINES3_AVAILABLE:
            logger.warning("⚠️ stable_baselines3 not available. Using Mock RL Model.")
            self.model = MockRLModel()
            return
        
        try:
            # Check if model directory exists (for extracted models) or zip file exists
            model_dir_exists = os.path.exists(self.model_path) and os.path.isdir(self.model_path)
            model_zip_exists = os.path.exists(self.model_path + ".zip")
            
            if model_dir_exists or model_zip_exists:
                logger.info(f"🔄 Loading RL model from {self.model_path}")
                
                # 載入PPO模型
                self.model = PPO.load(
                    self.model_path,
                    device=self.device,
                    custom_objects={
                        "policy_class": EnhancedNetworkPolicy
                    }
                )
                
                logger.info("✅ RL model loaded successfully")
                
            else:
                logger.warning(f"⚠️ Model file/directory not found: {self.model_path}")
                logger.info("🔄 Using Mock RL Model for testing...")
                self.model = MockRLModel()
                
        except Exception as e:
            logger.error(f"❌ Failed to load RL model: {e}")
            logger.info("🔄 Falling back to Mock RL Model...")
            self.model = MockRLModel()
    
    def preprocess_telemetry_data(self, telemetry_data: Dict) -> np.ndarray:
        """
        預處理telemetry數據，轉換為模型輸入格式
        與訓練時的_state()方法保持一致
        
        Args:
            telemetry_data: 包含link流量數據的字典
            
        Returns:
            np.ndarray: 預處理後的觀察空間 (n_links, 7)
        """
        try:
            # 確保我們只使用預定義的50個link
            n_links = len(self.links)
            if n_links != 50:
                logger.error(f"❌ Expected exactly 50 links, but have {n_links}")
                # 強制設定為50個link
                self.links = self.links[:50] if len(self.links) > 50 else self.links
                n_links = 50
            
            state = np.zeros((50, 7), dtype=np.float32)  # 強制設定為(50, 7)
            
            # 檢查telemetry_data的格式
            if not isinstance(telemetry_data, dict):
                logger.warning(f"⚠️ Expected dict for telemetry_data, got {type(telemetry_data)}")
                return state  # 返回零陣列
            
            # 調試信息
            logger.info(f"🔧 Processing {len(self.links)} predefined links, telemetry has {len(telemetry_data)} links")
            logger.info(f"🔧 State shape will be: {state.shape}")
            
            for i, link in enumerate(self.links):
                if link in telemetry_data:
                    current_data = telemetry_data[link]
                    
                    # 檢查current_data是否為字典格式
                    if not isinstance(current_data, dict):
                        # 如果current_data是數值，假設它是traffic值
                        if isinstance(current_data, (int, float)):
                            traffic = float(current_data)
                            output_drops = 0
                            output_queue_drops = 0
                            max_capacity = 1000  # 預設值
                        else:
                            logger.warning(f"⚠️ Unexpected data format for {link}: {type(current_data)}")
                            continue
                    else:
                        # 正常的字典格式
                        output_drops = current_data.get('output-drops', 0)
                        output_queue_drops = current_data.get('output-queue-drops', 0)
                        traffic = current_data.get('traffic', 0)
                        max_capacity = current_data.get('max-capacity', 1000)
                    
                    # 計算統計值
                    total_drops = output_drops + output_queue_drops
                    link_utilization = min(1.0, traffic / max_capacity) if max_capacity > 0 else 0.0
                    
                    # 估算buffer utilization (基於掉包率)
                    drop_rate = total_drops / max(1, traffic + total_drops)
                    buffer_utilization = min(1.0, drop_rate * 10)
                    
                    # 更新歷史數據
                    if link not in self.buffer_history:
                        self.buffer_history[link] = [buffer_utilization] * self.history_length
                    if link not in self.util_history:
                        self.util_history[link] = [link_utilization] * self.history_length
                    
                    self.buffer_history[link].append(buffer_utilization)
                    self.util_history[link].append(link_utilization)
                    
                    # 保持歷史長度
                    if len(self.buffer_history[link]) > self.history_length:
                        self.buffer_history[link] = self.buffer_history[link][-self.history_length:]
                    if len(self.util_history[link]) > self.history_length:
                        self.util_history[link] = self.util_history[link][-self.history_length:]
                    
                    # 計算變化率
                    buffer_change = 0.0
                    util_change = 0.0
                    if len(self.buffer_history[link]) >= 2:
                        buffer_change = self.buffer_history[link][-1] - self.buffer_history[link][-2]
                    if len(self.util_history[link]) >= 2:
                        util_change = self.util_history[link][-1] - self.util_history[link][-2]
                    
                    # 時間相關特徵（簡化）
                    time_since_change = 0.0  # 簡化處理
                    
                    # 正規化節點度數
                    max_degree = np.max(self.node_degrees) if len(self.node_degrees) > 0 else 1.0
                    norm_degree = self.node_degrees[i] / max_degree if max_degree > 0 else 0.0
                    
                    # 組裝狀態向量 (與訓練時保持一致)
                    state[i] = [
                        buffer_utilization,    # Buffer utilization
                        link_utilization,      # Link utilization
                        1.0,                   # Link status (假設都是up)
                        buffer_change,         # Buffer change rate
                        util_change,           # Utilization change rate
                        time_since_change,     # Time since last link state change
                        norm_degree,           # Normalized node degree
                    ]
            
            # 確保返回的state形狀正確
            if state.shape != (50, 7):
                logger.error(f"❌ State shape mismatch! Expected (50, 7), got {state.shape}")
                state = np.zeros((50, 7), dtype=np.float32)
            
            logger.info(f"✅ Returning state with shape: {state.shape}")
            return state
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing telemetry data: {e}")
            # 返回零陣列作為fallback - 確保是(50, 7)
            return np.zeros((50, 7), dtype=np.float32)
    
    def predict_links_to_close(self, telemetry_data: Dict) -> List[str]:
        """
        使用RL模型預測閾值，然後通過sophisticated heuristic算法決定關閉哪些link
        
        Args:
            telemetry_data: 即時流量數據
            
        Returns:
            List[str]: 應該關閉的link列表
        """
        try:
            if self.model is None:
                logger.warning("⚠️ No RL model loaded, using fallback")
                return ["S1-S2", "S3-S4", "S5-S9"]
            
            # 如果是模擬模型，使用預定義策略
            if isinstance(self.model, MockRLModel):
                # 隨機選擇一個策略
                strategy = random.choice(self.model.closing_strategies)
                logger.info(f"🤖 Mock RL selected strategy: {strategy}")
                return strategy
            
            # 預處理數據
            observation = self.preprocess_telemetry_data(telemetry_data)
            
            # 進行推理 - RL模型預測3個閾值
            action, _ = self.model.predict(observation, deterministic=True)
            
            # RL輸出的action就是3個閾值 (bufLow, utilHi, utilCap)
            # 這些閾值會被heuristic算法進一步處理和縮放
            logger.info(f"🔧 RL raw thresholds: {action}")
            
            # 使用sophisticated heuristic算法進行決策
            if not hasattr(self, '_heuristic_manager'):
                from network_heuristic import NetworkHeuristicManager
                self._heuristic_manager = NetworkHeuristicManager()
            
            # 通過heuristic算法處理RL預測的閾值
            links_to_close = self._heuristic_manager.step(action, telemetry_data)
            
            logger.info(f"🤖 RL+Heuristic predicted {len(links_to_close)} links to close: {links_to_close}")
            return links_to_close
            
        except Exception as e:
            logger.error(f"❌ Error in RL prediction: {e}")
            return ["S1-S2", "S3-S4", "S5-S9"]  # 預設值
    
    def get_model_info(self) -> Dict:
        """獲取模型信息"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        model_type = "MockRLModel" if isinstance(self.model, MockRLModel) else "PPO"
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": str(self.device),
            "model_type": model_type,
            "policy_type": "EnhancedNetworkPolicy" if model_type == "PPO" else "MockPolicy",
            "use_mock": self.use_mock
        }
    
    def save_model(self, save_path: str = None):
        """保存模型"""
        if self.model is None:
            logger.error("❌ No model to save")
            return
        
        save_path = save_path or self.model_path
        try:
            if not isinstance(self.model, MockRLModel) and STABLE_BASELINES3_AVAILABLE:
                self.model.save(save_path)
                logger.info(f"✅ Model saved to {save_path}")
            else:
                logger.warning("⚠️ Cannot save MockRLModel or stable_baselines3 not available")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

# 全局模型管理器實例
rl_manager = None

def get_rl_manager(use_mock: bool = True) -> RLModelManager:
    """獲取全局RL模型管理器實例"""
    global rl_manager
    if rl_manager is None:
        rl_manager = RLModelManager(use_mock=use_mock)
    return rl_manager 