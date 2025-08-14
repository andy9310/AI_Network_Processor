"""
Network Heuristic Manager
Implements the sophisticated link closure algorithm with RL integration
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)

class NetworkHeuristicManager:
    """
    Implements the sophisticated heuristic algorithm for link closure decisions
    Integrates with RL model predictions for threshold optimization
    """
    
    def __init__(self, step_duration: float = 10.0, min_link_state_duration: float = 30.0):
        """
        Initialize the heuristic manager
        
        Args:
            step_duration: Time between decision steps (seconds)
            min_link_state_duration: Minimum time before allowing link state changes (seconds)
        """
        self.step_t = step_duration
        self.min_link_state_duration = min_link_state_duration
        
        # Link topology - using exact hardcoded links from RL training (bidirectional)
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
        
        # Convert to bidirectional pairs (total: 50 pairs)
        self._pairs = []
        for u, v in hardcoded_links:
            # Add bidirectional connections
            self._pairs.append((f"S{u+1}", f"S{v+1}"))
            self._pairs.append((f"S{v+1}", f"S{u+1}"))
        
        # Convert pairs to link names
        self.link_names = [f"{u}-{v}" for u, v in self._pairs]
        
        # State tracking
        self.buffer_history = defaultdict(lambda: deque(maxlen=10))  # Buffer history for each link
        self.last_link_change = defaultdict(float)  # Time since last state change
        self.prev_states = {}  # Previous link states
        self.current_loading_mode = 'low'  # 'low' or 'high'
        
        # Initialize all links as up
        for i, link_name in enumerate(self.link_names):
            self.prev_states[i] = True
            self.last_link_change[i] = self.min_link_state_duration  # Allow immediate changes initially
        
        logger.info(f"🔧 NetworkHeuristicManager initialized with {len(self.link_names)} links")
    
    def update_loading_mode(self, telemetry_data: Dict) -> str:
        """
        Determine current network loading mode based on overall traffic
        
        Args:
            telemetry_data: Current telemetry data
            
        Returns:
            str: 'low' or 'high' loading mode
        """
        try:
            total_utilization = 0
            active_links = 0
            
            for link_name in self.link_names:
                if link_name in telemetry_data:
                    data = telemetry_data[link_name]
                    traffic = data.get('traffic', 0)
                    max_capacity = data.get('max-capacity', 1000)
                    
                    if max_capacity > 0:
                        utilization = min(1.0, traffic / max_capacity)
                        total_utilization += utilization
                        active_links += 1
            
            if active_links > 0:
                avg_utilization = total_utilization / active_links
                # Switch to high mode if average utilization > 40%
                self.current_loading_mode = 'high' if avg_utilization > 0.4 else 'low'
            
            return self.current_loading_mode
            
        except Exception as e:
            logger.warning(f"⚠️ Error determining loading mode: {e}")
            return self.current_loading_mode
    
    def get_link_stats(self, telemetry_data: Dict) -> List[Tuple[float, float, bool]]:
        """
        Extract link statistics from telemetry data
        
        Args:
            telemetry_data: Current telemetry data
            
        Returns:
            List of (buffer_utilization, link_utilization, is_up) for each link
        """
        stats = []
        
        for i, link_name in enumerate(self.link_names):
            if link_name in telemetry_data:
                data = telemetry_data[link_name]
                
                # Handle different telemetry data formats (same as RL model preprocessing)
                if not isinstance(data, dict):
                    # If data is numeric, assume it's traffic value
                    if isinstance(data, (int, float)):
                        traffic = float(data)
                        output_drops = 0
                        output_queue_drops = 0
                        max_capacity = 1000  # Default value
                    else:
                        logger.warning(f"⚠️ Unexpected data format for {link_name}: {type(data)}")
                        stats.append((0.0, 0.0, False))
                        self.buffer_history[i].append(0.0)
                        continue
                else:
                    # Normal dictionary format
                    output_drops = data.get('output-drops', 0)
                    output_queue_drops = data.get('output-queue-drops', 0)
                    traffic = data.get('traffic', 0)
                    max_capacity = data.get('max-capacity', 1000)
                
                # Calculate buffer utilization from drops
                total_drops = output_drops + output_queue_drops
                drop_rate = total_drops / max(1, traffic + total_drops)
                buffer_util = min(1.0, drop_rate * 10)  # Scale drop rate to buffer utilization
                
                # Calculate link utilization
                link_util = min(1.0, traffic / max_capacity) if max_capacity > 0 else 0.0
                
                # Update buffer history
                self.buffer_history[i].append(buffer_util)
                
                # Link is considered up if it has recent traffic or is in prev_states
                is_up = self.prev_states.get(i, True)
                
                stats.append((buffer_util, link_util, is_up))
            else:
                # No data for this link - consider it down
                stats.append((0.0, 0.0, False))
                self.buffer_history[i].append(0.0)
        
        return stats
    
    def step(self, action: np.ndarray, telemetry_data: Dict) -> List[str]:
        """
        Main heuristic algorithm step - your sophisticated algorithm implementation
        
        Args:
            action: RL model output (3 thresholds: bufLow, utilHi, utilCap)
            telemetry_data: Current telemetry data
            
        Returns:
            List of link names that should be closed
        """
        try:
            # 1. Get current network state
            stats = self.get_link_stats(telemetry_data)
            upmask = [isup for *_, isup in stats]  # Current link states
            
            # 2. Map action to thresholds based on loading mode
            bufLow, utilHi, utilCap = np.clip(action, 0.0, 1.0)
            
            # 3. Scale thresholds based on current loading mode
            loading_mode = self.update_loading_mode(telemetry_data)
            
            if loading_mode == 'low':
                bufLow *= 0.30      # 0.00 - 0.30
                utilHi = 0.05 + utilHi * 0.15    # 0.05 - 0.20
                utilCap = 0.15 + utilCap * 0.25  # 0.15 - 0.40
            elif loading_mode == 'high':
                bufLow *= 0.70      # 0.00 - 0.70
                utilHi = 0.30 + utilHi * 0.50    # 0.30 - 0.80
                utilCap = 0.50 + utilCap * 0.50  # 0.50 - 1.00
            
            logger.info(f"🔧 Mode: {loading_mode}, Thresholds: bufLow={bufLow:.3f}, utilHi={utilHi:.3f}, utilCap={utilCap:.3f}")
            
            # 4. DECISION: Close links with low activity
            closed_links = 0
            for k, (buf, util, _) in enumerate(stats):
                # Calculate buffer trend (weighted average)
                buffer_trend = list(self.buffer_history[k])
                if len(buffer_trend) > 1:
                    weights = np.exp(np.linspace(0, 1, len(buffer_trend)))
                    weighted_avg = np.average(buffer_trend, weights=weights)
                    buffer_change = buffer_trend[-1] - weighted_avg
                else:
                    buffer_change = 0.0
                
                # CLOSE LINK if: low buffer AND low utilization AND stable/decreasing trend
                if buf < bufLow and util < utilHi and buffer_change <= 0.05:
                    upmask[k] = False
                    closed_links += 1
            
            # 5. SAFETY: Ensure network connectivity
            deg = {}  # Track node degrees
            for k, (u, v) in enumerate(self._pairs):
                deg[u] = deg.get(u, 0) + upmask[k]
                deg[v] = deg.get(v, 0) + upmask[k]
            
            # Force at least one link per node
            for k, (u, v) in enumerate(self._pairs):
                if not upmask[k] and (deg[u] == 0 or deg[v] == 0):
                    upmask[k] = True  # Keep link up if node would be isolated
                    deg[u] += 1
                    deg[v] += 1
            
            # 6. HYSTERESIS: Prevent rapid switching
            for k in range(len(upmask)):
                if self.prev_states[k] != upmask[k]:
                    if self.last_link_change[k] < self.min_link_state_duration:
                        upmask[k] = self.prev_states[k]  # Keep previous state
                    else:
                        self.last_link_change[k] = 0  # Reset timer
                else:
                    self.last_link_change[k] += self.step_t
            
            # 7. Update previous states and return closed links
            self.prev_states = {k: upmask[k] for k in range(len(upmask))}
            
            # Return list of closed link names
            closed_link_names = [
                self.link_names[k] for k in range(len(upmask)) 
                if not upmask[k]
            ]
            
            logger.info(f"🤖 Heuristic closed {len(closed_link_names)} links: {closed_link_names}")
            return closed_link_names
            
        except Exception as e:
            logger.error(f"❌ Error in heuristic step: {e}")
            return []
    
    def get_network_state_summary(self) -> Dict:
        """Get summary of current network state"""
        return {
            "loading_mode": self.current_loading_mode,
            "total_links": len(self.link_names),
            "active_links": sum(1 for state in self.prev_states.values() if state),
            "buffer_history_length": {k: len(v) for k, v in self.buffer_history.items()},
            "last_changes": dict(self.last_link_change)
        }
