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
        self._current_telemetry_data = {}  # Store current telemetry for bidirectional calculations
        
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
            (9, 11), # S10-S12
            (9, 12), # S10-S13
            (9, 13), # S10-S14
            (9, 14), # S10-S15
            (9, 15), # S10-S16
            (9, 16), # S10-S17
            (10, 14), # S11-S15
            (11, 14), # S12-S15
            (12, 14), # S13-S15
            (13, 14), # S14-S15
            (14, 15), # S15-S16
            (14, 16), # S15-S17
        ]
        
        # Convert to bidirectional pairs (total: 68 pairs)
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
        
        # LIFO tracking for link reopening
        self.closed_links_stack = deque()  # Stack of closed links (LIFO order)
        self.link_traffic_before_closure = {}  # Store traffic before closing for redistribution calculation
        
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
                    
                    # Handle different data formats
                    if isinstance(data, dict):
                        traffic = data.get('traffic', 0)
                        max_capacity = data.get('max-capacity', 1000)
                    elif isinstance(data, (int, float)):
                        traffic = float(data)
                        max_capacity = 8000  # Default capacity for rate data
                    else:
                        continue
                    
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
                        max_capacity = 8000  # Default value for rate data (bytes/sec)
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
    
    def _get_current_telemetry_data(self) -> Dict:
        """Get current telemetry data for bidirectional calculations"""
        return self._current_telemetry_data

    def step(self, action: np.ndarray, telemetry_data: Dict) -> List[str]:
        """
        Execute one step of the heuristic algorithm
        
        Args:
            action: RL model output thresholds [bufLow, utilHi, utilCap]
            telemetry_data: Current telemetry data
            
        Returns:
            List of link names to close
        """
        try:
            # Store current telemetry data for bidirectional calculations
            self._current_telemetry_data = telemetry_data
            
            # Extract thresholds from RL model output
            bufLow, utilHi, utilCap = action[0], action[1], action[2]
            
            logger.info(f"🎯 RL MODEL THRESHOLDS: bufLow={bufLow:.4f}, utilHi={utilHi:.4f}, utilCap={utilCap:.4f}")
            print(f"🎯 RL MODEL THRESHOLDS: bufLow={bufLow:.4f}, utilHi={utilHi:.4f}, utilCap={utilCap:.4f}")
            logger.info(f"🎯 Heuristic step with thresholds: bufLow={bufLow:.3f}, utilHi={utilHi:.3f}, utilCap={utilCap:.3f}")
            logger.info(f"📊 Telemetry data has {len(telemetry_data)} links")
            logger.info(f"🔗 Algorithm configured for {len(self.link_names)} links: {self.link_names[:5]}...")
            
            # Update loading mode based on current traffic
            loading_mode = self.update_loading_mode(telemetry_data)
            logger.info(f"📊 Current loading mode: {loading_mode}")
            
            # Step 1: Calculate buffer and utilization statistics
            stats = self.calculate_buffer_utilization_stats(telemetry_data)
            logger.info(f"📈 Calculated stats for {len(stats)} links")
            
            # Step 2: Apply thresholds to determine candidate links
            upmask = self.apply_thresholds(stats, bufLow, utilHi, utilCap)
            logger.info(f"🎯 After thresholds: {sum(upmask)} links remain open")
            
            # Step 3: Update link states and durations
            self.update_link_states(upmask)
            
            # Step 4: Check current overload and reopen links if needed (LIFO)
            upmask = self._check_current_overload_and_reopen(upmask, telemetry_data, utilCap)
            logger.info(f"🚨 After overload check: {sum(upmask)} links remain open")
            
            # Step 5: Apply hysteresis (prevent rapid switching)
            upmask = self.apply_hysteresis(upmask)
            logger.info(f"⏱️ After hysteresis: {sum(upmask)} links remain open")
            
            # Step 6: Apply safety checks (prevent node isolation) - FINAL SAFETY CHECK
            upmask = self.apply_safety_checks(upmask)
            logger.info(f"🛡️ After final safety checks: {sum(upmask)} links remain open")
            
            # Step 7: Store traffic before closure for redistribution calculation
            self._store_traffic_before_closure(upmask, telemetry_data)
            
            # Step 8: Check redistribution overload and reopen if needed (LIFO)
            upmask = self._check_redistribution_and_reopen(upmask, telemetry_data, utilCap)
            logger.info(f"📊 After redistribution check: {sum(upmask)} links remain open")
            
            # Step 9: Update closed links stack and return results
            closed_link_names = []
            for k in range(len(upmask)):
                if not upmask[k]:
                    link_name = self.link_names[k]
                    if link_name in telemetry_data:
                        closed_link_names.append(link_name)
            
            logger.info(f"🔗 Final decision: {len(closed_link_names)} links to close: {closed_link_names}")
            return closed_link_names
            
        except Exception as e:
            logger.error(f"❌ Error in heuristic step: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _check_current_overload_and_reopen(self, upmask: List[bool], telemetry_data: Dict, utilCap: float) -> List[bool]:
        """
        Check if any currently open links are overloaded and reopen closed links in LIFO order to relieve overload.
        This is a proactive check at the start of each algorithm run.
{{ ... }}
        Args:
            upmask: Current link states (True=open, False=closed)
            telemetry_data: Current telemetry data
            utilCap: Maximum utilization threshold
            
        Returns:
            Updated upmask after potential link reopening
        """
        try:
            # Check current utilization of all open links
            overloaded_links = []
            open_links = [k for k in range(len(upmask)) if upmask[k]]
            
            for k in open_links:
                link_name = self.link_names[k]
                if link_name in telemetry_data:
                    data = telemetry_data[link_name]
                    if isinstance(data, dict):
                        current_traffic = data.get('traffic', 0)
                        max_capacity = data.get('max-capacity', 1000)
                    elif isinstance(data, (int, float)):
                        current_traffic = float(data)
                        max_capacity = 8000  # Default capacity for rate data
                    else:
                        continue
                    
                    if max_capacity > 0:
                        current_utilization = current_traffic / max_capacity
                        
                        if current_utilization > utilCap:
                            overloaded_links.append((k, link_name, current_utilization))
            
            # If overload detected, reopen closed links in LIFO order
            if overloaded_links:
                logger.warning(f"🚨 PROACTIVE: Found {len(overloaded_links)} overloaded links: {[name for _, name, _ in overloaded_links]}")
                
                # Reopen links from stack (LIFO: last closed, first opened)
                reopened_count = 0
                while self.closed_links_stack and overloaded_links:
                    # Pop from stack (most recently closed link)
                    try:
                        item = self.closed_links_stack.pop()
                        if isinstance(item, tuple) and len(item) >= 3:
                            k, link_name, timestamp = item
                        else:
                            logger.error(f"❌ Invalid item in LIFO stack: {item}")
                            continue
                    except Exception as e:
                        logger.error(f"❌ Error popping from LIFO stack: {e}")
                        break
                    
                    # Reopen the link
                    if k < len(upmask):
                        upmask[k] = True
                        reopened_count += 1
                        logger.info(f"🔓 PROACTIVE: Reopening link {link_name} (index {k}) to relieve current overload")
                        
                        # Remove stored traffic data
                        if link_name in self.link_traffic_before_closure:
                            del self.link_traffic_before_closure[link_name]
                        
                        # Recheck overload conditions with the newly opened link
                        new_open_links = [k for k in range(len(upmask)) if upmask[k]]
                        overloaded_links = []
                        
                        for k_check in new_open_links:
                            link_name_check = self.link_names[k_check]
                            if link_name_check in telemetry_data:
                                data = telemetry_data[link_name_check]
                                if isinstance(data, dict):
                                    current_traffic = data.get('traffic', 0)
                                    max_capacity = data.get('max-capacity', 1000)
                                elif isinstance(data, (int, float)):
                                    current_traffic = float(data)
                                    max_capacity = 8000
                                else:
                                    continue
                                
                                if max_capacity > 0:
                                    current_utilization = current_traffic / max_capacity
                                
                                    if current_utilization > utilCap:
                                        overloaded_links.append((k_check, link_name_check, current_utilization))
                
                if reopened_count > 0:
                    logger.info(f"🔄 PROACTIVE: Reopened {reopened_count} links to relieve existing overload")
            else:
                logger.debug("✅ PROACTIVE: No overloaded links detected, proceeding with normal algorithm")
            
            return upmask
            
        except Exception as e:
            logger.error(f"❌ Error in proactive overload check: {e}")
            return upmask

    def _check_redistribution_and_reopen(self, upmask: List[bool], telemetry_data: Dict, utilCap: float) -> List[bool]:
        """
        Check if traffic redistribution from closed links causes overload on remaining links.
        If so, reopen links in LIFO order (last closed, first opened) until utilization is acceptable.
        
        Args:
            upmask: Current link states (True=open, False=closed)
            telemetry_data: Current telemetry data
            utilCap: Maximum utilization threshold
            
        Returns:
            Updated upmask after potential link reopening
        """
        try:
            # Calculate total redistributed traffic from closed links
            total_redistributed_traffic = 0
            for k, link_name in enumerate(self.link_names):
                if not upmask[k] and link_name in self.link_traffic_before_closure:
                    total_redistributed_traffic += self.link_traffic_before_closure[link_name]
            
            if total_redistributed_traffic == 0:
                return upmask  # No redistribution needed
            
            # Count open links for redistribution
            open_links = [k for k in range(len(upmask)) if upmask[k]]
            if len(open_links) == 0:
                logger.warning("⚠️ No open links available for redistribution!")
                return upmask
            
            # Calculate average additional traffic per open link
            avg_additional_traffic = total_redistributed_traffic / len(open_links)
            
            # Check if any open link would exceed utilCap after redistribution
            overloaded_links = []
            for k in open_links:
                link_name = self.link_names[k]
                if link_name in telemetry_data:
                    current_traffic = telemetry_data[link_name].get('traffic', 0)
                    max_capacity = telemetry_data[link_name].get('max-capacity', 1000)
                    
                    if max_capacity > 0:
                        new_traffic = current_traffic + avg_additional_traffic
                        new_utilization = new_traffic / max_capacity
                        
                        if new_utilization > utilCap:
                            overloaded_links.append((k, link_name, new_utilization))
            
            # If overload detected, reopen links in LIFO order
            if overloaded_links:
                logger.warning(f"⚠️ Redistribution would overload {len(overloaded_links)} links: {[name for _, name, _ in overloaded_links]}")
                
                # Reopen links from stack (LIFO: last closed, first opened)
                reopened_count = 0
                while self.closed_links_stack and overloaded_links:
                    # Pop from stack (most recently closed link)
                    try:
                        item = self.closed_links_stack.pop()
                        if isinstance(item, tuple) and len(item) >= 3:
                            k, link_name, timestamp = item
                        else:
                            logger.error(f"❌ Invalid item in LIFO stack: {item}")
                            continue
                    except Exception as e:
                        logger.error(f"❌ Error popping from LIFO stack: {e}")
                        break
                    
                    # Reopen the link
                    if k < len(upmask):
                        upmask[k] = True
                        reopened_count += 1
                        logger.info(f"📈 Reopening link {link_name} (index {k}) to prevent overload")
                        
                        # Remove stored traffic data
                        if link_name in self.link_traffic_before_closure:
                            del self.link_traffic_before_closure[link_name]
                    
                    # Recalculate redistribution with one more open link
                    new_open_links = [k for k in range(len(upmask)) if upmask[k]]
                    if len(new_open_links) > 0:
                        new_avg_additional = total_redistributed_traffic / len(new_open_links)
                        
                        # Recheck overload conditions
                        overloaded_links = []
                        for k_check in new_open_links:
                            link_name_check = self.link_names[k_check]
                            if link_name_check in telemetry_data:
                                current_traffic = telemetry_data[link_name_check].get('traffic', 0)
                                max_capacity = telemetry_data[link_name_check].get('max-capacity', 1000)
                                
                                if max_capacity > 0:
                                    new_traffic = current_traffic + new_avg_additional
                                    new_utilization = new_traffic / max_capacity
                                    
                                    if new_utilization > utilCap:
                                        overloaded_links.append((k_check, link_name_check, new_utilization))
                
                if reopened_count > 0:
                    logger.info(f"🔄 Reopened {reopened_count} links to prevent redistribution overload")
            
            return upmask
            
        except Exception as e:
            logger.error(f"❌ Error in redistribution check: {e}")
            return upmask

    def calculate_buffer_utilization_stats(self, telemetry_data: Dict) -> List[Tuple[float, float, bool]]:
        """
        Calculate buffer and utilization statistics for all links
        
        Args:
            telemetry_data: Current telemetry data
            
        Returns:
            List of (buffer_utilization, link_utilization, is_up) for each link
        """
        return self.get_link_stats(telemetry_data)
    
    def _find_bidirectional_pairs(self, stats: List[Tuple[float, float, bool]]) -> Dict[str, Tuple[int, int]]:
        """
        Find bidirectional link pairs in the network
        
        Args:
            stats: List of (buffer_util, link_util, is_up) for each link
            
        Returns:
            Dict mapping canonical link name to (index1, index2) tuple
        """
        bidirectional_pairs = {}
        
        for i, link_name in enumerate(self.link_names):
            if '-' not in link_name:
                continue
                
            src, dst = link_name.split('-')
            reverse_link = f"{dst}-{src}"
            
            # Find the reverse link index
            reverse_idx = None
            for j, reverse_name in enumerate(self.link_names):
                if reverse_name == reverse_link:
                    reverse_idx = j
                    break
            
            if reverse_idx is not None:
                # Create canonical name (alphabetically sorted)
                canonical_name = f"{min(src, dst)}-{max(src, dst)}"
                
                # Only add if not already processed
                if canonical_name not in bidirectional_pairs:
                    bidirectional_pairs[canonical_name] = (i, reverse_idx)
                    
        logger.info(f"🔗 Found {len(bidirectional_pairs)} bidirectional link pairs")
        return bidirectional_pairs
    
    def _calculate_bidirectional_traffic(self, telemetry_data: Dict, link_idx1: int, link_idx2: int) -> Tuple[float, float]:
        """
        Calculate combined traffic and utilization for a bidirectional link pair
        
        Args:
            telemetry_data: Current telemetry data
            link_idx1: Index of first direction
            link_idx2: Index of second direction
            
        Returns:
            Tuple of (combined_traffic, combined_utilization)
        """
        link_name1 = self.link_names[link_idx1]
        link_name2 = self.link_names[link_idx2]
        
        traffic1, capacity1 = 0, 1000
        traffic2, capacity2 = 0, 1000
        
        # Get traffic for first direction
        if link_name1 in telemetry_data:
            data1 = telemetry_data[link_name1]
            if isinstance(data1, dict):
                traffic1 = data1.get('traffic', 0)
                capacity1 = data1.get('max-capacity', 1000)
            elif isinstance(data1, (int, float)):
                traffic1 = float(data1)
                capacity1 = 8000  # Default capacity for rate data
        
        # Get traffic for second direction
        if link_name2 in telemetry_data:
            data2 = telemetry_data[link_name2]
            if isinstance(data2, dict):
                traffic2 = data2.get('traffic', 0)
                capacity2 = data2.get('max-capacity', 1000)
            elif isinstance(data2, (int, float)):
                traffic2 = float(data2)
                capacity2 = 8000  # Default capacity for rate data
        
        # Sum traffic from both directions
        combined_traffic = traffic1 + traffic2
        
        # Use the maximum capacity of both directions for utilization calculation
        max_capacity = max(capacity1, capacity2)
        combined_utilization = combined_traffic / max_capacity if max_capacity > 0 else 0.0
        
        return combined_traffic, combined_utilization

    def apply_thresholds(self, stats: List[Tuple[float, float, bool]], bufLow: float, utilHi: float, utilCap: float) -> List[bool]:
        """
        Apply adaptive thresholds to determine which links should remain open
        
        CRITICAL CHANGE: Only process bidirectional link pairs. Unidirectional links
        are automatically kept open and do not go through the closure decision process.
        
        Args:
            stats: List of (buffer_util, link_util, is_up) for each link
            bufLow: Buffer utilization threshold from RL model
            utilHi: High utilization threshold from RL model  
            utilCap: Capacity utilization threshold from RL model
            
        Returns:
            List of booleans indicating which links should remain open
        """
        # Initialize all links as open (default state)
        upmask = [True] * len(stats)
        
        # Find bidirectional pairs - only these can be considered for closure
        bidirectional_pairs = self._find_bidirectional_pairs(stats)
        
        if not bidirectional_pairs:
            logger.warning("⚠️ No bidirectional pairs found - keeping all links open")
            return upmask
        
        # Calculate overall traffic level to adapt thresholds (using bidirectional pairs only)
        total_utilization = 0
        active_pairs = 0
        
        for canonical_name, (idx1, idx2) in bidirectional_pairs.items():
            _, link_util1, is_up1 = stats[idx1]
            _, link_util2, is_up2 = stats[idx2]
            
            if is_up1 and is_up2:  # Both directions must be up
                # Use combined utilization for the pair
                combined_traffic, combined_utilization = self._calculate_bidirectional_traffic(
                    self._get_current_telemetry_data(), idx1, idx2
                )
                total_utilization += combined_utilization
                active_pairs += 1
        
        avg_utilization = total_utilization / max(1, active_pairs)
        
        # Adaptive threshold scaling based on traffic level
        if avg_utilization < 0.1:  # Very low traffic scenario
            # Scale down thresholds significantly for low traffic
            adaptive_bufLow = bufLow * 0.3  # Much lower buffer threshold
            adaptive_utilHi = utilHi * 0.2  # Much lower utilization threshold
            logger.info(f"🔽 Very low traffic detected (avg={avg_utilization:.3f}), scaling thresholds down: bufLow={adaptive_bufLow:.3f}, utilHi={adaptive_utilHi:.3f}")
        elif avg_utilization < 0.3:  # Low traffic scenario
            # Moderate scaling for low traffic
            adaptive_bufLow = bufLow * 0.6
            adaptive_utilHi = utilHi * 0.5
            logger.info(f"📉 Low traffic detected (avg={avg_utilization:.3f}), scaling thresholds: bufLow={adaptive_bufLow:.3f}, utilHi={adaptive_utilHi:.3f}")
        else:  # Normal/high traffic
            # Use original thresholds
            adaptive_bufLow = bufLow
            adaptive_utilHi = utilHi
            logger.info(f"📊 Normal traffic (avg={avg_utilization:.3f}), using original thresholds")
        
        # Process only bidirectional pairs for closure decisions
        logger.info(f"🔍 Processing {len(bidirectional_pairs)} bidirectional pairs for closure decisions")
        
        for canonical_name, (idx1, idx2) in bidirectional_pairs.items():
            buffer_util1, link_util1, is_up1 = stats[idx1]
            buffer_util2, link_util2, is_up2 = stats[idx2]
            
            # Both directions must be up to consider for closure
            if not (is_up1 and is_up2):
                logger.debug(f"⏭️ Skipping {canonical_name}: one or both directions down")
                continue
            
            # Calculate combined traffic utilization for the bidirectional pair
            combined_traffic, combined_utilization = self._calculate_bidirectional_traffic(
                self._get_current_telemetry_data(), idx1, idx2
            )
            
            # Calculate buffer trends for both directions
            buffer_trend1 = buffer_util1
            buffer_trend2 = buffer_util2
            
            if len(self.buffer_history[idx1]) > 1:
                recent_buffers1 = list(self.buffer_history[idx1])[-5:]
                weights = np.array([0.1, 0.2, 0.3, 0.4, 1.0])[-len(recent_buffers1):]
                buffer_trend1 = np.average(recent_buffers1, weights=weights)
            
            if len(self.buffer_history[idx2]) > 1:
                recent_buffers2 = list(self.buffer_history[idx2])[-5:]
                weights = np.array([0.1, 0.2, 0.3, 0.4, 1.0])[-len(recent_buffers2):]
                buffer_trend2 = np.average(recent_buffers2, weights=weights)
            
            # Use the maximum buffer trend (most congested direction)
            max_buffer_trend = max(buffer_trend1, buffer_trend2)
            
            # Decision logic based on loading mode using COMBINED utilization
            if self.current_loading_mode == 'low':
                # In low loading mode, be more aggressive about closing links
                # Close if buffer is low AND combined utilization is low
                should_close = (max_buffer_trend <= adaptive_bufLow and combined_utilization <= adaptive_utilHi)
            else:
                # In high loading mode, be more conservative
                # Only close if buffer is very low AND combined utilization is moderate
                should_close = (max_buffer_trend <= adaptive_bufLow * 0.5 and combined_utilization <= adaptive_utilHi * 0.7)
            
            # Always keep link open if combined utilization exceeds capacity threshold
            if combined_utilization > utilCap:
                should_close = False
            
            # Apply decision to BOTH directions of the bidirectional pair
            if should_close:
                upmask[idx1] = False
                upmask[idx2] = False
                logger.info(f"🔴 Bidirectional pair {canonical_name}: buffer={max_buffer_trend:.3f} <= {adaptive_bufLow:.3f}, combined_util={combined_utilization:.3f} <= {adaptive_utilHi:.3f} -> CLOSE BOTH")
            else:
                # Keep both directions open (they're already True by default)
                logger.debug(f"🟢 Bidirectional pair {canonical_name}: buffer={max_buffer_trend:.3f}, combined_util={combined_utilization:.3f} -> KEEP OPEN")
        
        logger.info(f"📊 Bidirectional closure decisions: {len(bidirectional_pairs)} pairs processed, {sum(1 for x in upmask if not x)} total links marked for closure")
        
        return upmask
    
    def update_link_states(self, upmask: List[bool]) -> None:
        """
        Update link states and timing information
        
        Args:
            upmask: Current link states (True=open, False=closed)
        """
        current_time = time.time()
        
        for i, is_up in enumerate(upmask):
            # Update state change timing
            if self.prev_states.get(i, True) != is_up:
                self.last_link_change[i] = current_time
            
            self.prev_states[i] = is_up
    
    def apply_safety_checks(self, upmask: List[bool]) -> List[bool]:
        """
        Apply safety checks to prevent network partitioning and node isolation
        Uses GLOBAL analysis of all proposed closures together, not individual link testing
        
        Args:
            upmask: Current link states
            
        Returns:
            Updated upmask after safety checks
        """
        safety_upmask = upmask.copy()
        
        # Count bidirectional connections per node (physical links, not directional)
        def count_node_connections(current_upmask):
            node_connections = defaultdict(set)  # Use set to avoid double counting bidirectional pairs
            
            for i, is_up in enumerate(current_upmask):
                if is_up:
                    link_name = self.link_names[i]
                    if '-' in link_name:
                        src, dst = link_name.split('-')
                        # Create canonical connection (bidirectional pair)
                        canonical = tuple(sorted([src, dst]))
                        node_connections[src].add(canonical)
                        node_connections[dst].add(canonical)
            
            return {node: len(connections) for node, connections in node_connections.items()}
        
        # CRITICAL FIX: Analyze the FINAL state after ALL proposed closures
        final_connections = count_node_connections(safety_upmask)
        
        # CRITICAL BUG FIX: Find ALL nodes in topology, not just those with connections
        all_nodes = set()
        for link_name in self.link_names:
            if '-' in link_name:
                src, dst = link_name.split('-')
                all_nodes.add(src)
                all_nodes.add(dst)
        
        # Find nodes that would be isolated or have insufficient connectivity
        at_risk_nodes = []
        for node in all_nodes:
            conn_count = final_connections.get(node, 0)  # Default to 0 if not in dict
            if conn_count < 2:  # Need at least 2 connections for redundancy
                at_risk_nodes.append((node, conn_count))
        
        if at_risk_nodes:
            logger.warning(f"🚨 GLOBAL SAFETY VIOLATION: Nodes at risk: {at_risk_nodes}")
            
            # Find all bidirectional pairs that involve at-risk nodes
            pairs_to_preserve = set()
            
            for node, _ in at_risk_nodes:
                # Find all links involving this at-risk node
                for i, link_name in enumerate(self.link_names):
                    if not safety_upmask[i] and '-' in link_name:  # Link marked for closure
                        src, dst = link_name.split('-')
                        if src == node or dst == node:
                            # This link involves an at-risk node - preserve the bidirectional pair
                            canonical_pair = tuple(sorted([src, dst]))
                            pairs_to_preserve.add(canonical_pair)
                            logger.info(f"🛡️ Preserving pair {canonical_pair} to protect {node}")
            
            # Reopen all links in the pairs that need to be preserved
            reopened_count = 0
            for i, link_name in enumerate(self.link_names):
                if not safety_upmask[i] and '-' in link_name:  # Link marked for closure
                    src, dst = link_name.split('-')
                    canonical_pair = tuple(sorted([src, dst]))
                    
                    if canonical_pair in pairs_to_preserve:
                        safety_upmask[i] = True
                        reopened_count += 1
                        logger.info(f"🔓 Reopening {link_name} for safety")
            
            logger.info(f"🛡️ Safety intervention: Reopened {reopened_count} links to prevent isolation")
        
        # Final verification after safety interventions
        final_connections_after = count_node_connections(safety_upmask)
        still_at_risk = [node for node, count in final_connections_after.items() if count < 2]
        
        if still_at_risk:
            logger.error(f"❌ CRITICAL SAFETY FAILURE: Nodes {still_at_risk} would still be isolated!")
            # Emergency fallback - reopen ALL links to ensure safety
            logger.error("🚨 EMERGENCY: Reopening all links to prevent network partition")
            safety_upmask = [True] * len(safety_upmask)
        else:
            logger.info(f"✅ Global safety check passed: All nodes have adequate connectivity")
        
        return safety_upmask
    
    def apply_hysteresis(self, upmask: List[bool]) -> List[bool]:
        """
        Apply hysteresis to prevent rapid link state changes
        
        Args:
            upmask: Current link states
            
        Returns:
            Updated upmask after hysteresis
        """
        current_time = time.time()
        hysteresis_upmask = upmask.copy()
        
        for i, new_state in enumerate(upmask):
            old_state = self.prev_states.get(i, True)
            time_since_change = current_time - self.last_link_change.get(i, 0)
            
            # If state would change and not enough time has passed, keep old state
            if new_state != old_state and time_since_change < self.min_link_state_duration:
                hysteresis_upmask[i] = old_state
                logger.debug(f"⏱️ Hysteresis: Keeping link {self.link_names[i]} in state {old_state} (changed {time_since_change:.1f}s ago)")
        
        return hysteresis_upmask
    
    def _store_traffic_before_closure(self, upmask: List[bool], telemetry_data: Dict) -> None:
        """
        Store traffic data for links that are about to be closed
        
        Args:
            upmask: Current link states
            telemetry_data: Current telemetry data
        """
        for i, is_up in enumerate(upmask):
            link_name = self.link_names[i]
            
            if not is_up and link_name in telemetry_data:  # Link is being closed
                data = telemetry_data[link_name]
                
                if isinstance(data, dict):
                    traffic = data.get('traffic', 0)
                elif isinstance(data, (int, float)):
                    traffic = float(data)
                else:
                    traffic = 0
                
                # Store traffic and add to closed links stack
                self.link_traffic_before_closure[link_name] = traffic
                
                # Add to LIFO stack if not already there
                try:
                    # Check if link index is already in stack
                    already_in_stack = False
                    for item in self.closed_links_stack:
                        if isinstance(item, tuple) and len(item) >= 3:
                            k, _, _ = item
                            if k == i:
                                already_in_stack = True
                                break
                    
                    if not already_in_stack:
                        self.closed_links_stack.append((i, link_name, time.time()))
                        logger.debug(f"📚 Stored traffic {traffic} for closing link {link_name}")
                except Exception as e:
                    logger.error(f"❌ Error checking LIFO stack: {e}")
                    # Clear corrupted stack and add current item
                    self.closed_links_stack.clear()
                    self.closed_links_stack.append((i, link_name, time.time()))

    def get_network_state_summary(self) -> Dict:
        """Get summary of current network state"""
        return {
            "loading_mode": self.current_loading_mode,
            "total_links": len(self.link_names),
            "active_links": sum(1 for state in self.prev_states.values() if state),
            "closed_links_in_stack": len(self.closed_links_stack),
            "buffer_history_length": {k: len(v) for k, v in self.buffer_history.items()},
            "last_changes": dict(self.last_link_change),
            "stored_traffic_data": len(self.link_traffic_before_closure)
        }
