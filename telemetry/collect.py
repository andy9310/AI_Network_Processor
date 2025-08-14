'''
# collect traffic data from eveng 
# using pipe to send to (1) prompt processor (2) eveng model 
** try to use subscriber and publisher system
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集 EVE-NG SDN 節點下各 XR 裝置介面的 Generic Counters
--------------------------------------------------------

1. 先拉 Netconf topology，取得所有 XR nodes (node-id)
2. 對每個 node 再列出 interface 名稱
3. 針對每個 interface 取 generic-counters
4. 將結果印出或寫檔
"""
from flask import Flask, Response
import urllib3
import requests
from requests.auth import HTTPBasicAuth
import urllib.parse
import json
import time
import sys
import ipaddress
import re
from typing import Dict, List, Tuple

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)

# Base configuration
BASE_URL = "http://192.168.10.22:8181/restconf"
AUTH = HTTPBasicAuth("admin", "admin")
HEADERS = {"Accept": "application/json"}

# Node mapping: S1 -> node1, S2 -> node2, etc.
NODE_MAPPING = {
    f"S{i}": f"node{i}" for i in range(1, 18)  # S1-S17 -> node1-node17
}

# Interface mapping for each link
# This maps link pairs to their corresponding interfaces
INTERFACE_MAPPING = {
    # S1 interfaces
    "S1-S2": "GigabitEthernet0/0/0/0",
    "S1-S3": "GigabitEthernet0/0/0/1", 
    "S1-S4": "GigabitEthernet0/0/0/2",
    "S1-S9": "GigabitEthernet0/0/0/3",
    
    # S2 interfaces
    "S2-S1": "GigabitEthernet0/0/0/0",
    "S2-S4": "GigabitEthernet0/0/0/1",
    "S2-S9": "GigabitEthernet0/0/0/2",
    
    # S3 interfaces
    "S3-S1": "GigabitEthernet0/0/0/0",
    "S3-S4": "GigabitEthernet0/0/0/1",
    "S3-S9": "GigabitEthernet0/0/0/2",
    
    # S4 interfaces (hub switch with many connections)
    "S4-S1": "GigabitEthernet0/0/0/0",
    "S4-S2": "GigabitEthernet0/0/0/1",
    "S4-S3": "GigabitEthernet0/0/0/2",
    "S4-S5": "GigabitEthernet0/0/0/3",
    "S4-S6": "GigabitEthernet0/0/0/4",
    "S4-S7": "GigabitEthernet0/0/0/5",
    "S4-S8": "GigabitEthernet0/0/0/6",
    "S4-S9": "GigabitEthernet0/0/0/7",
    "S4-S10": "GigabitEthernet0/0/0/8",
    "S4-S11": "GigabitEthernet0/0/0/9",
    "S4-S15": "GigabitEthernet0/0/0/10",
    
    # S5 interfaces
    "S5-S4": "GigabitEthernet0/0/0/0",
    "S5-S9": "GigabitEthernet0/0/0/1",
    
    # S6 interfaces
    "S6-S4": "GigabitEthernet0/0/0/0",
    "S6-S15": "GigabitEthernet0/0/0/1",
    
    # S7 interfaces
    "S7-S4": "GigabitEthernet0/0/0/0",
    "S7-S9": "GigabitEthernet0/0/0/1",
    
    # S8 interfaces
    "S8-S4": "GigabitEthernet0/0/0/0",
    "S8-S9": "GigabitEthernet0/0/0/1",
    
    # S9 interfaces (another hub)
    "S9-S1": "GigabitEthernet0/0/0/0",
    "S9-S2": "GigabitEthernet0/0/0/1",
    "S9-S3": "GigabitEthernet0/0/0/2",
    "S9-S4": "GigabitEthernet0/0/0/3",
    "S9-S5": "GigabitEthernet0/0/0/4",
    "S9-S7": "GigabitEthernet0/0/0/5",
    "S9-S8": "GigabitEthernet0/0/0/6",
    "S9-S10": "GigabitEthernet0/0/0/7",
    "S9-S15": "GigabitEthernet0/0/0/8",
    
    # S10 interfaces (major hub)
    "S10-S4": "GigabitEthernet0/0/0/0",
    "S10-S9": "GigabitEthernet0/0/0/1",
    "S10-S12": "GigabitEthernet0/0/0/2",
    "S10-S13": "GigabitEthernet0/0/0/3",
    "S10-S14": "GigabitEthernet0/0/0/4",
    "S10-S15": "GigabitEthernet0/0/0/5",
    "S10-S16": "GigabitEthernet0/0/0/6",
    "S10-S17": "GigabitEthernet0/0/0/7",
    
    # S11 interfaces
    "S11-S4": "GigabitEthernet0/0/0/0",
    "S11-S15": "GigabitEthernet0/0/0/1",
    
    # S12 interfaces
    "S12-S10": "GigabitEthernet0/0/0/0",
    "S12-S15": "GigabitEthernet0/0/0/1",
    
    # S13 interfaces
    "S13-S10": "GigabitEthernet0/0/0/0",
    "S13-S15": "GigabitEthernet0/0/0/1",
    
    # S14 interfaces
    "S14-S10": "GigabitEthernet0/0/0/0",
    "S14-S15": "GigabitEthernet0/0/0/1",
    
    # S15 interfaces (major hub)
    "S15-S4": "GigabitEthernet0/0/0/0",
    "S15-S6": "GigabitEthernet0/0/0/1",
    "S15-S9": "GigabitEthernet0/0/0/2",
    "S15-S10": "GigabitEthernet0/0/0/3",
    "S15-S11": "GigabitEthernet0/0/0/4",
    "S15-S12": "GigabitEthernet0/0/0/5",
    "S15-S13": "GigabitEthernet0/0/0/6",
    "S15-S14": "GigabitEthernet0/0/0/7",
    "S15-S16": "GigabitEthernet0/0/0/8",
    "S15-S17": "GigabitEthernet0/0/0/9",
    
    # S16 interfaces
    "S16-S10": "GigabitEthernet0/0/0/0",
    "S16-S15": "GigabitEthernet0/0/0/1",
    
    # S17 interfaces
    "S17-S10": "GigabitEthernet0/0/0/0",
    "S17-S15": "GigabitEthernet0/0/0/1",
}

# ---------- Dynamic Interface Mapping Generation ----------
def parse_topology_data(topology_info: Dict) -> Dict[str, Dict]:
    """
    Parse the topology_info data structure to extract interface information
    
    Args:
        topology_info: Dictionary containing topology information from RESTCONF
        
    Returns:
        Dict mapping node_id to their interface information
    """
    nodes = {}
    
    for interface_key, interface_data in topology_info.items():
        node_id = interface_data['node_id']
        interface_name = interface_key.split(':', 1)[1]  # Remove node_id prefix
        
        if node_id not in nodes:
            nodes[node_id] = {}
            
        nodes[node_id][interface_name] = {
            'ip_address': interface_data['ip_address'],
            'netmask': interface_data['netmask'],
            'shutdown': interface_data['shutdown'],
            'active': interface_data['active']
        }
    
    return nodes

def find_connected_interfaces(nodes: Dict[str, Dict]) -> List[Tuple[str, str, str, str]]:
    """
    Find connected interfaces by matching IP subnets
    
    Args:
        nodes: Dictionary of nodes and their interfaces
        
    Returns:
        List of tuples: (node1, interface1, node2, interface2)
    """
    connections = []
    processed_pairs = set()
    
    # Get all interfaces with valid IP addresses
    all_interfaces = []
    for node_id, interfaces in nodes.items():
        for interface_name, interface_data in interfaces.items():
            ip = interface_data['ip_address']
            netmask = interface_data['netmask']
            
            # Skip interfaces with no IP or default IPs
            if ip == '0.0.0.0' or ip.startswith('192.168.'):
                continue
                
            all_interfaces.append((node_id, interface_name, ip, netmask))
    
    # Find matching subnets
    for i, (node1, iface1, ip1, netmask1) in enumerate(all_interfaces):
        for j, (node2, iface2, ip2, netmask2) in enumerate(all_interfaces[i+1:], i+1):
            if node1 == node2:  # Skip same node
                continue
                
            # Check if they're in the same subnet
            try:
                network1 = ipaddress.IPv4Network(f"{ip1}/{netmask1}", strict=False)
                network2 = ipaddress.IPv4Network(f"{ip2}/{netmask2}", strict=False)
                
                if network1.network_address == network2.network_address:
                    # They're connected!
                    pair_key = tuple(sorted([f"{node1}:{iface1}", f"{node2}:{iface2}"]))
                    if pair_key not in processed_pairs:
                        connections.append((node1, iface1, node2, iface2))
                        processed_pairs.add(pair_key)
                        
            except (ipaddress.AddressValueError, ValueError):
                continue
    
    return connections

def generate_interface_mapping(topology_info: Dict) -> Dict[str, str]:
    """
    Generate INTERFACE_MAPPING dynamically from topology data
    
    Args:
        topology_info: The topology_info dictionary from RESTCONF
        
    Returns:
        Dictionary mapping link names (e.g., 'S1-S2') to interface names
    """
    nodes = parse_topology_data(topology_info)
    connections = find_connected_interfaces(nodes)
    
    interface_mapping = {}
    
    # Convert node names to switch names (node1 -> S1, node3 -> S3, etc.)
    for node1, iface1, node2, iface2 in connections:
        # Extract node numbers
        node1_num = re.search(r'node(\d+)', node1)
        node2_num = re.search(r'node(\d+)', node2)
        
        if node1_num and node2_num:
            s1 = f"S{node1_num.group(1)}"
            s2 = f"S{node2_num.group(1)}"
            
            # Create bidirectional mapping
            interface_mapping[f"{s1}-{s2}"] = iface1
            interface_mapping[f"{s2}-{s1}"] = iface2
    
    return interface_mapping

def get_dynamic_interface_mapping() -> Dict[str, str]:
    """
    Get the current dynamic interface mapping by fetching topology info
    
    Returns:
        Dictionary mapping link names to interface names
    """
    try:
        # Import the fetch_topology_info function from restconf_processor
        from restconf_processor import fetch_topology_info
        print("🔄 Fetching current topology to generate dynamic interface mapping...")
        topology_data = fetch_topology_info()
        print("🔍 Debug: Received topology data structure")
        print(f"📊 Data type: {type(topology_data)}")
        print(f"🔑 Keys: {list(topology_data.keys()) if isinstance(topology_data, dict) else 'Not a dict'}")
        
        # Handle both wrapped and unwrapped formats
        if isinstance(topology_data, dict):
            if 'topology_info' in topology_data:
                # Wrapped format
                print("✅ Found wrapped topology_info format")
                dynamic_mapping = generate_interface_mapping(topology_data['topology_info'])
            elif any(key.startswith(('node', 'switch')) and ':' in key for key in topology_data.keys()):
                # Direct format (unwrapped)
                print("✅ Found direct topology format, using as-is")
                dynamic_mapping = generate_interface_mapping(topology_data)
            else:
                print("⚠️  Unknown topology data format, falling back to static mapping")
                return INTERFACE_MAPPING
            
            print(f"✅ Generated {len(dynamic_mapping)} dynamic interface mappings")
            return dynamic_mapping
        else:
            print("⚠️  Invalid topology data format, falling back to static mapping")
            return INTERFACE_MAPPING
            
    except Exception as e:
        print(f"⚠️  Error generating dynamic mapping: {e}")
        print("   Falling back to static INTERFACE_MAPPING")
        return INTERFACE_MAPPING

# ---------- 1. 取得 Netconf topology ----------
def fetch_topology():
    url = f"{BASE_URL}/operational/network-topology:network-topology"
    r = requests.get(url, auth=AUTH, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def list_xr_nodes(topology_json):
    nodes = []
    for topo in topology_json.get("network-topology:network-topology", {}).get("topology", []):
        if topo.get("topology-id") != "topology-netconf":
            continue
        for n in topo.get("node", []):
            nodes.append(n["node-id"])
    return nodes

# ---------- 2. 取得介面名稱清單 ----------
def list_interfaces(node_id):
    url = (
        f"{BASE_URL}/operational/network-topology:network-topology/topology/topology-netconf"
        f"/node/{urllib.parse.quote(node_id, safe='')}"
        "/yang-ext:mount/Cisco-IOS-XR-ifmgr-oper:interface-properties/data-nodes/data-node"
        "/system-view/interfaces/interface"      # ← 取 interface 名稱
    )
    r = requests.get(url, auth=AUTH, headers=HEADERS, timeout=30)
    if r.status_code == 404:         # 沒開 ifmgr model 或 node 離線
        return []
    r.raise_for_status()
    data = r.json()
    return [
        intf["interface-name"]
        for intf in data
        .get("interface-properties", {})
        .get("data-nodes", {})
        .get("data-node", [{}])[0]
        .get("system-view", {})
        .get("interfaces", {})
        .get("interface", [])
    ]

# ---------- 3. 取得 generic-counters for specific node and interface ----------
def fetch_generic_counters(node_id, interface_name):
    """
    Fetch generic counters for a specific node and interface
    
    Args:
        node_id (str): The node ID (e.g., 'node1', 'node2')
        interface_name (str): The interface name (e.g., 'GigabitEthernet0/0/0/0')
    
    Returns:
        dict: Generic counters data
    """
    try:
        # URL encode the interface name for the REST API
        # This converts GigabitEthernet0/0/0/0 to GigabitEthernet0%2F0%2F0%2F0
        encoded_interface = urllib.parse.quote(interface_name, safe='')
        url = f"{BASE_URL}/operational/network-topology:network-topology/topology/topology-netconf/node/{node_id}/yang-ext:mount/Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/{encoded_interface}/generic-counters"
        
        resp = requests.get(url, verify=False, auth=AUTH, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            print(f"Interface {interface_name} not found on node {node_id}")
            return {}
        
        resp.raise_for_status()
        data = resp.json()
        return data.get('generic-counters', {})
    except Exception as e:
        print(f"Error fetching generic counters for node {node_id}, interface {interface_name}: {e}")
        return {}

# ---------- 4. Traffic Rate Calculation ----------
# Global storage for previous measurements to calculate traffic rates
previous_measurements = {}

def calculate_traffic_rate(node_id, interface_name, link_name, interval_seconds=10):
    """
    Calculate real-time traffic rate for a specific interface
    
    Args:
        node_id (str): Node ID (e.g., 'node1')
        interface_name (str): Interface name
        link_name (str): Link identifier for tracking
        interval_seconds (int): Measurement interval in seconds
    
    Returns:
        float: Traffic rate in bytes per second
    """
    current_time = time.time()
    
    # Get current counters
    counters = fetch_generic_counters(node_id, interface_name)
    if not counters:
        return 0.0
    
    current_bytes = counters.get('bytes-received', 0) + counters.get('bytes-sent', 0)
    
    # Check if we have previous measurement for this link
    if link_name in previous_measurements:
        prev_time, prev_bytes = previous_measurements[link_name]
        time_diff = current_time - prev_time
        
        # Only calculate rate if enough time has passed (avoid division by very small numbers)
        if time_diff >= 1.0:  # At least 1 second
            byte_diff = current_bytes - prev_bytes
            # Handle counter rollover (rare but possible)
            if byte_diff < 0:
                print(f"⚠️  Counter rollover detected for {link_name}, resetting...")
                rate = 0.0
            else:
                rate = byte_diff / time_diff  # bytes per second
        else:
            # Too soon, return previous rate if available
            rate = previous_measurements.get(f"{link_name}_rate", 0.0)
    else:
        # First measurement, no rate available yet
        rate = 0.0
    
    # Store current measurement for next calculation
    previous_measurements[link_name] = (current_time, current_bytes)
    previous_measurements[f"{link_name}_rate"] = rate
    
    return rate

def collect_traffic_rates(links, use_dynamic_mapping=True, measurement_interval=10):
    """
    Collect real-time traffic rates for all specified links
    
    Args:
        links (list): List of link strings (e.g., ['S1-S2', 'S2-S1'])
        use_dynamic_mapping (bool): Whether to use dynamic interface mapping
        measurement_interval (int): Seconds to wait between measurements for rate calculation
    
    Returns:
        dict: Traffic rates for all links in bytes/second
    """
    traffic_rates = {}
    
    # Get interface mapping
    if use_dynamic_mapping:
        interface_mapping = get_dynamic_interface_mapping()
        print(f"📋 Using dynamic interface mapping with {len(interface_mapping)} entries")
    else:
        interface_mapping = INTERFACE_MAPPING
        print(f"📋 Using static interface mapping with {len(interface_mapping)} entries")
    
    print(f"⏱️  Calculating traffic rates with {measurement_interval}s interval...")
    
    # First pass: Initialize measurements
    print("🔄 Taking initial measurements...")
    for link in links:
        try:
            src_switch = link.split('-')[0]
            node_id = NODE_MAPPING.get(src_switch)
            interface_name = interface_mapping.get(link)
            
            if node_id and interface_name:
                # Take initial measurement
                calculate_traffic_rate(node_id, interface_name, link, measurement_interval)
        except Exception as e:
            print(f"⚠️  Error in initial measurement for {link}: {e}")
    
    # Wait for measurement interval
    print(f"⏳ Waiting {measurement_interval} seconds for rate calculation...")
    time.sleep(measurement_interval)
    
    # Second pass: Calculate rates
    print("📊 Calculating traffic rates...")
    for link in links:
        try:
            src_switch = link.split('-')[0]
            node_id = NODE_MAPPING.get(src_switch)
            interface_name = interface_mapping.get(link)
            
            if not node_id:
                print(f"Warning: No node mapping found for switch {src_switch}")
                traffic_rates[link] = 0.0
                continue
            
            if not interface_name:
                print(f"Warning: No interface mapping found for link {link}")
                traffic_rates[link] = 0.0
                continue
            
            # Calculate traffic rate
            rate = calculate_traffic_rate(node_id, interface_name, link, measurement_interval)
            traffic_rates[link] = rate
            
            # Convert to human-readable format for display
            if rate > 1_000_000:  # > 1 MB/s
                rate_str = f"{rate/1_000_000:.2f} MB/s"
            elif rate > 1_000:    # > 1 KB/s
                rate_str = f"{rate/1_000:.2f} KB/s"
            else:
                rate_str = f"{rate:.2f} B/s"
            
            print(f"✓ {link:<10} {node_id:<8} {interface_name:<20} Rate={rate_str}")
            
        except Exception as e:
            traffic_rates[link] = 0.0
            print(f"✗ {link:<10} Error: {e} (set to 0.0)")
    
    return traffic_rates

# ---------- 5. Collect traffic data for all links (Legacy - Cumulative) ----------
def collect_link_traffic(links, use_dynamic_mapping=True):
    """
    Collect traffic data for all specified links
    
    Args:
        links (list): List of link strings (e.g., ['S1-S2', 'S2-S1'])
        use_dynamic_mapping (bool): Whether to use dynamic interface mapping (default: True)
    
    Returns:
        dict: Traffic data for all links in the same format as generate_random_traffic
    """
    traffic = {}
    
    # Get interface mapping (dynamic or static)
    if use_dynamic_mapping:
        interface_mapping = get_dynamic_interface_mapping()
        print(f"📋 Using dynamic interface mapping with {len(interface_mapping)} entries")
    else:
        interface_mapping = INTERFACE_MAPPING
        print(f"📋 Using static interface mapping with {len(interface_mapping)} entries")
    
    for link in links:
        try:
            # Parse the link to get source switch
            src_switch = link.split('-')[0]
            
            # Get corresponding node ID
            node_id = NODE_MAPPING.get(src_switch)
            if not node_id:
                print(f"Warning: No node mapping found for switch {src_switch}")
                continue
            
            # Get interface name for this link
            interface_name = interface_mapping.get(link)
            if not interface_name:
                print(f"Warning: No interface mapping found for link {link}")
                continue
            
            # Fetch counters
            counters = fetch_generic_counters(node_id, interface_name)
            
            if counters:
                # Extract traffic data - use total bytes (received + sent) as traffic value
                # Based on actual data structure: "bytes-received" and "bytes-sent"
                bytes_received = counters.get('bytes-received', 0)
                bytes_sent = counters.get('bytes-sent', 0)
                total_traffic = bytes_received + bytes_sent
                
                # Store in the same format as generate_random_traffic
                traffic[link] = total_traffic
                
                print(f"✓ {link:<10} {node_id:<8} {interface_name:<20} Traffic={total_traffic}")
            else:
                # If no data, use 0 as traffic value
                traffic[link] = 0
                print(f"✗ {link:<10} {node_id:<8} {interface_name:<20} No data (set to 0)")
                
        except Exception as e:
            # If error, use 0 as traffic value
            traffic[link] = 0
            print(f"✗ {link:<10} Error: {e} (set to 0)")
    
    return traffic

# ---------- 5. Legacy function for backward compatibility ----------
def fetch_generic_counters_legacy():
    """
    Legacy function that fetches data from the hardcoded URL
    Kept for backward compatibility
    """
    url = "http://192.168.10.22:8181/restconf/operational/network-topology:network-topology/topology/topology-netconf/node/node9/yang-ext:mount/Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/GigabitEthernet0%2F0%2F0%2F0/generic-counters"
    try:
        resp = requests.get(url, verify=False, auth=AUTH, headers=HEADERS)
        data = resp.json()['generic-counters']
        # Note: This returns the raw counters data with fields like 'bytes-received', 'bytes-sent'
        return data
    except Exception as e:
        print(f"Error fetching generic counters: {e}")
        return {}

# ---------- 6. 主流程 (updated with rate calculation) ----------
def collect_caller(use_rates=True, measurement_interval=10):
    """
    Main function to collect traffic data
    
    Args:
        use_rates (bool): If True, collect traffic rates; if False, use cumulative counters
        measurement_interval (int): Seconds between measurements for rate calculation
    
    Returns:
        dict: Traffic data (rates or cumulative based on use_rates parameter)
    """
    # Import links from main.py
    from main import LINKS
    
    if use_rates:
        print("🚀 Collecting real-time traffic RATES for all links...")
        traffic_data = collect_traffic_rates(LINKS, measurement_interval=measurement_interval)
        data_type = "rates"
        unit = "bytes_per_second"
    else:
        print("📊 Collecting cumulative traffic TOTALS for all links...")
        traffic_data = collect_link_traffic(LINKS)
        data_type = "cumulative"
        unit = "total_bytes"
    
    # Save to file with timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    output_data = {
        "timestamp": timestamp,
        "data_type": data_type,
        "unit": unit,
        "measurement_interval_seconds": measurement_interval if use_rates else None,
        "traffic": traffic_data
    }
    
    file_suffix = "rates" if use_rates else "cumulative"
    out_file = f"link_traffic_{file_suffix}_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Complete! Results saved to {out_file}")
    
    return traffic_data

def collect_caller_legacy():
    """Legacy function that collects cumulative traffic data"""
    return collect_caller(use_rates=False)

if __name__ == "__main__":
    collect_caller()
    