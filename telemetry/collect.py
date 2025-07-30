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

# ---------- 1. 取得 Netconf topology ----------
def fetch_topology():
    url = f"{BASE_URL}/operational/network-topology:network-topology"
    r = requests.get(url, auth=AUTH, headers=HEADERS, timeout=10)
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
    r = requests.get(url, auth=AUTH, headers=HEADERS, timeout=10)
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
        
        resp = requests.get(url, verify=False, auth=AUTH, headers=HEADERS, timeout=10)
        if resp.status_code == 404:
            print(f"Interface {interface_name} not found on node {node_id}")
            return {}
        
        resp.raise_for_status()
        data = resp.json()
        return data.get('generic-counters', {})
    except Exception as e:
        print(f"Error fetching generic counters for node {node_id}, interface {interface_name}: {e}")
        return {}

# ---------- 4. Collect traffic data for all links ----------
def collect_link_traffic(links):
    """
    Collect traffic data for all specified links
    
    Args:
        links (list): List of link strings (e.g., ['S1-S2', 'S2-S1'])
    
    Returns:
        dict: Traffic data for all links in the same format as generate_random_traffic
    """
    traffic = {}
    
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
            interface_name = INTERFACE_MAPPING.get(link)
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

# ---------- 6. 主流程 (updated) ----------
def collect_caller():
    # Import links from main.py
    from main import LINKS
    
    print("Collecting traffic data for all links...")
    traffic_data = collect_link_traffic(LINKS)
    
    # Save to file with timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    output_data = {
        "timestamp": timestamp,
        "traffic": traffic_data
    }
    
    out_file = f"link_traffic_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Complete! Results saved to {out_file}")
    
    return traffic_data

if __name__ == "__main__":
    collect_caller()
    