#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Interface Mapping Generator
Generates INTERFACE_MAPPING dynamically from topology data instead of hardcoding
"""

import ipaddress
from typing import Dict, Set, Tuple, List
import re

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

def print_mapping_comparison(dynamic_mapping: Dict[str, str], static_mapping: Dict[str, str]):
    """
    Print comparison between dynamic and static mappings
    """
    print("\n🔍 Mapping Comparison:")
    print("=" * 80)
    
    all_keys = set(dynamic_mapping.keys()) | set(static_mapping.keys())
    
    matches = 0
    mismatches = 0
    
    for key in sorted(all_keys):
        dynamic_val = dynamic_mapping.get(key, "MISSING")
        static_val = static_mapping.get(key, "MISSING")
        
        if dynamic_val == static_val:
            print(f"✅ {key:12} -> {dynamic_val}")
            matches += 1
        else:
            print(f"❌ {key:12} -> Dynamic: {dynamic_val:25} | Static: {static_val}")
            mismatches += 1
    
    print("=" * 80)
    print(f"📊 Summary: {matches} matches, {mismatches} mismatches")
    
    return matches, mismatches

def update_collect_py_with_dynamic_mapping():
    """
    Update collect.py to use dynamic mapping instead of hardcoded mapping
    """
    print("\n🔄 To use dynamic mapping, replace the hardcoded INTERFACE_MAPPING in collect.py")
    print("   with a call to generate_interface_mapping() using your topology data.")

if __name__ == "__main__":
    # Example usage with your topology data
    sample_topology = {
        "node4:GigabitEthernet0/0/0/0": {"node_id": "node4", "ip_address": "10.0.4.1", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
        "node1:GigabitEthernet0/0/0/2": {"node_id": "node1", "ip_address": "10.0.4.2", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
        # Add more sample data as needed
    }
    
    mapping = generate_interface_mapping(sample_topology)
    print("Generated Interface Mapping:", mapping)
