#!/usr/bin/env python3
"""
Test script for dynamic interface mapping generation
"""

from dynamic_mapping import generate_interface_mapping, print_mapping_comparison
from collect import INTERFACE_MAPPING

# Your actual topology data
topology_data = {
    "node4:GigabitEthernet0/0/0/15": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/0": {"node_id": "node4", "ip_address": "10.0.4.1", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/14": {"node_id": "node4", "ip_address": "10.0.4.41", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/1": {"node_id": "node4", "ip_address": "10.0.4.5", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/13": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/2": {"node_id": "node4", "ip_address": "10.0.4.9", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/12": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/3": {"node_id": "node4", "ip_address": "192.168.5.2", "netmask": "255.255.255.0", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/11": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/4": {"node_id": "node4", "ip_address": "10.0.4.13", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/10": {"node_id": "node4", "ip_address": "10.0.4.37", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/5": {"node_id": "node4", "ip_address": "10.0.4.17", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/6": {"node_id": "node4", "ip_address": "10.0.4.21", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/18": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/17": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/16": {"node_id": "node4", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node4:GigabitEthernet0/0/0/7": {"node_id": "node4", "ip_address": "10.0.4.25", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/8": {"node_id": "node4", "ip_address": "10.0.9.14", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node4:GigabitEthernet0/0/0/9": {"node_id": "node4", "ip_address": "10.0.4.33", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node3:GigabitEthernet0/0/0/0": {"node_id": "node3", "ip_address": "10.0.9.10", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node3:GigabitEthernet0/0/0/1": {"node_id": "node3", "ip_address": "10.0.1.5", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node3:GigabitEthernet0/0/0/2": {"node_id": "node3", "ip_address": "10.0.4.10", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node3:GigabitEthernet0/0/0/3": {"node_id": "node3", "ip_address": "10.0.1.25", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/15": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/0": {"node_id": "node9", "ip_address": "10.0.9.1", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/14": {"node_id": "node9", "ip_address": "10.0.9.33", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/1": {"node_id": "node9", "ip_address": "10.0.9.5", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/13": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/2": {"node_id": "node9", "ip_address": "10.0.9.9", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/12": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/3": {"node_id": "node9", "ip_address": "10.0.9.13", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/11": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/4": {"node_id": "node9", "ip_address": "10.0.9.17", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/10": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/5": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/6": {"node_id": "node9", "ip_address": "10.0.9.21", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/18": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/17": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/16": {"node_id": "node9", "ip_address": "0.0.0.0", "netmask": "255.255.255.252", "active": "act", "shutdown": True},
    "node9:GigabitEthernet0/0/0/7": {"node_id": "node9", "ip_address": "10.0.9.25", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/8": {"node_id": "node9", "ip_address": "192.168.2.2", "netmask": "255.255.255.0", "active": "act", "shutdown": False},
    "node9:GigabitEthernet0/0/0/9": {"node_id": "node9", "ip_address": "10.0.9.29", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node1:GigabitEthernet0/0/0/0": {"node_id": "node1", "ip_address": "10.0.1.2", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node1:GigabitEthernet0/0/0/1": {"node_id": "node1", "ip_address": "10.0.1.6", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node1:GigabitEthernet0/0/0/2": {"node_id": "node1", "ip_address": "10.0.4.2", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node1:GigabitEthernet0/0/0/3": {"node_id": "node1", "ip_address": "10.0.9.2", "netmask": "255.255.255.252", "active": "act", "shutdown": False},
    "node1:GigabitEthernet0/0/0/4": {"node_id": "node1", "ip_address": "192.168.6.2", "netmask": "255.255.255.252", "active": "act", "shutdown": False}
}

def main():
    print("🧪 Testing Dynamic Interface Mapping Generation")
    print("=" * 60)
    
    # Generate dynamic mapping
    dynamic_mapping = generate_interface_mapping(topology_data)
    
    print(f"\n📊 Generated {len(dynamic_mapping)} dynamic mappings:")
    for link, interface in sorted(dynamic_mapping.items()):
        print(f"  {link:12} -> {interface}")
    
    # Compare with static mapping
    print_mapping_comparison(dynamic_mapping, INTERFACE_MAPPING)
    
    print(f"\n✅ Dynamic mapping successfully generated!")
    print(f"   • Total dynamic mappings: {len(dynamic_mapping)}")
    print(f"   • Total static mappings: {len(INTERFACE_MAPPING)}")

if __name__ == "__main__":
    main()
