#!/usr/bin/env python3
"""
Demo script showing dynamic interface mapping in action
"""

from collect import generate_interface_mapping, INTERFACE_MAPPING

# Your actual topology data from the working system
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
    print("🎯 Dynamic Interface Mapping Demo")
    print("=" * 50)
    
    # Generate dynamic mapping from your actual topology
    dynamic_mapping = generate_interface_mapping(topology_data)
    
    print(f"\n📊 Your Current Topology (4 nodes: 1, 3, 4, 9)")
    print(f"   Generated {len(dynamic_mapping)} dynamic interface mappings:")
    
    for link, interface in sorted(dynamic_mapping.items()):
        print(f"   {link:12} -> {interface}")
    
    print(f"\n🔄 Benefits of Dynamic Mapping:")
    print("   ✅ Automatically adapts to topology changes")
    print("   ✅ No need to manually update hardcoded mappings")
    print("   ✅ Discovers connections by analyzing IP subnets")
    print("   ✅ Graceful fallback to static mapping if needed")
    
    print(f"\n🚀 Usage in your code:")
    print("   • collect_link_traffic(links, use_dynamic_mapping=True)  # Default")
    print("   • collect_link_traffic(links, use_dynamic_mapping=False) # Static fallback")
    
    print(f"\n✨ The dynamic mapping is now integrated into collect.py!")
    print("   When you run collect_caller(), it will automatically use dynamic mapping.")

if __name__ == "__main__":
    main()
