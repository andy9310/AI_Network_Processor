#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESTCONF Command Processor
Processes predicted links from RL model into RESTCONF commands for network configuration
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import urllib.parse

# Import the interface mapping from collect.py
from collect import INTERFACE_MAPPING, NODE_MAPPING, BASE_URL, AUTH, HEADERS
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# RESTCONF URL format for shutting down interfaces
REST_ROOT_FMT = ("http://{host}:{port}/restconf/data/"
                 "network-topology:network-topology/topology/topology-netconf/"
                 "node/{node}/yang-ext:mount/Cisco-IOS-XR-ifmgr-cfg/"
                 "interface-configurations/interface-configuration/{interface_name}")

def _iface_to_url(interface: str, *, host: str, port: int, node: str) -> str:
    """Convert interface name to RESTCONF URL with proper URL encoding"""
    safe_interface = urllib.parse.quote(interface, safe='')
    return REST_ROOT_FMT.format(host=host, port=port, node=node, interface_name=safe_interface)

def fetch_interface_config(node_id: str, interface_name: str) -> Dict:
    """Fetch interface configuration including IPv4 address from RESTCONF API
    
    Args:
        node_id: Node identifier (e.g., 'node1')
        interface_name: Interface name (e.g., 'GigabitEthernet0/0/0/0')
    
    Returns:
        Dict containing interface configuration data
    """
    # URL encode the interface name
    safe_interface = urllib.parse.quote(interface_name, safe='')
    
    # RESTCONF URL for interface configuration
    url = f"{BASE_URL}/operational/network-topology:network-topology/topology/topology-netconf/node/{node_id}/yang-ext:mount/Cisco-IOS-XR-ifmgr-cfg:interface-configurations/interface-configuration/{safe_interface}"
    
    try:
        resp = requests.get(url, verify=False, auth=AUTH, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠️  Error fetching config for {node_id}/{interface_name}: {e}")
        return {}

def get_interface_ipv4_address(node_id: str, interface_name: str) -> Tuple[str, str]:
    """Get IPv4 address and netmask for an interface
    
    Args:
        node_id: Node identifier
        interface_name: Interface name
    
    Returns:
        Tuple of (ip_address, netmask)
    """
    config = fetch_interface_config(node_id, interface_name)
    
    try:
        # Navigate through the JSON structure to find IPv4 configuration
        interface_config = config.get('interface-configuration', [{}])[0]
        ipv4_network = interface_config.get('Cisco-IOS-XR-ifmgr-cfg:ipv4-network', {})
        addresses = ipv4_network.get('addresses', {})
        primary = addresses.get('primary', {})
        
        ip_address = primary.get('address', '0.0.0.0')
        netmask = primary.get('netmask', '255.255.255.252')
        
        return ip_address, netmask
    except Exception as e:
        print(f"⚠️  Error parsing IPv4 config for {node_id}/{interface_name}: {e}")
        return '0.0.0.0', '255.255.255.252'

def fetch_all_nodes() -> List[str]:
    """Fetch all nodes from the topology endpoint
    
    Returns:
        List of node IDs available in the topology
    """
    topology_url = f"{BASE_URL}/operational/network-topology:network-topology/topology/topology-netconf"
    
    try:
        print(f"🔍 Fetching topology from: {topology_url}")
        resp = requests.get(topology_url, verify=False, auth=AUTH, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        topology_data = resp.json()
        # Extract node IDs from topology
        nodes = []
        topology = topology_data.get('topology', [{}])[0]
        node_list = topology.get('node', [])
        
        for node in node_list:
            node_id = node.get('node-id', '')
            if node_id and 'node' in node_id:  # Filter for actual network nodes
                nodes.append(node_id)
                print(f"✓ Found node: {node_id}")
        
        print(f"📊 Total nodes found: {len(nodes)}")
        return nodes
        
    except Exception as e:
        print(f"⚠️  Error fetching topology: {e}")
        return []

def fetch_node_interfaces(node_id: str) -> Dict[str, Dict]:
    """Fetch all interface configurations for a specific node
    
    Args:
        node_id: Node identifier (e.g., 'node1')
    
    Returns:
        Dict mapping interface names to their configuration data
    """
    interfaces_url = (f"{BASE_URL}/operational/network-topology:network-topology/topology/topology-netconf/"
                     f"node/{node_id}/yang-ext:mount/Cisco-IOS-XR-ifmgr-cfg:interface-configurations")
    
    try:
        print(f"🔍 Fetching interfaces for {node_id}...")
        resp = requests.get(interfaces_url, verify=False, auth=AUTH, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        interface_data = resp.json()
        print(interface_data)
        interfaces = {}
        interface_configs = interface_data.get('interface-configurations', {}).get('interface-configuration', [])
        
        for config in interface_configs:
            interface_name = config.get('interface-name', '')
            if interface_name and 'GigabitEthernet' in interface_name:
                # Extract IPv4 configuration if available
                ipv4_config = config.get('Cisco-IOS-XR-ipv4-io-cfg:ipv4-network', {})
                addresses = ipv4_config.get('addresses', {})
                primary = addresses.get('primary', {})
                
                ip_address = primary.get('address', '0.0.0.0')
                netmask = primary.get('netmask', '255.255.255.252')
                
                interfaces[interface_name] = {
                    'node_id': node_id,
                    'ip_address': ip_address,
                    'netmask': netmask,
                    'active': config.get('active', 'act'),
                    'shutdown': 'shutdown' in config
                }
                
                print(f"  ✓ {interface_name:<25} {ip_address}/{netmask}")
        
        print(f"📊 Found {len(interfaces)} interfaces for {node_id}")
        return interfaces
        
    except Exception as e:
        print(f"⚠️  Error fetching interfaces for {node_id}: {e}")
        return {}

def fetch_topology_info() -> Dict[str, Dict[str, str]]:
    """Fetch topology information using the two-step approach:
    1. Get all nodes from topology endpoint
    2. Get all interface configurations for each node
    Returns:
        Dict mapping interface names to their configuration info
    """
    print(f"🚀 Starting topology discovery...")
    
    # Step 1: Fetch all nodes from topology
    all_nodes = fetch_all_nodes()
    if not all_nodes:
        print(" No nodes found in topology")
        return {}
    
    # Step 2: Fetch interface configurations for each node
    topology_info = {}
    
    for node_id in all_nodes:
        node_interfaces = fetch_node_interfaces(node_id)
        
        # Add node interfaces to topology info
        for interface_name, interface_config in node_interfaces.items():
            # Create a unique key for the interface
            interface_key = f"{node_id}:{interface_name}"
            topology_info[interface_key] = interface_config
    
    print(f"📊 Total interfaces discovered: {len(topology_info)}")
    
    # Return in the expected format for compatibility with collect.py
    return {
        'topology_info': topology_info
    }

def build_shutdown_commands(links: List[str], *, host="192.168.10.22", port=8181, auth="admin:admin", for_file=True) -> List[str]:
    """
    Build curl commands to shutdown interfaces for the given links
    
    Args:
        links: List of link names (e.g., ['S1-S2', 'S2-S1'])
        host: RESTCONF host address
        port: RESTCONF port
        auth: Authentication credentials
        for_file: If True, format for shell file execution (with escapes). If False, clean format for API response.
    
    Returns:
        List of curl command strings
    """
    commands = []
    
    for link in links:
        if link not in INTERFACE_MAPPING:
            print(f"⚠️  Warning: Link {link} not found in interface mapping, skipping...")
            continue
            
        # Get interface name and source node
        interface = INTERFACE_MAPPING[link]
        src_node = link.split('-')[0]  # Extract source node (e.g., 'S1' from 'S1-S2')
        
        # Get corresponding node ID
        node_id = NODE_MAPPING.get(src_node)
        if not node_id:
            print(f"⚠️  Warning: Node {src_node} not found in node mapping, skipping...")
            continue
        
        # Build RESTCONF URL
        url = _iface_to_url(interface, host=host, port=port, node=node_id)
        
        # Build curl command for shutting down the interface
        if for_file:
            # Format with escapes for shell file execution
            cmd = (f"curl -u {auth} -X PUT "
                   f"-H 'Content-Type: application/yang-data+json' \\\n"
                   f"     '{url}' \\\n"
                   f"     -d '{{\n"
                   f"       \"Cisco-IOS-XR-ifmgr-cfg:interface-configuration\": {{\n"
                   f"         \"active\": \"act\",\n"
                   f"         \"interface-name\": \"{interface}\",\n"
                   f"         \"shutdown\": [null]\n"
                   f"       }}\n"
                   f"     }}'")
        else:
            # Clean format for API response - single line with spaces
            json_data = '{"Cisco-IOS-XR-ifmgr-cfg:interface-configuration": {"active": "act", "interface-name": "' + interface + '", "shutdown": [null]}}'
            cmd = f"curl -u {auth} -X PUT -H 'Content-Type: application/yang-data+json' '{url}' -d '{json_data}'"
        
        commands.append(cmd)
    
    return commands

def build_config_files(links: List[str]) -> Dict[str, Dict]:
    """
    Build JSON configuration files for each interface to be shut down
    Uses real IPv4 addresses fetched from the topology
    
    Args:
        links: List of link names
    
    Returns:
        Dictionary mapping interface names to their JSON configurations
    """
    configs = {}
    
    # Fetch real topology information including IPv4 addresses
    topology_data = fetch_topology_info()
    
    for link in links:
        if link not in INTERFACE_MAPPING:
            continue
            
        interface = INTERFACE_MAPPING[link]
        
        # Find the interface in topology data
        # topology_data format: {"node1:GigabitEthernet0/0/0/0": {"node_id": "node1", "ip_address": "10.0.1.2", ...}}
        interface_info = {}
        for key, data in topology_data.items():
            if key.endswith(f":{interface}"):
                interface_info = data
                break
        
        # Get real IPv4 address and netmask from topology info
        ip_address = interface_info.get('ip_address', '0.0.0.0')
        netmask = interface_info.get('netmask', '255.255.255.252')
        
        configs[interface] = {
            "interface-configuration": [{
                "active": "act",
                "interface-name": interface,
                "shutdown": [None],
                "Cisco-IOS-XR-ifmgr-cfg:ipv4-network":{
                    "addresses": {
                        "primary": {
                            "address": ip_address,
                            "netmask": netmask
                        }
                    }
                }
            }]
        }
        
        print(f"📝 Config for {interface}: {ip_address}/{netmask}")
    
    return configs

def write_command_files(commands: List[str], configs: Dict[str, Dict], 
                       output_dir: Path = Path("restconf_output")) -> Tuple[Path, List[Path]]:
    """
    Write commands to a text file and config files to JSON files
    
    Args:
        commands: List of curl commands
        configs: Dictionary of interface configurations
        output_dir: Output directory path
    
    Returns:
        Tuple of (commands_file_path, list_of_config_file_paths)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write commands to text file
    commands_file = output_dir / f"restconf_commands_{timestamp}.txt"
    with commands_file.open("w", encoding="utf-8") as f:
        f.write(f"# RESTCONF Commands to Shutdown Interfaces\n")
        f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total commands: {len(commands)}\n\n")
        
        for i, cmd in enumerate(commands, 1):
            f.write(f"# Command {i}\n")
            f.write(f"{cmd}\n\n")
    
    # Write individual config files
    config_files = []
    for interface, config in configs.items():
        config_filename = f"config_{interface.replace('/', '_')}_{timestamp}.json"
        config_file = output_dir / config_filename
        
        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        config_files.append(config_file)
    
    return commands_file, config_files

def process_predicted_links(links_to_close: List[str], **kwargs) -> Tuple[List[str], List[str], Dict[str, Dict], Path, List[Path]]:
    """
    Main function to process predicted links into RESTCONF commands
    
    Args:
        links_to_close: List of links predicted to be closed by RL model
        **kwargs: Additional arguments for build_shutdown_commands
    
    Returns:
        Tuple of (file_commands, api_commands, configs, commands_file_path, config_file_paths)
    """
    print(f"\n🔧 Processing {len(links_to_close)} predicted links into RESTCONF commands...")
    
    # Build commands in both formats
    file_commands = build_shutdown_commands(links_to_close, for_file=True, **kwargs)  # For file writing
    api_commands = build_shutdown_commands(links_to_close, for_file=False, **kwargs)  # For API response
    configs = build_config_files(links_to_close)
    
    # Write files using the file format commands
    commands_file, config_files = write_command_files(file_commands, configs)
    
    # Print summary
    print(f"\n📋 Generated Commands:")
    print(f"   • Total commands: {len(file_commands)}")
    print(f"   • Total config files: {len(config_files)}")
    print(f"   • Commands file: {commands_file}")
    print(f"   • Config files directory: {commands_file.parent}")
    
    # Print clean commands to console (for readability)
    print(f"\n🖥️  RESTCONF Commands (Clean Format):")
    print("=" * 80)
    for i, cmd in enumerate(api_commands, 1):
        print(f"\n# Command {i} - Shutdown interface for link: {links_to_close[i-1] if i-1 < len(links_to_close) else 'N/A'}")
        print(cmd)
    print("=" * 80)
    
    return file_commands, api_commands, configs, commands_file, config_files

def main():
    """Test function with sample predicted links"""
    # Sample predicted links for testing
    sample_links = ["S1-S2", "S4-S5", "S9-S7", "S10-S12"]
    
    print("🧪 Testing RESTCONF processor with sample links...")
    process_predicted_links(sample_links)

if __name__ == "__main__":
    main()
