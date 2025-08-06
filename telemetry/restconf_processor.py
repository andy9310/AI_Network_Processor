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
    url = f"{BASE_URL}/data/network-topology:network-topology/topology/topology-netconf/node/{node_id}/yang-ext:mount/Cisco-IOS-XR-ifmgr-cfg:interface-configurations/interface-configuration/act/{safe_interface}"
    
    try:
        resp = requests.get(url, verify=False, auth=AUTH, headers=HEADERS, timeout=10)
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

def fetch_topology_info(links: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch topology information for all specified links
    
    Args:
        links: List of link names
    
    Returns:
        Dict mapping interface names to their configuration info
    """
    topology_info = {}
    
    print(f"🔍 Fetching topology information for {len(links)} links...")
    
    for link in links:
        if link not in INTERFACE_MAPPING:
            continue
            
        # Get interface and node information
        interface = INTERFACE_MAPPING[link]
        src_node = link.split('-')[0]
        node_id = NODE_MAPPING.get(src_node)
        
        if not node_id:
            continue
            
        # Fetch IPv4 configuration
        ip_address, netmask = get_interface_ipv4_address(node_id, interface)
        
        topology_info[interface] = {
            'node_id': node_id,
            'link': link,
            'ip_address': ip_address,
            'netmask': netmask
        }
        
        print(f"✓ {link:<10} {interface:<25} {ip_address}/{netmask}")
    
    return topology_info

def build_shutdown_commands(links: List[str], *, host="192.168.10.22", port=8181, auth="admin:admin") -> List[str]:
    """
    Build curl commands to shutdown interfaces for the given links
    
    Args:
        links: List of link names (e.g., ['S1-S2', 'S2-S1'])
        host: RESTCONF host address
        port: RESTCONF port
        auth: Authentication credentials
    
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
    topology_info = fetch_topology_info(links)
    
    for link in links:
        if link not in INTERFACE_MAPPING:
            continue
            
        interface = INTERFACE_MAPPING[link]
        
        # Get real IPv4 address and netmask from topology info
        interface_info = topology_info.get(interface, {})
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

def process_predicted_links(links_to_close: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict], Path, List[Path]]:
    """
    Main function to process predicted links into RESTCONF commands
    
    Args:
        links_to_close: List of links predicted to be closed by RL model
        **kwargs: Additional arguments for build_shutdown_commands
    
    Returns:
        Tuple of (commands, configs, commands_file_path, config_file_paths)
    """
    print(f"\n🔧 Processing {len(links_to_close)} predicted links into RESTCONF commands...")
    
    # Build commands and configurations
    commands = build_shutdown_commands(links_to_close, **kwargs)
    configs = build_config_files(links_to_close)
    
    # Write files
    commands_file, config_files = write_command_files(commands, configs)
    
    # Print summary
    print(f"\n📋 Generated Commands:")
    print(f"   • Total commands: {len(commands)}")
    print(f"   • Total config files: {len(config_files)}")
    print(f"   • Commands file: {commands_file}")
    print(f"   • Config files directory: {commands_file.parent}")
    
    # Print commands to console
    print(f"\n🖥️  RESTCONF Commands:")
    print("=" * 80)
    for i, cmd in enumerate(commands, 1):
        print(f"\n# Command {i} - Shutdown interface for link: {links_to_close[i-1] if i-1 < len(links_to_close) else 'N/A'}")
        print(cmd)
    print("=" * 80)
    
    return commands, configs, commands_file, config_files

def main():
    """Test function with sample predicted links"""
    # Sample predicted links for testing
    sample_links = ["S1-S2", "S4-S5", "S9-S7", "S10-S12"]
    
    print("🧪 Testing RESTCONF processor with sample links...")
    process_predicted_links(sample_links)

if __name__ == "__main__":
    main()
