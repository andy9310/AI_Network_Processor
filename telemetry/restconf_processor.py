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
from collect import INTERFACE_MAPPING, NODE_MAPPING

# RESTCONF URL format for shutting down interfaces
REST_ROOT_FMT = ("http://{host}:{port}/restconf/data/"
                 "network-topology:network-topology/topology/topology-netconf/"
                 "node/{node}/yang-ext:mount/Cisco-IOS-XR-ifmgr-cfg/"
                 "interface-configurations/interface-configuration/{interface_name}")

def _iface_to_url(interface: str, *, host: str, port: int, node: str) -> str:
    """Convert interface name to RESTCONF URL with proper URL encoding"""
    safe_interface = urllib.parse.quote(interface, safe='')
    return REST_ROOT_FMT.format(host=host, port=port, node=node, interface_name=safe_interface)

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
    
    Args:
        links: List of link names
    
    Returns:
        Dictionary mapping interface names to their JSON configurations
    """
    configs = {}
    
    for link in links:
        if link not in INTERFACE_MAPPING:
            continue
            
        interface = INTERFACE_MAPPING[link]
        
        configs[interface] = {
            "Cisco-IOS-XR-ifmgr-cfg:interface-configuration": {
                "active": "act",
                "interface-name": interface,
                "shutdown": [None]
            }
        }
    
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
