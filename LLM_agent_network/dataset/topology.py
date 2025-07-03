# odl collect the topology data
# read outcome from the curl command 
'''
"http://{host}:{port}/restconf/operational/"
"network-topology:network-topology/topology/topology-netconf/"
"node/{node}/yang-ext:mount/Cisco-IOS-XR-ifmgr-cfg/"
"interfaces/interface/{iface}
'''
# output to topology.json
'''
"L1": {{"GigabitEthernet0/0/0/0","10.1.1.1"},{"GigabitEthernet0/0/0/3","10.1.2.1"}},
"<link number>": {{"< interface 1 >","< ip 1 >"}, {"< interface 2 >","< ip 2 >"} },
'''