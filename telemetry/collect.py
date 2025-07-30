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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)
## device info 192.168.10.22(eveng's linux ip)
url = "http://192.168.10.22:8181/restconf/operational/network-topology:network-topology/topology/topology-netconf/node/node9/yang-ext:mount/Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/GigabitEthernet0%2F0%2F0%2F0/generic-counters"
AUTH     = HTTPBasicAuth("admin", "admin")
HEADERS  = {"Accept": "application/json"}
# need to forwarding the localhost:8181
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

# ---------- 3. 取得 generic-counters ----------
def fetch_generic_counters():
    try:
        resp = requests.get(url, verify=False, auth=AUTH, headers=HEADERS)
        data = resp.json()['generic-counters']
        return data
    except Exception as e:
        print(f"Error fetching generic counters: {e}")
        return {}

# ---------- 4. 主流程 ----------
def collect_caller():
    topo = fetch_topology()
    nodes = list_xr_nodes(topo)
    if not nodes:
        print("找不到任何 XR Netconf 節點，請檢查拓撲/連線。")
        sys.exit(1)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    all_stats = {"timestamp": timestamp, "nodes": []}

    for node in nodes:
        print(f"Node {node}")
        node_entry = {"node-id": node, "interfaces": []}
        for intf in list_interfaces(node):
            try:
                counters = fetch_generic_counters(node, intf)
                node_entry["interfaces"].append(counters)
                rx = counters["generic-counters"]["in-octets"]
                tx = counters["generic-counters"]["out-octets"]
                print(f"  • {intf:<20} RX={rx:<12} TX={tx}")
            except Exception as e:
                print(f"  • {intf:<20} 取值失敗：{e}")
        all_stats["nodes"].append(node_entry)

    # 將完整 JSON 寫檔
    out_file = f"xr_counters_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n 完成，結果寫入 {out_file}")

    