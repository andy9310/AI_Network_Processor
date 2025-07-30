# Telemetry Collection System

This system collects traffic data from EVE-NG SDN nodes using RESTCONF API calls to fetch generic counters from different interfaces.

## Overview

The system has been updated to support:
- **Multiple nodes**: S1, S2, S3, ..., S17 → node1, node2, node3, ..., node17
- **Multiple interfaces**: Different GigabitEthernet interfaces for each link
- **Link-based collection**: Collects data based on the links defined in `main.py`

## Key Components

### 1. Node Mapping
```python
NODE_MAPPING = {
    "S1": "node1",
    "S2": "node2", 
    "S3": "node3",
    # ... up to S17 -> node17
}
```

### 2. Interface Mapping
Each link (e.g., "S1-S2") is mapped to a specific interface on the source switch:
```python
INTERFACE_MAPPING = {
    "S1-S2": "GigabitEthernet0/0/0/0",
    "S1-S3": "GigabitEthernet0/0/0/1",
    "S1-S4": "GigabitEthernet0/0/0/2",
    # ... etc
}
```

### 3. Main Functions

#### `fetch_generic_counters(node_id, interface_name)`
Fetches generic counters for a specific node and interface.

**Parameters:**
- `node_id`: The node ID (e.g., 'node1', 'node2')
- `interface_name`: The interface name (e.g., 'GigabitEthernet0/0/0/0')

**Returns:** Dictionary containing counter data (in-octets, out-octets, in-pkts, out-pkts)

#### `collect_link_traffic(links)`
Collects traffic data for all specified links.

**Parameters:**
- `links`: List of link strings (e.g., ['S1-S2', 'S2-S1'])

**Returns:** Dictionary with link names as keys and traffic values as integers (same format as generate_random_traffic)

## Usage Examples

### 1. Collect data for a single link
```python
from collect import fetch_generic_counters

# Fetch data for S1-S2 link (node1, GigabitEthernet0/0/0/0)
counters = fetch_generic_counters("node1", "GigabitEthernet0/0/0/0")
print(f"RX: {counters.get('in-octets', 0)}, TX: {counters.get('out-octets', 0)}")
```

### 2. Collect data for multiple links
```python
from collect import collect_link_traffic

links = ["S1-S2", "S1-S3", "S4-S1"]
traffic_data = collect_link_traffic(links)

for link, traffic_value in traffic_data.items():
    print(f"{link}: Traffic = {traffic_value}")
```

### 3. Collect data for all links
```python
from collect import collect_link_traffic
from main import LINKS

# Collect data for all links defined in main.py
traffic_data = collect_link_traffic(LINKS)
```

### 4. Run the test script
```bash
cd telemetry
python test_collection.py
```

## Data Structure

The collected data has the same format as `generate_random_traffic`:
```json
{
  "S1-S2": 8888888,
  "S1-S3": 11111110,
  "S2-S1": 7654321,
  "S3-S1": 8765432,
  "S4-S1": 1234567
}
```

Where:
- **Keys**: Link names (e.g., "S1-S2", "S2-S1")
- **Values**: Total traffic in bytes (bytes_received + bytes_sent)

## Configuration

### Base URL
The system connects to EVE-NG at: `http://192.168.10.22:8181/restconf`

### Authentication
- Username: `admin`
- Password: `admin`

### Headers
- Accept: `application/json`

### URL Encoding
Interface names are automatically URL-encoded for REST API calls:
- `GigabitEthernet0/0/0/0` → `GigabitEthernet0%2F0%2F0%2F0`
- `GigabitEthernet0/0/0/1` → `GigabitEthernet0%2F0%2F0%2F1`
- This ensures proper URL construction for RESTCONF API calls

## Error Handling

The system includes comprehensive error handling:
- 404 errors for non-existent interfaces
- Network timeout handling
- JSON parsing error handling
- Missing node/interface mapping warnings

## Integration with main.py

The `main.py` file has been updated to use the new collection system:
- `collector()` function now calls `collect_link_traffic(LINKS)`
- Legacy function `fetch_generic_counters_legacy()` is available for backward compatibility

## Files

- `collect.py`: Main collection logic
- `main.py`: FastAPI application with LLM integration
- `test_collection.py`: Test script to verify functionality
- `README.md`: This documentation

## Running the System

1. **Test the collection:**
   ```bash
   python test_collection.py
   ```

2. **Run the main application:**
   ```bash
   python main.py
   ```

3. **Access the API endpoints:**
   - `GET /telemetry`: Get current traffic data
   - `GET /output`: Get LLM-generated recommendations 