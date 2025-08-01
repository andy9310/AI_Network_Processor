# Telemetry Collection System with RL Model and RAG Integration

This system collects traffic data from EVE-NG SDN nodes using RESTCONF API calls and integrates with:
- A trained PPO (Proximal Policy Optimization) reinforcement learning model to intelligently predict which network links should be closed
- RAG (Retrieval-Augmented Generation) system to enhance LLM responses with relevant documentation
- Mock RL model for testing while the real model is training

## Overview

The system has been updated to support:
- **Multiple nodes**: S1, S2, S3, ..., S17 → node1, node2, node3, ..., node17
- **Multiple interfaces**: Different GigabitEthernet interfaces for each link
- **Link-based collection**: Collects data based on the links defined in `main.py`
- **RL Model Integration**: Uses trained PPO model to predict optimal link closures
- **Mock RL Model**: Provides realistic link closure predictions for testing
- **RAG System**: Retrieval-Augmented Generation to enhance LLM responses with relevant documentation
- **LLM Integration**: Generates RESTCONF commands for network configuration
- **Free Embeddings**: Uses local sentence-transformers models (no API costs)

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

### 3. RL Model Architecture

#### Neural Network Architecture
- **NetworkFeatureExtractor**: Network feature extractor
  - Link Encoder: Processes features for each link
  - Attention Mechanism: Attention mechanism
  - Global Network: Global network features
- **EnhancedNetworkPolicy**: Enhanced network policy
  - Based on ActorCriticPolicy
  - Uses custom feature extractor

#### Input Features (7-dimensional)
Each link's state vector contains:
1. **Buffer Utilization**: Estimated based on output-drops and output-queue-drops
2. **Link Utilization**: Calculated based on traffic and maximum capacity
3. **Link Status**: Link status (up/down)
4. **Buffer Change Rate**: Buffer utilization change trend
5. **Utilization Change Rate**: Link utilization change trend
6. **Time Since Last Change**: Time since last state change
7. **Normalized Node Degree**: Normalized node degree

#### Output Actions (3-dimensional)
Model outputs 3 thresholds:
- **bufLow**: Buffer low threshold (0.00 - 0.50)
- **utilHi**: Utilization high threshold (0.30 - 0.80)
- **utilCap**: Utilization capacity threshold (0.60 - 1.00)

### 4. Mock RL Model (Testing)
For testing while the real model is training:
- **MockRLModel**: Simulates RL model behavior
- **Predefined Strategies**: 5 different link closure strategies
- **Random Selection**: Randomly selects from strategies for variety
- **Realistic Output**: Generates proper link names like ["S1-S2", "S1-S3", ...]

### 5. RAG System Architecture

#### Document Processing
- **DocumentChunker**: Splits documents into overlapping chunks
- **EmbeddingManager**: Generates embeddings using free local models (sentence-transformers)
- **VectorStore**: In-memory vector store with cosine similarity search

#### RAG Features
- **Document Loading**: Supports .docx and .txt files
- **Semantic Search**: Finds relevant documents based on query similarity
- **Prompt Enhancement**: Enhances LLM prompts with retrieved context
- **Caching**: Saves processed documents and embeddings for faster loading
- **Free Embeddings**: Uses local sentence-transformers models (no API costs)

## Main Functions

### Data Collection Functions

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

**Returns:** Dictionary with link names as keys and traffic values as integers

### RL Model Functions

#### `predict_links_to_close_rl(telemetry_data)`
Uses the trained RL model to predict which links should be closed.

**Parameters:**
- `telemetry_data`: Real-time traffic data dictionary

**Returns:** List of link names that should be closed

#### `llm_inference(user_prompt: str = None, use_rag: bool = True)`
Generates RESTCONF commands using LLM based on RL model predictions with optional RAG enhancement.

**Parameters:**
- `user_prompt`: Custom user prompt (optional)
- `use_rag`: Whether to use RAG enhancement (default: True)

**Returns:** RESTCONF curl commands and configuration JSON

### RAG System Functions

#### `rag_system.load_documents(file_path: str, force_reload: bool = False)`
Loads and processes documents for RAG system.

**Parameters:**
- `file_path`: Path to document file (.docx or .txt)
- `force_reload`: Force reload even if cache exists

#### `rag_system.enhance_prompt(user_prompt: str, system_prompt: str = "", top_k: int = 3)`
Enhances user prompt with relevant documents.

**Parameters:**
- `user_prompt`: Original user prompt
- `system_prompt`: System prompt for context
- `top_k`: Number of relevant documents to include

**Returns:** Enhanced prompt with relevant context

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

### 3. Use RL model for link prediction
```python
from rl_model import get_rl_manager

# Get RL model manager (uses mock model for testing)
rl_manager = get_rl_manager(use_mock=True)

# Predict links to close
telemetry_data = {
    "S1-S2": {
        "traffic": 500,
        "output-drops": 10,
        "output-queue-drops": 5,
        "max-capacity": 1000
    },
    # ... more link data
}

links_to_close = rl_manager.predict_links_to_close(telemetry_data)
print(f"Links to close: {links_to_close}")
```

### 4. Use RAG system for enhanced prompts
```python
from rag_system import get_rag_system

# Get RAG system with free local embeddings
rag_system = get_rag_system("all-MiniLM-L6-v2")

# Load documents
rag_system.load_documents("Guide.docx")

# Enhance prompt with relevant documents
enhanced_prompt = rag_system.enhance_prompt(
    "如何配置網路接口？",
    "你是網路管理助手",
    top_k=3
)
print(f"Enhanced prompt: {enhanced_prompt}")
```

### 5. Test the mock RL model
```bash
cd telemetry
python test_mock_rl.py
```

### 6. Test the RAG system
```bash
cd telemetry
python test_free_rag.py
```

## API Endpoints

### Data Collection
- `GET /telemetry`: Get current traffic data
- `GET /input`: Get input data

### Model Predictions
- `GET /predict-links-rl`: Use RL model to predict which links to close
- `GET /rl-model-info`: Get RL model information

### RAG System
- `POST /load-document`: Load documents into RAG system
- `GET /rag-info`: Get RAG system information
- `GET /search-documents`: Search for relevant documents

### LLM Integration
- `GET /output`: Get LLM-generated RESTCONF commands (with optional RAG)
- `POST /input`: Submit custom prompts with RAG enhancement

## Data Structure

### Telemetry Data Format
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

### RL Model Input Format
```python
{
    "link_name": {
        "traffic": int,           # Current traffic
        "output-drops": int,      # Output drops
        "output-queue-drops": int, # Queue drops
        "max-capacity": int       # Maximum capacity
    }
}
```

## Model Decision Logic

### Link Closure Conditions
The model decides to close links based on:

1. **Buffer and Utilization are both low**:
   ```python
   if buffer_utilization < bufLow and link_utilization < utilHi:
       close_link()
   ```

2. **Network Connectivity Guarantee**:
   - Ensures each node has at least one connected link
   - Prevents network partitioning

3. **Hysteresis Mechanism**:
   - Avoids frequent link state switching
   - Minimum state duration

### Mock Model Strategies
The mock model uses 5 predefined strategies:
1. **Low Traffic Links**: Close links with low traffic
2. **Edge Links**: Close peripheral links
3. **Redundant Links**: Close redundant connections
4. **High Delay Links**: Close links with high delay
5. **Random Selection**: Random link selection

## Configuration

### Base URL
The system connects to EVE-NG at: `http://192.168.10.22:8181/restconf`

### Authentication
- Username: `admin`
- Password: `admin`

### Headers
- Accept: `application/json`

### Model Files
- **Default model**: `enhanced_energy_rl_latest.zip`
- **Checkpoints**: `enhanced_model_checkpoints/`
- **Mock model**: Built-in for testing

### RAG Files
- **Document cache**: `vector_store_cache.pkl`
- **Supported formats**: `.docx`, `.txt`
- **Default document**: `Guide.docx`
- **Embedding model**: `all-MiniLM-L6-v2` (free, local)

## Training Environment Reference

The RL model is based on NS3 training environment with:

- **EnergyRoutingEnv**: Energy routing environment
- **PPO Algorithm**: Proximal Policy Optimization
- **Reward Function**: throughput - 0.5 × energy_usage
- **State Space**: (n_links, 7) observation space
- **Action Space**: 3-dimensional continuous action space

## Error Handling

The system includes comprehensive error handling:
- 404 errors for non-existent interfaces
- Network timeout handling
- JSON parsing error handling
- Missing node/interface mapping warnings
- Model loading fallback mechanisms
- Mock model fallback for testing

## Performance Optimization

1. **GPU Acceleration**: Model automatically detects and uses available GPU
2. **Batch Processing**: Supports batch prediction for efficiency
3. **Memory Optimization**: Uses historical data management to reduce memory usage
4. **Free Embeddings**: Local sentence-transformers models (no API costs)

## Files

### Core Files
- `collect.py`: Main collection logic
- `main.py`: FastAPI application with LLM, RL, and RAG integration
- `rl_model.py`: RL model loading and inference module
- `rag_system.py`: RAG system for document retrieval and prompt enhancement
- `rules.txt`: System rules for LLM

### Test Files
- `test_mock_rl.py`: Test script for mock RL model functionality
- `test_free_rag.py`: Test script for RAG system with free embeddings
- `test_collection.py`: Test script to verify collection functionality

### Configuration Files
- `requirements_rag.txt`: RAG system dependencies
- `Guide.docx`: Default document for RAG system
- `README.md`: This documentation

### Legacy Files (for reference)
- `mlp_model.py`: Previous MLP model implementation

## Running the System

1. **Test the collection:**
   ```bash
   python test_collection.py
   ```

2. **Test the mock RL model:**
   ```bash
   python test_mock_rl.py
   ```

3. **Test the RAG system:**
   ```bash
   python test_free_rag.py
   ```

4. **Run the main application:**
   ```bash
   python main.py
   ```

5. **Access the API endpoints:**
   - `GET /telemetry`: Get current traffic data
   - `GET /predict-links-rl`: Get RL model predictions
   - `GET /rag-info`: Get RAG system information
   - `GET /output`: Get LLM-generated RESTCONF commands (with RAG)
   - `POST /input`: Submit custom prompts with RAG enhancement

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Solution: Model will use mock model for testing

2. **Missing dependencies**
   ```bash
   pip install stable-baselines3 torch networkx
   pip install -r requirements_rag.txt
   ```

3. **CUDA version mismatch**
   - Check PyTorch and CUDA version compatibility
   - Use CPU version as fallback

4. **Document loading failed**
   - Check if Guide.docx exists in the directory
   - Use `/load-document` endpoint to load custom documents

5. **Embedding model download issues**
   - First run will download the model (~90MB)
   - Ensure stable internet connection for initial download

6. **Mock model not working**
   - Check if `use_mock=True` is set in `get_rl_manager()`
   - Verify predefined strategies are available

### Debug Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Training Script

### Model Architecture Correspondence
- `NetworkFeatureExtractor` ↔ Training `NetworkFeatureExtractor`
- `EnhancedNetworkPolicy` ↔ Training `EnhancedNetworkPolicy`
- State preprocessing ↔ Training `_state()` method

### Decision Logic Correspondence
- Threshold mapping ↔ Training `step()` method threshold calculation
- Link selection logic ↔ Training link closure conditions
- Network connectivity check ↔ Training degree check

### Data Format Correspondence
- Telemetry data ↔ NS3 environment observation data
- Feature calculation ↔ Training state calculation
- Historical data management ↔ Training history records

## Testing Strategy

### Mock RL Model Testing
- **Strategy Variety**: Tests all 5 predefined strategies
- **Real Data Integration**: Works with real telemetry data
- **Model Switching**: Can switch between mock and real models
- **Link Validation**: Ensures proper link name format

### RAG System Testing
- **Document Loading**: Tests .docx and .txt file loading
- **Embedding Generation**: Verifies free local embeddings work
- **Search Functionality**: Tests semantic search capabilities
- **Prompt Enhancement**: Verifies context enhancement

### System Integration Testing
- **End-to-End Flow**: From telemetry collection to LLM output
- **API Endpoints**: Tests all FastAPI endpoints
- **Error Handling**: Verifies graceful error handling
- **Performance**: Tests system performance under load 