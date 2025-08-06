from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import random, json, os, torch, textwrap
from unsloth import FastLanguageModel
from dotenv import load_dotenv
from collect import collect_link_traffic, fetch_generic_counters_legacy
from rl_model import get_rl_manager
from rag_system import get_rag_system
from restconf_processor import process_predicted_links
load_dotenv()
# ────────────────────── 載入LLM模型 ──────────────────────────
MAX_SEQ_LEN  = 2048
MODEL_NAME   = "taide/Llama-3.1-TAIDE-LX-8B-Chat"
print("Loading model… (只載一次)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,                # 🔧 4-bit 量化→ 8 GB VRAM 夠用
    token          = os.getenv("HUGGINGFACE_TOKEN", ""),
)           
print("Model on", model.device)

# ────────────────────── 載入RL模型 ───────────────────
# 初始化RL模型管理器 (使用真實訓練好的模型)
rl_manager = get_rl_manager(use_mock=True)
print(f"🤖 RL Model Info: {rl_manager.get_model_info()}")

# ────────────────────── 載入RAG系統 ───────────────────
# 初始化RAG系統 (使用免費的本地嵌入模型)
try:
    rag_system = get_rag_system("all-MiniLM-L6-v2")
    print("📚 RAG System initialized with free local embeddings")

    # 嘗試載入Guide.docx文檔
    try:
        rag_system.load_documents("Guide.docx")
        print("✅ Guide.docx loaded into RAG system")
    except Exception as e:
        print(f"⚠️ Failed to load Guide.docx: {e}")
        print("📝 You can load documents later using the /load-document endpoint")
except Exception as e:
    print(f"⚠️ RAG System initialization failed: {e}")
    print("📝 RAG features will be disabled")
    rag_system = None

# ────────────────────── ❸ System prompt ───────────────────
RULES_PATH = Path("rules.txt")
rules_text = RULES_PATH.read_text(encoding="utf-8")
SYSTEM_PROMPT = textwrap.dedent(f"""{rules_text.strip()}
你是 Cisco IOS XR 網管助手，完整輸出要關閉連結對應 interface 的 RESTCONF curl 指令與對應 config JSON""")

# ────────────────────── ❹ Telemetry 產生器 (略)… ───────────
# LINKS … generate_random_traffic() 與 collector() 完全照舊
# (此處省略，保持原來函式不變)
LINKS = [
    "S1-S2", "S1-S3", "S1-S4", "S1-S9",
    "S2-S1", "S2-S4", "S2-S9",
    "S3-S1", "S3-S4", "S3-S9",
    "S4-S1", "S4-S2", "S4-S3", "S4-S5", "S4-S6", "S4-S7",
    "S4-S8", "S4-S9", "S4-S10", "S4-S11", "S4-S15",
    "S5-S4", "S5-S9",
    "S6-S4", "S6-S15",
    "S7-S4", "S7-S9",
    "S8-S4", "S8-S9",
    "S9-S1", "S9-S2", "S9-S3", "S9-S4", "S9-S5",
    "S9-S7", "S9-S8", "S9-S10", "S9-S15",
    "S10-S4", "S10-S9", "S10-S12", "S10-S13",
    "S10-S14", "S10-S15", "S10-S16", "S10-S17",
    "S11-S4", "S11-S15",
    "S12-S10", "S12-S15",
    "S13-S10", "S13-S15",
    "S14-S10", "S14-S15",
    "S15-S4", "S15-S6", "S15-S9", "S15-S10", "S15-S11",
    "S15-S12", "S15-S13", "S15-S14", "S15-S16", "S15-S17",
    "S16-S10", "S16-S15",
    "S17-S10", "S17-S15",
]

def generate_random_traffic(links=LINKS, min_traffic=1, max_traffic=1_000):
    traffic = {}
    for link in links:
        src, dst = link.split("-")
        fwd_key, rev_key = f"{src}-{dst}", f"{dst}-{src}"
        if fwd_key not in traffic:
            traffic[fwd_key] = random.randint(min_traffic, max_traffic)
        if rev_key not in traffic:
            traffic[rev_key] = random.randint(min_traffic, max_traffic)
    return traffic
def default_collector():
    """default Telemetry 產生器。"""
    return generate_random_traffic()

def collector():
    """完整 Telemetry 產生器。"""
    return collect_link_traffic(LINKS)
# ────────────────────── ❺ RL模型預測函數 ───────────────────
def predict_links_to_close_rl(telemetry_data: dict) -> list:
    """
    使用RL模型預測應該關閉哪些link
    
    Args:
        telemetry_data: 即時流量數據
        
    Returns:
        list: 應該關閉的link列表
    """
    try:
        return rl_manager.predict_links_to_close(telemetry_data)
    except Exception as e:
        print(f"❌ Error in RL prediction: {e}")
        return ["L1", "L3", "L6"]  # 預設值

# ────────────────────── ❻ LLM 推論 ─────────────────────────
def llm_inference(user_prompt: str = None, use_rag: bool = True):
    """
    LLM inference with optional RAG enhancement
    
    Args:
        user_prompt: Custom user prompt (optional)
        use_rag: Whether to use RAG enhancement (default: True)
    """
    telemetry_data = collector() ## link utilization
    links_to_close = predict_links_to_close_rl(telemetry_data)
    print(f"🤖 RL predicted links to close: {links_to_close}")
    
    # Process predicted links into RESTCONF commands
    commands, configs, commands_file, config_files = process_predicted_links(links_to_close)
    print(f"📁 RESTCONF commands saved to: {commands_file}")
    ## 
    telemetry_json = json.dumps(telemetry_data, ensure_ascii=False)
    
    # Use custom prompt or default
    if user_prompt:
        prompt = user_prompt
    else:
        prompt = f"請根據上述資料，關閉 {', '.join(links_to_close)}"
    
    # Enhance prompt with RAG if enabled
    if use_rag and rag_system is not None:
        try:
            enhanced_prompt = rag_system.enhance_prompt(prompt, SYSTEM_PROMPT)
            print(f"📚 RAG enhanced prompt with relevant documents")
        except Exception as e:
            print(f"⚠️ RAG enhancement failed: {e}")
            enhanced_prompt = prompt
            print(f"📝 Using original prompt without RAG")
    else:
        enhanced_prompt = prompt
        print(f"📝 Using original prompt without RAG")

    full_prompt = (
        f"以下是即時流量資料（JSON）：\n{telemetry_json}\n\n"
        f"{enhanced_prompt}"
    )

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": full_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")                           # 🔧 張量也搬到 GPU

    with torch.no_grad():
        outputs = model.generate(
            input_ids       = input_ids,
            max_new_tokens  = 256,
            do_sample       = True,
            temperature     = 0.7,
            top_p           = 0.9,
        )

    generated = outputs[0, input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ────────────────────── ❼ FastAPI 端點 (原樣) ──────────────
app = FastAPI(
    title       = "XR Telemetry + LLM Demo",
    description = "產生Telemetry，並用 UnsLoTH Llama-3.1 taide 產 RESTCONF 指令",
    version     = "1.0.0",
)

@app.get("/telemetry")
async def get_telemetry():
    try:
        return default_collector()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/input")
async def get_input():
    try:
        return {"input": "input"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/input")
async def post_input(user_prompt: str = None, use_rag: bool = True):
    """Allow users to add custom prompts to the LLM with RAG enhancement"""
    try:
        result = llm_inference(user_prompt=user_prompt, use_rag=use_rag)
        
        telemetry_data = collector()
        links_to_close = predict_links_to_close_rl(telemetry_data)
        
        return {
            "user_prompt": user_prompt,
            "use_rag": use_rag,
            "telemetry_data": telemetry_data,
            "predicted_links_to_close": links_to_close,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/output")
async def get_output(use_rag: bool = True):
    try:
        result = llm_inference(use_rag=use_rag)
        return {"result": result, "use_rag": use_rag}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：使用RL模型預測要關閉的link
@app.get("/predict-links-rl")
async def predict_links_rl():
    """使用RL模型來預測應該關閉哪些link"""
    try:
        telemetry_data = collector()
        links_to_close = predict_links_to_close_rl(telemetry_data)
        
        # Process predicted links into RESTCONF commands
        commands, configs, commands_file, config_files = process_predicted_links(links_to_close)
        
        return {
            "telemetry_data": telemetry_data,
            "predicted_links_to_close": links_to_close,
            "restconf_commands": commands,
            "commands_file": str(commands_file),
            "config_files": [str(f) for f in config_files],
            "model_info": rl_manager.get_model_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：獲取RL模型信息
@app.get("/rl-model-info")
async def get_rl_model_info():
    """獲取RL模型信息"""
    try:
        return rl_manager.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：載入文檔到RAG系統
@app.post("/load-document")
async def load_document(file_path: str, force_reload: bool = False):
    """載入文檔到RAG系統"""
    try:
        rag_system.load_documents(file_path, force_reload=force_reload)
        return {
            "message": f"Document {file_path} loaded successfully",
            "document_info": rag_system.get_document_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：獲取RAG系統信息
@app.get("/rag-info")
async def get_rag_info():
    """獲取RAG系統信息"""
    try:
        return rag_system.get_document_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：搜索相關文檔
@app.get("/search-documents")
async def search_documents(query: str, top_k: int = 3):
    """搜索相關文檔"""
    try:
        results = rag_system.retrieve_relevant_docs(query, top_k=top_k)
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：下載RESTCONF命令文件
@app.get("/download-commands")
async def download_commands(filename: str = None):
    """下載生成的RESTCONF命令文件"""
    try:
        output_dir = Path("restconf_output")
        
        if filename:
            # Download specific file
            file_path = output_dir / filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File {filename} not found")
        else:
            # Download the most recent commands file
            command_files = list(output_dir.glob("restconf_commands_*.txt"))
            if not command_files:
                raise HTTPException(status_code=404, detail="No command files found")
            
            # Get the most recent file
            file_path = max(command_files, key=lambda f: f.stat().st_mtime)
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：列出所有可用的命令文件
@app.get("/list-command-files")
async def list_command_files():
    """列出所有可用的RESTCONF命令文件"""
    try:
        output_dir = Path("restconf_output")
        
        if not output_dir.exists():
            return {"files": [], "message": "No output directory found"}
        
        # Get all command files
        command_files = list(output_dir.glob("restconf_commands_*.txt"))
        config_files = list(output_dir.glob("config_*.json"))
        
        file_info = []
        
        # Add command files info
        for file_path in command_files:
            stat = file_path.stat()
            file_info.append({
                "filename": file_path.name,
                "type": "commands",
                "size": stat.st_size,
                "created": stat.st_mtime,
                "download_url": f"/download-commands?filename={file_path.name}"
            })
        
        # Add config files info
        for file_path in config_files:
            stat = file_path.stat()
            file_info.append({
                "filename": file_path.name,
                "type": "config",
                "size": stat.st_size,
                "created": stat.st_mtime,
                "download_url": f"/download-config?filename={file_path.name}"
            })
        
        # Sort by creation time (newest first)
        file_info.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "files": file_info,
            "total_files": len(file_info),
            "command_files": len(command_files),
            "config_files": len(config_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增端點：下載配置文件
@app.get("/download-config")
async def download_config(filename: str):
    """下載生成的JSON配置文件"""
    try:
        output_dir = Path("restconf_output")
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file {filename} not found")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
