from fastapi import FastAPI, HTTPException
from pathlib import Path
import random, json, os, torch, textwrap
from unsloth import FastLanguageModel
from dotenv import load_dotenv
from collect import collect_link_traffic, fetch_generic_counters_legacy
load_dotenv()
# ────────────────────── ❶ 載入模型 ──────────────────────────
MAX_SEQ_LEN  = 2048
MODEL_NAME   = "taide/Llama-3.1-TAIDE-LX-8B-Chat"

print("🚀 Loading model… (只載一次)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,                # 🔧 4-bit 量化→ 8 GB VRAM 夠用
    token          = os.getenv("HUGGINGFACE_TOKEN", ""),
)                 # 🔧 一次性把整個模型搬上 GPU
print("✅ Model on", model.device)

# ────────────────────── ❷ System prompt ───────────────────
RULES_PATH = Path("rules.txt")
rules_text = RULES_PATH.read_text(encoding="utf-8")

SYSTEM_PROMPT = textwrap.dedent(f"""{rules_text.strip()}

你是 Cisco IOS XR 網管助手，只輸出 RESTCONF curl 指令與對應 config JSON，
禁止加上任何解釋或多餘文字。""")

# ────────────────────── ❸ Telemetry 產生器 (略)… ───────────
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
# ────────────────────── ❹ LLM 推論 ─────────────────────────
def llm_inference(links_to_close=None):
    if links_to_close is None:
        links_to_close = ["L1", "L3", "L6"]

    telemetry_json = json.dumps(collector(), ensure_ascii=False)
    user_prompt = (
        f"以下是即時流量資料（JSON）：\n{telemetry_json}\n\n"
        f"請根據上述資料，關閉 {', '.join(links_to_close)}"
    )

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
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

# ────────────────────── ❺ FastAPI 端點 (原樣) ──────────────
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

@app.get("/output")
async def get_output(links: str | None = None):
    try:
        result = llm_inference(links.split(",") if links else None)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ────────────────────── ❻ 啟動 (保持原有) ───────────────────
# if __name__ == "__main__":
#     import uvicorn, sys
#     uvicorn.run("main:app", host="0.0.0.0", port=8000,
#                 reload="--reload" in sys.argv)
