# api.py
from fastapi import FastAPI, HTTPException
from telemetry import collect_caller  # ← 匯入剛才那支函式
from LLM_agent_network.dataset import inference
app = FastAPI(
    title="EVE-NG XR Telemetry API",
    description="Expose XR generic-counters via REST",
    version="1.0.0",
)

# ── API #1：給自己的應用程式取即時 Telemetry ───────────────
@app.get("/telemetry")
async def get_telemetry():
    """
    立即呼叫 XR Netconf 拿最新 generic-counters。
    回傳格式 = telemetry.collect_stats() 的原始 dict。
    """
    try:
        data = collect_caller()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/output")
async def get_output():
    """
    ouput of LLM
    """
    try:
        data = inference()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── API #2：示範另一用途（健康檢查或自訂摘要）──────────────
@app.get("/health")
async def health_check():
    """簡易存活檢查，可給外部監控使用。"""
    return {"status": "ok"}