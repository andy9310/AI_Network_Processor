## simulation in EVENG
we need to first forwrading to the EVENG VM in virtual box
## algorithm

we remain the heuristic-rule
H-Rule 仍負責確保連通與不超載；RL 只微調「開關頻率」或「觸發點」，不易導致災難性行動。

可遷移 (transferability)
不同網路規模下，動作維度不變；訓練好的 agent 可微調就遷移。

## LLM usage 
install dependencies 
```
!pip install transformers==4.37.2
!pip install unsloth==2025.2.15 unsloth_zoo==2025.2.7
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
!pip install --no-deps cut_cross_entropy
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
```

## running api file 
'''
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
ps aux | grep uvicorn
'''