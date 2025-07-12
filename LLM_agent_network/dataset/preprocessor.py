"""
# read the command from the file
# read the telemetry data
# read extra input from the user
# read the output from RAG system
# output the final prompt to the LLM
# LLM output 

RAG + TAIDE inference
─────────────────────
  1.  Build / load a Chroma vector-DB (1-time cost).
  2.  Create a retriever and pull top-k context chunks.
  3.  Format a chat prompt (system + context + user question).
  4.  Run Uns­loth TAIDE model to generate the answer.
"""

# ------------------------- 0. Standard imports -------------------------
from pathlib import Path
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama        # optional sanity check
from langchain.prompts import PromptTemplate

# ------------------------- 1. Build / load vector-DB -------------------
docs_path = Path("docs")
docs_text = docs_path.read_text(encoding="utf-8")

chunks = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
         ).split_text(docs_text)

emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
)

vectordb = Chroma.from_texts(
             texts=chunks,
             embedding=emb,
             persist_directory="db",
          )         # comment out if you already have a persisted DB

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ------------------------- 2.  Load TAIDE model ------------------------
from unsloth import FastLanguageModel

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = "taide/Llama-3.1-TAIDE-LX-8B-Chat",   # do not change
    max_seq_length   = max_seq_length,
    dtype            = None,          # auto bf16 / fp16
    load_in_4bit     = True,          # memory-efficient
    token            = "",
)

device = model.device

# ------------------------- 3.  Prepare system prompt -------------------
rules_text = Path("rules.txt").read_text(encoding="utf-8").strip()

base_system_prompt = (
    rules_text +
    "\n\nYou are a professional assistant. "
    "Answer in Traditional Chinese. "
    "If the retrieved documents do not contain the answer, reply '文件未涵蓋'。"
)

# ------------------------- 4.  RAG helper ------------------------------
def build_chat_prompt(question: str) -> torch.Tensor:
    # 4-1 Retrieve supporting chunks
    docs = retriever.get_relevant_documents(question)
    context = "\n".join(f"【{i+1}】" + d.page_content for i, d in enumerate(docs))

    # 4-2 Fill the template
    tmpl = (
        "以下為可參考文件：\n"
        "{context}\n"
        "------------------------------\n"
        "{question}"
    )
    filled = tmpl.format(context=context, question=question)

    chat = [
        {"role": "system", "content": base_system_prompt},
        {"role": "user",   "content": filled},
    ]

    # Uns­loth helper turns chat → tensor prompt ids
    prompt_ids = tokenizer.apply_chat_template(
        chat,
        tokenize             = True,
        add_generation_prompt= True,   # adds `<|assistant|>` automatically
        return_tensors       = "pt",
    ).to(device)

    return prompt_ids

# ------------------------- 5.  Inference function ----------------------
@torch.no_grad()
def rag_answer(question: str,
               max_new_tokens: int = 256,
               temperature   : float = 0.2,
               top_p         : float = 0.95):
    prompt_ids = build_chat_prompt(question)

    outputs = model.generate(
        prompt_ids,
        max_new_tokens = max_new_tokens,
        do_sample      = True,
        temperature    = temperature,
        top_p          = top_p,
    )

    # Remove prompt tokens → keep only the generated part
    generated = outputs[0][prompt_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ------------------------- 6.  Test ------------------------------------
if __name__ == "__main__":
    q = "台積電 2024 年度營收是多少？"
    print(rag_answer(q))
