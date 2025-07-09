'''
RAG preprocessor
'''
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate

# ① 讀入與切分文件
docs = Path("docs").read_text(encoding="utf-8")
chunks = RecursiveCharacterTextSplitter(
              chunk_size=800, chunk_overlap=100).split_text(docs)

# ② 建立向量資料庫
# LangChain 寫法 ── 把 normalize_embeddings=True 很重要！
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},  # 轉成 unit-norm，和 cosine 距離相容
)
vectordb = Chroma.from_texts(chunks, emb, persist_directory="db")

# ③ 查詢→取回
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ④ 建立生成模型
llm = ChatOllama(model="taide/llama3.1", temperature=0.2)

# ⑤ 串成 RAG graph
template = """你是一位專業助理，下列<<文件>>是你可用的參考資料。
<<文件>>
{context}
<<問題>>
{question}
請用繁體中文回答，若文件沒有答案就說「文件未涵蓋」。"""
rag_prompt = PromptTemplate.from_template(template)

rag_chain = RunnableParallel(
    context   = retriever,
    question  = RunnablePassthrough()
) | rag_prompt | llm

print(rag_chain.invoke("台積電 2024 年度營收是多少？"))
