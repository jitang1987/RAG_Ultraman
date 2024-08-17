import os
import gradio as gr
from pathlib import Path
from datetime import datetime
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import whisper

# Step 1: Initialize API key
nvapi_key = "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"
os.environ["NVIDIA_API_KEY"] = nvapi_key

# Step 2: Initialize the LLM and embedding models
llm = ChatNVIDIA(model="microsoft/phi-3-small-128k-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)
embedder = NVIDIAEmbeddings(model="NV-Embed-QA")

# Initialize Whisper model for STT
whisper_model = whisper.load_model("tiny")

# Define paths
data_folder = "./zh_data/"
vector_store_path = "./zh_data/nv_embedding"

# Function to get the latest modification time of the documents
def get_latest_modification_time(data_folder):
    latest_time = 0
    for p in os.listdir(data_folder):
        if p.endswith('.txt'):
            path2file = os.path.join(data_folder, p)
            file_time = os.path.getmtime(path2file)
            if file_time > latest_time:
                latest_time = file_time
    return latest_time

latest_mod_time = get_latest_modification_time(data_folder)
formatted_time = datetime.fromtimestamp(latest_mod_time).strftime('%Y-%m-%d %H:%M:%S')

# Step 3: Rebuild the vector store based on current .txt files
ps = os.listdir(data_folder)
data = []
sources = []
for p in ps:
    if p.endswith('.txt'):
        path2file = os.path.join(data_folder, p)
        with open(path2file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) >= 1:
                    data.append(line)
                    sources.append(path2file)

# Data cleaning
documents = [d for d in data if d.strip() != '']

# Batch processing with concurrency
def batch_process(docs, batch_size=5):
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]

def embed_batch(batch):
    return embedder.embed_documents(batch)

if documents:
    # Creating a vector store from the documents and saving it to disk
    text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n\n")
    docs = []
    metadatas = []

    for i, d in enumerate(documents):
        splits = text_splitter.split_text(d)
        if splits:
            docs.extend(splits)
            metadatas.extend([{"source": sources[i]}] * len(splits))

    if docs:
        try:
            all_embeddings = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(embed_batch, batch) for batch in batch_process(docs, batch_size=5)]
                for future in futures:
                    all_embeddings.extend(future.result())
            store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
            store.save_local(vector_store_path)
        except Exception as e:
            print(f"Error during embedding: {e}")
            store = None
    else:
        print("No valid document chunks to embed.")
        store = None
else:
    print("No documents found. Skipping vector store creation.")
    store = None

# Step 4: Only proceed if the vector store is not None
if store:
    retriever = store.as_retriever(search_kwargs={"k": 5, "filter_duplicates": True})  # 降低 k 值并过滤重复

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "请根据以下内容详细回答问题，并确保答案简洁且没有重复内容。\n<Documents>\n{context}\n</Documents>",
            ),
            ("user", "{question}"),
        ]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Function to convert speech to text
    def speech_to_text(audio_path):
        result = whisper_model.transcribe(audio_path, word_timestamps=True, fp16=False, language='zh')
        return result["text"]

    # Function to handle the input and return the result
    def ask_question(audio_file=None, question=None):
        if audio_file:
            question = speech_to_text(audio_file)
            print(f"识别到的文本: {question}")
        
        result = chain.invoke(question)
        print(f"生成的答案: {result}")
        return f"识别的文本：{question}\n\n生成的答案：{result}"

    # Gradio interface
    iface = gr.Interface(
        fn=ask_question,
        inputs=[
            gr.Audio(type="filepath", label="点击下方的麦克风按钮说出你想要了解的奥特曼的问题"),
            gr.Textbox(lines=2, placeholder="要相信光...", label="或者输入你想要了解的奥特曼的知识")
        ],
        outputs="text",
        title="奥特曼知识库语音文本问答系统",
        description=f"奥特曼知识库更新时间：{formatted_time}\n\n用文本或用语音了解关于奥特曼的知识，答案会出现在右侧"
    )

    iface.launch()
else:
    print("No vector store available. Exiting.")
