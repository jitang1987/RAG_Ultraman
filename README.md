## 要相信光 — 基于RAG的奥特曼知识库系统（我知道名字很傻）

 

项目名称：要相信光 — 基于RAG的奥特曼知识库系统（我知道名字很傻）

报告日期：2024年8月17日

项目负责人：Tang Ji

 

#### 项目概述：

**项目背景：**

“奥特曼”系列是日本特摄片中的经典之作，自1966年首次登场以来，已发展出多个系列和衍生作品，深受全球观众尤其是青少年和动漫爱好者的喜爱。随着时间的推移，奥特曼宇宙中的角色和故事不断丰富，涵盖了数十种不同的奥特曼角色及其各自的背景故事、技能和战斗经历。为了帮助奥特曼的粉丝、研究人员以及文化爱好者更好地理解和探索奥特曼的广阔宇宙，本项目旨在创建一个基于RAG（Retrieval-Augmented Generation）的奥特曼知识库系统。

**项目目标：**

“要相信光 — 基于RAG的奥特曼知识库语音文本问答系统”旨在通过整合和处理大量与奥特曼相关的文本数据，建立一个智能问答系统，使用户能够通过自然语言交互，获取有关奥特曼的详细信息。该系统将通过最新的自然语言处理技术，结合信息检索与生成的强大能力，提供准确且丰富的知识支持。

**项目涉及内容**

项目涉及RAG，语音合成技术。通过语音识别提问者的问题，转成文字，进而利用RAG和LLM进行知识库的检索和答案生成。



#### 技术方案与实施步骤

l 模型选择： 项目选择的模型为`phi-3-small-128k-instruct`。理由是这个模型具有相对较小的参数量（128k），非常适合部署在资源有限的边缘设备上，例如树莓派或者其他嵌入式设备。此模型也适合部署在小型嵌入式设备上，这个小型设备可以内置在奥特曼玩具中，使熊孩子可以和奥特曼进行对话，并在奥特曼的大脸屏幕上显示文本答案。

 embedding模型选择了`NV-Embed-QA`

-  **数据构建**

1. 文本知识库的构建
   - 建立了一系列关于奥特曼知识的txt文档。如下图
   - ![RAG_data](G:\My Drive\UoA\02.PhD Study\00.Document\01.Study\10.NVIDIA Summer Camp\RAG_data.png)

2. 读取文本数据：
   - 在 `data_folder` 中存放的所有 `.txt` 文件会被逐一读取。
   - 通过 `os.listdir()` 获取目录下的文件列表，然后逐个打开每个文件，读取其中的内容。
   - 所有文本数据存储在 `data` 列表中，每个文件的路径存储在 `sources` 列表中。

3. 数据清洗：
   - 对从文件中读取的文本数据进行清洗，去除空行和空格。
   - 确保只保留非空文本数据，从而在后续的处理过程中避免无效数据的干扰。

4. 文本分割：
   - 使用 `CharacterTextSplitter` 对文本数据进行分割。这里使用的 `chunk_size` 参数定义了每段文本的最大长度（500个字符），`separator` 用于分隔段落。
   - 文本被分割成多个段落，每个段落将被单独处理和存储。

5. 构建向量存储：
   - 文本数据经过清洗和分割后，将通过嵌入模型 `NVIDIAEmbeddings` 进行向量化处理。
   - 通过多线程并发（`ThreadPoolExecutor`）处理文档分割后的文本数据，进一步提升处理效率。
   - 向量化后的数据将被存储在 `FAISS` 中，构建索引以便于后续的高效检索。

6. 保存向量存储：
   - 处理后的向量数据及其对应的元数据（例如来源文件路径）被保存在指定的路径（`vector_store_path`）中，以便在后续的查询中直接加载使用。

- 向量化处理方法

1. 嵌入模型 (`NVIDIAEmbeddings`)：
   - 使用 `NVIDIAEmbeddings` 模型将文本段落向量化。每个文本段落被转换成一个固定长度的向量表示，这些向量捕捉了文本的语义信息。
   - 向量化的文本可以被用于相似性搜索、分类或聚类等任务。

2. 向量化的优势：
   - **高效检索**：向量化处理后的文本可以通过 `FAISS` 库进行快速检索。`FAISS` 支持高效的相似性搜索，尤其在大规模数据集上表现出色。
   - **语义理解**：通过嵌入模型生成的向量表示可以捕捉文本的语义信息，从而支持语义层面的搜索，而不仅仅是关键词匹配。这使得搜索引擎能够理解自然语言查询，并返回语义相关的结果。
   - **可扩展性**：使用 `FAISS` 进行向量存储和检索，能够处理大规模数据集，同时通过多线程处理进一步提高处理速度。

- 优势总结
  - **语义搜索**：相比传统的关键词搜索，语义搜索能更好地理解用户的意图，并返回更相关的结果。
  - **高效处理**：使用多线程并发处理文本数据，提高了数据处理的效率，特别是在处理大规模数据集时表现更为显著。
  - **可扩展性**：向量化处理和 `FAISS` 的结合使系统能够轻松扩展到大规模数据，并在不损失性能的情况下进行实时检索。


通过以上过程，系统能够将文本数据转化为语义向量，并通过高效的检索机制，在接收到查询请求时返回相关性高且符合用户意图的答案。

 

l **功能整合**： 代码实现了一个结合语音识别（STT，Speech-to-Text）和自然语言处理的问答系统，特别是围绕奥特曼知识库进行的语音与文本问答。系统使用了多个高级技术组件，包括 Whisper 模型进行语音识别，以及 NVIDIA 的 LLM 进行问题解答。

#### 实施步骤：

- **环境搭建**

1. 软件与工具准备

   - 操作系统：建议使用 Linux 或 Windows 作为开发环境。

   - Python 版本：建议使用 Python 3.8 或更高版本。

2. 安装必要的软件和库

首先，确保已经安装了 Python 和 pip（Python 的包管理工具）。接下来，通过命令行安装以下必要的库和工具：

```bash
# 创建虚拟环境（可选）
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate  # Windows

# 更新 pip
pip install --upgrade pip

# 安装所需的库
pip install gradio
pip install whisper
pip install langchain_nvidia_ai_endpoints
pip install langchain_community
pip install faiss-cpu  # 如果使用CPU版本的FAISS
```

3. NVIDIA AI Endpoints 配置

   - 获取 NVIDIA API Key，并将其配置为环境变量，如代码中的 `nvapi_key`。

   - 使用 NVIDIA 的模型时，确保网络通畅，并检查 API Key 的权限和限额。

- **代码实现**

`关键代码实现步骤`

1. 初始化环境与模型

   - 通过设置 API Key 初始化 NVIDIA 的 LLM 模型和嵌入模型（用于向量化处理）。
   - 使用 Whisper 模型进行语音识别的初始化。

    ```python
    nvapi_key = "your_nvidia_api_key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key
   
    llm = ChatNVIDIA(model="microsoft/phi-3-small-128k-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)
    embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
   
    whisper_model = whisper.load_model("tiny")
    ```

2. 数据构建与清洗

   - 从指定文件夹中读取 `.txt` 文件，进行数据清洗与分割。
   - 通过 `CharacterTextSplitter` 将长文本分割成较小的块。

    ```python
   data = []
   for p in os.listdir(data_folder):
       if p.endswith('.txt'):
           with open(os.path.join(data_folder, p), encoding="utf-8") as f:
               data.extend([line.strip() for line in f if line.strip()])
    ```

3. 向量化与存储

   - 使用嵌入模型将文本向量化，并存储到 FAISS 中，用于后续的快速检索。

    ```python
   store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
   store.save_local(vector_store_path)
    ```

4. 语音识别与问答处理

   - 通过 Whisper 模型将用户的语音输入转化为文本。
   - 将文本输入传递给 LLM 生成相应的答案。

    ```python
    def speech_to_text(audio_path):
        result = whisper_model.transcribe(audio_path, word_timestamps=True, fp16=False, language='zh')
        return result["text"]
   
    def ask_question(audio_file=None, question=None):
        if audio_file:
            question = speech_to_text(audio_file)
        result = chain.invoke(question)
        return f"识别的文本：{question}\n\n生成的答案：{result}"
    ```

5. 界面搭建

   - 使用 Gradio 库创建一个简单的用户界面，支持语音输入和文本输入，并显示生成的答案。

    ```python
   iface = gr.Interface(
       fn=ask_question,
       inputs=[gr.Audio(type="filepath", label="录制或上传语音"), gr.Textbox(lines=2, placeholder="要相信光...")],
       outputs="text",
       title="奥特曼知识库语音文本问答系统",
       description=f"奥特曼知识库更新时间：{formatted_time}\n\n输入你想要了解的关于奥特曼的知识"
   )
   iface.launch()
    ```

- **测试与调优**

1. 测试用例设计

   - 语音输入测试：上传或录制不同的中文语音，确保语音识别结果正确无误。

   - 文本输入测试：测试文本输入能否生成正确的答案。

   - 多轮对话测试：连续输入问题，测试系统的上下文理解能力和多轮对话生成质量。

   - 边界条件测试：如空白输入、噪声较大的语音、超长文本等，测试系统的鲁棒性。

2. 性能调优

   - 提高检索效率：调整 `k` 值，优化检索返回的文档数量，减少重复。

   - 并发处理优化：使用 `ThreadPoolExecutor` 提高嵌入过程的速度。

   - 模型优化：根据项目需求选择更适合的 LLM 模型版本，如更小或更快的版本。

- **集成与部署**

1. 模块集成
   - 模型集成：确保 Whisper 模型、NVIDIA LLM 模型和 FAISS 向量存储模块的正确集成。
   - 界面集成：通过 Gradio 创建的界面，允许用户通过网络浏览器访问系统，确保语音和文本处理模块能够无缝对接。

2. 部署到运行环境
   - 本地部署：在本地机器上运行 Python 脚本，确保本地网络能访问相关的 NVIDIA API 服务。
   - 云端部署：如果需要云端部署，可将脚本放置在云服务器上，确保所有依赖项在服务器上正确安装和配置。
   - 持续集成与监控：设置自动化脚本定期更新知识库数据和模型版本，同时监控系统性能和用户反馈。

 

#### 项目成果与展示：

l 应用场景展示： 非常适合部署在资源有限的边缘设备上，例如树莓派或者其他嵌入式设备。此模型也适合部署在小型嵌入式设备上，这个小型设备可以内置在奥特曼玩具中，使熊孩子可以和奥特曼进行对话，并在奥特曼的大脸屏幕上显示文本答案。项目适合用于儿童教育领域。

l 功能演示： 列出并展示实现的主要功能，附上UI页面截图，直观展示项目成果。

![RAG_奥特曼](G:\My Drive\UoA\02.PhD Study\00.Document\01.Study\10.NVIDIA Summer Camp\RAG_奥特曼.png)

1. **语音识别 (Speech-to-Text)**
   - 功能描述：用户可以通过录制或上传语音文件，系统会自动识别语音内容并将其转换为文本，显示在界面上。
   - 演示步骤：
     1. 在 Gradio 界面中点击“录制或上传语音”按钮。
     2. 录制一段语音，或者选择一个已经保存的语音文件上传。
     3. 系统将自动识别语音内容，并在界面上显示识别出的文本。
2. **文本输入与问答生成**
   - 功能描述：用户可以在输入框中输入问题，系统会根据奥特曼知识库生成相应的答案并显示。
   - 演示步骤：
     1. 在 Gradio 界面的文本输入框中输入关于奥特曼的问题，如“请告诉我有哪些奥特曼？”
     2. 点击提交按钮后，系统将生成并返回一个详细的答案，显示在界面上。
3. **语音与文本结合**
   - 功能描述：用户可以同时使用语音和文本进行查询，系统可以识别语音文本并返回答案。
   - 演示步骤：
     1. 录制语音并输入额外的文本内容。
     2. 系统将结合语音识别和文本输入的内容，生成答案并展示在界面上。
4. **答案展示与反馈**
   - 功能描述：系统根据用户输入（语音或文本）生成相应的答案，并且显示在界面的输出区域。
   - 演示步骤：
     1. 用户提交问题后，系统在几秒钟内生成答案。
     2. 答案展示在界面输出区域，用户可以看到系统识别的文本以及生成的详细答案。

- **UI 页面描述**
  - **页面结构**：
    - 顶部为系统标题，如“奥特曼知识库语音文本问答系统”。
    - 中间部分为用户输入区域，包含语音输入控件和文本输入框。
    - 底部为系统生成的答案展示区域，显示系统识别的文本以及生成的答案。

  - **界面布局**：
    - 左侧为录制或上传语音的按钮。
    - 右侧为文本输入框，用户可以直接输入文本查询。
    - 下方为答案输出区域，显示识别出的文本和生成的答案。

- **运行演示项目**

1. **启动项目**

   - 在终端中运行代码，将启动 Gradio 界面，用户可以通过浏览器访问本地运行的应用程序。

   ```bash
   python test.py
   ```

2. **访问 UI 界面**

   - 在浏览器中输入 `http://127.0.0.1:7860` 或者命令行中显示的本地 URL，访问 Gradio UI 页面。

3. **操作演示**

   - 尝试录制或上传语音文件，并输入相关的文本内容。系统会自动识别并生成答案，显示在界面上。
   - 用户可以通过刷新页面或更改输入内容来测试系统的不同功能。

 

#### 问题与解决方案：

1. **语法错误**
   - 问题描述

在编写和修改代码时，出现了一些语法错误。例如，某处代码中的多余括号导致 Python 解释器无法正确解析代码，出现了 `SyntaxError`。

- ​	解决措施

我们通过逐步检查代码，发现并删除了多余的括号或不匹配的括号。为了避免此类错误，我们采用了以下措施：

- ​	使用代码编辑器的语法检查功能。
  - 在每次代码修改后立即运行以捕捉早期的语法问题。


2. **依赖库问题**

- 问题描述

在集成不同的库时，遇到了一些兼容性和配置问题：

- **Whisper 模型**：在使用 Whisper 进行语音识别时，出现了版本警告和模型加载问题。
- **Gradio**：在处理音频输入时，使用了不支持的参数（如 `source` 和 `optional`），导致 `TypeError`。

- 解决措施

- **Whisper 模型**：对于 Whisper 的版本警告问题，我们注意到这是关于安全性的提醒，因此根据项目需求继续使用了当前配置。此外，我们确保使用 `torch.load` 的正确参数来加载模型。
- **Gradio**：我们通过查阅 Gradio 文档，了解了音频组件的正确用法，移除了不支持的参数，并确保在最新版本的 Gradio 中音频功能正常运行。

3. **输入输出不一致**
   - 问题描述


在某些情况下，生成的答案显示在终端而不是预期的 Gradio 界面中。特别是在语音识别完成后，答案没有出现在用户界面，而是通过终端输出。

- ​	解决措施

我们检查了代码中对输出结果的处理逻辑，确保所有输出都通过 Gradio 的 `outputs` 参数传递并显示在界面上。具体措施包括：

- ​	确保函数 `ask_question` 的返回值正确传递给 Gradio 的 `outputs`。
  - 在界面中正确配置 `outputs` 参数，确保将结果显示在用户界面上。


4. **重复性答案**

- 问题描述

生成的答案中存在重复内容，影响了用户体验。例如，在回答中多次出现相同的奥特曼角色描述，这可能是由于检索时返回了过多相似的文本片段。

- 解决措施
  - **调整 `k` 值**：在检索阶段，我们将 `k` 值从较高值调整为 5，以减少检索到的文档片段数量，从而减少重复。
  - **添加去重逻辑**：在 `retriever` 中增加了 `filter_duplicates=True` 参数，以自动过滤掉相似或重复的内容。






#### 项目总结与展望：

l 项目评估： 项目成功集成了语音识别（STT）和RAG技术，实现了语音与文本结合的问答系统。用户可以通过语音或文本方式进行查询，系统能够实时生成准确的答案。通过 Gradio 库构建了一个简洁直观的用户界面，用户能够轻松地录制或上传语音、输入文本，并在界面上查看系统生成的答案。在部分情况下，系统生成的答案存在重复或不准确的情况，尤其在面对复杂或多层次问题时。系统的检索与生成机制仍有改进空间，以确保答案的多样性和准确性。项目依赖于 NVIDIA 的 API 服务进行语言模型调用，可能会受到 API 调用限制或网络状况的影响，导致系统在某些情况下的可用性受到限制。



l 未来方向： 在系统中集成 TTS 技术，将生成的文本答案合成为语音输出，实现全语音交互，增强用户体验。增加语音情感识别功能，使系统能够根据用户的情感状态调整回答内容和语气，提供更加人性化的服务。进一步优化 Gradio 界面的交互性，增加动态反馈和可视化元素，如图表、图片或视频，以增强用户的视觉体验。

 

 

#### 附件与参考资料

 [Try NVIDIA NIM APIs](https://build.nvidia.com/explore/discover)

https://github.com/kinfey/Microsoft-Phi-3-NvidiaNIMWorkshop

 https://github.com/rany2/edge-tts
