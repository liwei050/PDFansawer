from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符文本分割器
from langchain_community.embeddings import HuggingFaceEmbeddings  # 改用HuggingFace接口
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain  # 对话检索链


def qa_agent(openai_api_key, memory, uploaded_file, question):
    # 修改为使用DeepSeek的API密钥格式
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=openai_api_key,  # 确保这是DeepSeek提供的有效API密钥
        openai_api_base="https://api.deepseek.com"
    )

    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "，", "？", "！", "、", ""]
    )

    texts = text_splitter.split_documents(docs)
    # 添加检查确保分割后的文本不为空
    if not texts:
        raise ValueError("文档分割后没有得到有效文本内容")

    # 修改为使用HuggingFace接口的BGE-M3模型
    # 修改为使用本地模型路径
    embeddings = HuggingFaceEmbeddings(
        model_name="/Users/liwei/Desktop/test/bge-m3",  # 本地模型路径
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        memory=memory,
        retriever=retriever
    )

    response = qa.invoke({"chat_history": memory, "question": question})
    return response
