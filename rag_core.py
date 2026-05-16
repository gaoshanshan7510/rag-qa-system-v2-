import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

# --- 核心修改点 1：新导入了支持记忆功能的包 ---
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. 配置参数
DOC_PATH = "./my_knowledge_docs"  # PDF存放路径！
DB_PATH = "./chroma_db"           # 向量数据库保存路径

llm = Ollama(model="qwen2.5:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def build_vector_db():
    """读取PDF并构建向量数据库"""
    print("正在加载PDF文档...")
    loader = PyPDFDirectoryLoader(DOC_PATH)
    docs = loader.load()
    print("正在将文档切分为小块...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print("正在生成向量并存入数据库 ...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_PATH)
    print("数据库构建完成！")
    return vectorstore

def get_qa_chain():
    """构建带有记忆功能的问答链"""
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 添加一个专门把“追问”翻译成完整句子的 Prompt ---
    contextualize_q_system_prompt = (
        "给定聊天历史记录和最新的用户问题，"
        "该问题可能会引用历史记录中的上下文，"
        "请制定一个不需要历史记录也能听懂的独立问题。"
        "注意：不要回答问题，只需在需要时重新表述，否则按原样返回。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), # 动态占位符，专门用来放聊天记录
        ("human", "{input}"),
    ])
    
    # 得到一个“懂历史”的高级检索器
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "你是一个专业的航空航天与自动化设备工程师。请使用以下检索到的检索内容来回答用户的问题。"
        "如果你不知道答案，请直接说不知道，不要编造。回答要尽量专业、有条理。\n\n"
        "检索内容：\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), # 给模型注入记忆
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 将“懂历史的检索器”和“问答链”连起来
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_vector_db()
        
    chain = get_qa_chain()
    
    # 初始化一个空列表，用来当大脑存记忆
    chat_history = [] 
    
    print("=========== 开始测试多轮对话 ===========")
    
    question1 = "什么是闭环控制系统？" 
    print(f"\n👤 用户: {question1}")
    
    response1 = chain.invoke({
        "input": question1,
        "chat_history": chat_history # 第一次提问时，历史记录是空的
    })
    answer1 = response1["answer"]
    print(f"🤖 AI回答: {answer1}")
    
    # 把第一轮的对话，塞进记忆列表里
    chat_history.extend([
        HumanMessage(content=question1),
        AIMessage(content=answer1)
    ])
    
    # 第二轮对话：故意进行模糊的追问
    question2 = "那它的优点和缺点是什么？" 
    print(f"\n👤 用户 (故意追问): {question2}")
    
    response2 = chain.invoke({
        "input": question2,
        "chat_history": chat_history # 把包含第一轮对话的记忆传给它
    })
    print(f"🤖 AI回答: {response2['answer']}")
