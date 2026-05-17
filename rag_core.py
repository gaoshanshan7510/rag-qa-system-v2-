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

# --- 核心修改点 2： 重排序 Reranker 所需的包 ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from typing import Sequence, Any, Optional
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks


# 1. 配置参数
DOC_PATH = "./my_knowledge_docs"  # PDF存放路径！
DB_PATH = "./chroma_db"           # 向量数据库保存路径

llm = Ollama(model="qwen2.5:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

class ScoreInjectingReranker(BaseDocumentCompressor):
    model: Any
    top_n: int = 3

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        # 1. 将用户问题和粗筛出来的文档组装成对
        texts = [[query, doc.page_content] for doc in documents]
        
        # 2. 让 BGE 模型打分
        scores = self.model.score(texts)
        
        # 3. 把文档和分数绑定在一起
        docs_with_scores = list(zip(documents, scores))
        
        # 4. 按分数从高到低排序
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # 5. 提取前 N 个，并把分数强行塞进 metadata
        final_docs = []
        for doc, score in docs_with_scores[:self.top_n]:
            # 把低于 0.1 分的垃圾文档直接扔掉，就取消注释下面两行
            if score < 0.1:
              continue 
                
            new_metadata = doc.metadata.copy()
            new_metadata["relevance_score"] = float(score) # 🔥 强制记录分数！
            final_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))
            
        return final_docs


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
    """构建带有记忆功能 + Rerank重排 的终极问答链"""
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 把“粗筛”扩大到 10 个 
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 加载 BGE 重排模型 
    # 这里使用智源研究院开源的霸榜重排模型 bge-reranker
    print("正在加载 Reranker 重排模型（首次运行会自动下载，请耐心等待）...")
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    # 从 10 个里挑出最准的 3 个 
   # 因为版本不兼容等原因我使用自定义的重排器替代官方的
    compressor = ScoreInjectingReranker(model=model, top_n=3)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    contextualize_q_system_prompt = (
        "给定聊天历史记录和最新的用户问题，"
        "该问题可能会引用历史记录中的上下文，"
        "请制定一个不需要历史记录也能听懂的独立问题。"
        "注意：不要回答问题，只需在需要时重新表述，否则按原样返回。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 把超级检索器 (compression_retriever) 传给大脑 
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    system_prompt = (
        "你是一个专业的航空航天与自动化设备工程师。请使用以下检索到的检索内容来回答用户的问题。"
        "如果你不知道答案，请直接说不知道，不要编造。回答要尽量专业、有条理。\n\n"
        "检索内容：\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_vector_db()
        
    chain = get_qa_chain()
    
    # 初始化一个空列表，用来当大脑存记忆
    chat_history = [] 
    
    print("=========== 开始测试多轮对话 & 重排打分 ===========")
    
    # ---------------- 第一轮对话 ----------------
    question1 = "什么是闭环控制系统？" 
    print(f"\n👤 用户: {question1}")
    
    response1 = chain.invoke({
        "input": question1,
        "chat_history": chat_history # 第一次提问时，历史记录是空的
    })
    answer1 = response1["answer"]
    
    # 查看 Reranker 的打分
    print("\n🔍 [后台探秘] Reranker 为第一轮问题找出的最强资料：")
    for i, doc in enumerate(response1.get("context", [])):
        score = doc.metadata.get('relevance_score', '未知') 
        print(f"  🏆 第{i+1}名 (BGE得分: {score}) -> {doc.page_content[:60]}...")
        
    print(f"\n🤖 AI回答: {answer1}")
    
    # 把第一轮的对话，塞进记忆列表里
    chat_history.extend([
        HumanMessage(content=question1),
        AIMessage(content=answer1)
    ])
    
    # ---------------- 第二轮对话 ----------------
    question2 = "那它的优点和缺点是什么？" 
    print(f"\n\n👤 用户 (故意追问): {question2}")
    
    response2 = chain.invoke({
        "input": question2,
        "chat_history": chat_history # 把包含第一轮对话的记忆传给它
    })
    
     # 查看 Reranker 的打分
    print("\n🔍 [后台探秘] Reranker 为第一轮问题找出的最强资料：")
    for i, doc in enumerate(response1.get("context", [])):
        # 直接把这块文档携带的所有隐藏属性全部打印出来
        print(f"👉 来源 {i+1} 的全部隐藏属性: {doc.metadata}")
        
        score = doc.metadata.get('relevance_score', '未知') 
        print(f"  🏆 第{i+1}名 (BGE得分: {score}) -> {doc.page_content[:60]}...")

        
    print(f"\n🤖 AI回答: {response2['answer']}")

