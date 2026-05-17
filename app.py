import streamlit as st
from rag_core import get_qa_chain

# ====== 1. 页面基础配置 ======
st.set_page_config(page_title="自动化机电知识库专家系统", page_icon="✈️", layout="wide")
st.title("✈️ 自动化与机电设备智能问答系统")
st.caption("基于本地 Qwen2.5 + LangChain LCEL 架构实现集成多轮历史记忆与 BGE-Rerank 重排高精度检索。")

# ====== 2. 初始化大模型问答链 ======
@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

# ====== 3. 初始化 UI 聊天历史记录 ======
if "messages" not in st.session_state:
    st.session_state.messages = []

# ====== 4. 侧边栏：控制面板 ======
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.success("✅ Rerank 重排引擎已激活")
    st.success("✅ 历史记忆功能已就绪")
    
    st.divider()
    if st.button("🧹 清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()  # 刷新页面

# ====== 5. 渲染历史聊天记录（包括引用来源） ======
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果历史记录里保存了参考来源，一并渲染出来
        if "sources" in message and message["sources"]:
            with st.expander("📚 查看参考来源 (Top-3 重排结果)"):
                for i, doc in enumerate(message["sources"]):
                    # 获取文档信息和重排打分
                    source_name = doc.metadata.get("source", "未知文档")
                    page_num = doc.metadata.get("page", "未知页")
                    # 获取 BGE 模型打出的相关性分数，保留4位小数以防太长
                    score = doc.metadata.get("relevance_score", "未知")
                    if isinstance(score, float):
                        score = round(score, 4)
                    
                    # 在界面上把分数展示出来
                    st.markdown(f"**📄 来源 {i+1}** [⭐BGE得分: `{score}`]: `{source_name}` (第 {page_num} 页)")
                    # 只展示前 150 个字符，避免太占地方
                    st.caption(f"{doc.page_content[:150]}...")


# ====== 6. 接收用户输入并处理 ======
if prompt := st.chat_input("请输入您关于设备维修/操作的技术问题..."):
    
    # 步骤 A：格式化历史记录
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(("ai", msg["content"]))

    # 步骤 B：显示用户提问
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 步骤 C：调用大模型并展示思考过程
    with st.chat_message("assistant"):
        # 体现出你的系统在干两件事
        with st.spinner('🔍 正在进行向量粗筛与 Rerank 深度重排，请稍候...'):
            try:
                response = qa_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history 
                })
                
                # 提取答案
                answer = response["answer"]
                st.markdown(answer)
                
                #  提取参考来源 (被 Reranker 筛选出的精华片段)
                source_docs = response.get("context", [])
                
                if source_docs:
                    with st.expander("📚 查看参考来源 (Top-3 重排结果)"):
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get("source", "未知文档")
                            page_num = doc.metadata.get("page", "未知页")
                            # 获取 BGE 模型打出的相关性分数
                            score = doc.metadata.get("relevance_score", "未知")
                            if isinstance(score, float):
                                score = round(score, 4)
                            
                            # 在界面上把分数展示出来
                            st.markdown(f"**📄 来源 {i+1}** [⭐BGE得分: `{score}`]: `{source_name}` (第 {page_num} 页)")
                            st.caption(f"{doc.page_content[:150]}...")


                
                # 将回答和来源一并存入 UI 记忆中
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_docs # 把文档片段也存进状态里
                })
                
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
