import streamlit as st
from rag_core import get_qa_chain

# ====== 1. 页面基础配置 ======
st.set_page_config(page_title="自动化机电知识库专家系统", page_icon="✈️", layout="wide")
st.title("✈️ 自动化与机电设备智能问答系统")
st.caption("基于本地 Qwen2.5 大模型 + LangChain RAG 架构实现，具备上下文记忆功能。")

# ====== 2. 初始化大模型问答链 ======
# 使用缓存避免每次对话都重新加载大模型
@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

# ====== 3. 初始化 UI 聊天历史记录 ======
if "messages" not in st.session_state:
    st.session_state.messages = []

# ====== 4. 侧边栏：增加一个清空记忆的按钮 ======
with st.sidebar:
    st.header("⚙️ 控制面板")
    if st.button("🧹 清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()  # 刷新页面

# ====== 5. 在界面上渲染已有的历史聊天记录 ======
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ====== 6. 接收用户输入并处理 ======
if prompt := st.chat_input("请输入您关于设备维修/操作的技术问题..."):
    
    # 步骤 A：把界面上的聊天记录转换成 LangChain 大模型能看懂的格式
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(("ai", msg["content"]))

    # 步骤 B：在界面上显示用户的最新问题，并存入 UI 记录
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 步骤 C：调用大模型生成回答
    with st.chat_message("assistant"):
        with st.spinner('🧠 正在从本地知识库中检索并思考答案...'):
            try:
                # 同时把“新问题”和“历史记录”传给你的 RAG 引擎！
                response = qa_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history 
                })
                
                answer = response["answer"]
                st.markdown(answer)
                
                # 将大模型的回答追加到 UI 记录中
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
