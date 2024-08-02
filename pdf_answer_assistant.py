import streamlit as st
import os

from langchain.memory import ConversationBufferMemory
from utils import generate_result

st.title('PDF Assistant 💡')

with st.sidebar:
    openai_api_key = st.text_input("请输入OPENAI API 密钥", type="password", value=os.getenv("OPENAI_API_KEY"))
    st.markdown("[获取OPENAI API KEY 密钥](https://openai.com/)")

pdf_file = st.file_uploader(label="请上传的你的PDF文件", type="pdf")

question = st.text_input(label="请输入你的问题", disabled=not pdf_file)

if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

if pdf_file and question and not openai_api_key:
    st.info("请输入OPENAI API KEY")
    st.stop()

if pdf_file and question and openai_api_key:
    with st.spinner("AI 正在检索PDF文件，请稍后..."):
        response = generate_result(question=question,
                                   memory=st.session_state["memory"],
                                   openai_api_key=openai_api_key,
                                   file_docs=pdf_file)
        st.markdown("#### 答案")
        st.write(response['answer'])
        st.session_state['chat_history'] = response['chat_history']

if "chat_history" in st.session_state:
    with st.expander(label="查看历史信息"):
        for i in range(0, len(st.session_state['chat_history']), 2):
            human_history = st.session_state['chat_history'][i]
            ai_history = st.session_state['chat_history'][i + 1]
            st.write(human_history.content)
            st.write(ai_history.content)
            if i < len(st.session_state['chat_history']) - 2:
                st.divider()

