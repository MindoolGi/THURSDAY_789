import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 환경 변수 로드
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("C:/RAG/Ch01/data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def initialize_vectorstore():
    persist_directory = "./chroma_db"
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        chunks = process_pdf()
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_directory
        )
    return vectorstore

@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.
컨텍스트: {context}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history",
    )

def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        chain = initialize_chain()
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

# 2. ngrok 연결
from pyngrok import ngrok
# 만약 authtoken 설정이 안되어 있다면 실행: ngrok.set_auth_token("여러분의_토큰")
public_url = ngrok.connect(8501)
ngrok.set_auth_token("3CQ91BhcxwuTa1cubME7xyUKl6J_7ZkYKjoBh7szzuPEdsNi")

try:
    # 8501 포트를 외부로 개방
    public_url = ngrok.connect(8501)
    print("\n" + "="*60)
    print("🎉 챗봇 배포 완료!")
    print(f"🔗 접속 주소: {public_url}")
    print("="*60)
    print("위 주소를 클릭하면 외부에서도 챗봇에 접속할 수 있습니다.")
except Exception as e:
    print(f"❌ 배포 중 오류 발생: {e}")