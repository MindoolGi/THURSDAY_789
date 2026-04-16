import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS  # Chroma 대신 FAISS 사용
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# [수정] Secrets에서 API 키 로드 (Streamlit Cloud용)
api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def process_pdf():
    # [주의] Github에 PDF를 올릴 때는 './data/파일명.pdf' 처럼 상대 경로를 쓰세요.
    # 아래는 현재 코드상의 경로입니다. 필요시 수정하세요.
    file_path = "C:/RAG/Ch01/data/2024_KB_부동산_보고서_최종.pdf"
    
    if not os.path.exists(file_path):
        st.error(f"PDF 파일을 찾을 수 없습니다: {file_path}")
        return []

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def initialize_vectorstore():
    # [요구사항 1, 2] FAISS 사용 및 로컬 저장/로드 로직
    faiss_index_path = "./faiss_db"
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    if os.path.exists(faiss_index_path):
        # 이미 존재하면 로드 (위험 방지 옵션 필수)
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # 존재하지 않으면 새로 생성 후 저장
        chunks = process_pdf()
        if not chunks: return None
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local(faiss_index_path)
    return vectorstore

def get_session_history(session_id):
    # 세션 상태에 히스토리 저장소 생성
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = {}
    
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    
    # [요구사항 3] 최근 4번의 대화(발화)만 남기도록 Trim
    history = st.session_state.chat_store[session_id]
    if len(history.messages) > 4:
        history.messages = history.messages[-4:]
        
    return history

@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    if not vectorstore: return None
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.
컨텍스트: {context}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"), # chat_history 필수 설정
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
        get_session_history,
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
        if chain:
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

# [중요] Streamlit Cloud 배포 시 하단의 ngrok 코드는 삭제하거나 주석 처리하세요.
# Cloud 환경은 이미 고유의 URL(streamlit.app)을 제공하므로 ngrok이 불필요하며 에러를 유발합니다.
