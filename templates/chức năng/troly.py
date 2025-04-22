import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit app
st.set_page_config(page_title="Trợ lý Tài chính AI", layout="wide")
st.title("🤖 Trợ lý Tài chính AI - Đọc & Học từ Tài liệu")

uploaded_file = st.file_uploader("Tải lên tài liệu PDF về phương pháp trade hoặc tài chính:", type="pdf")

if uploaded_file:
    with st.spinner("Đang đọc tài liệu..."):
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load()

        # Cắt nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)

        # Tạo embeddings và vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="db")
        vectordb.persist()

        # Tạo chuỗi truy vấn
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        st.success("Tài liệu đã được xử lý! Bây giờ bạn có thể hỏi bất kỳ điều gì từ nó.")

        query = st.text_input("Hãy nhập câu hỏi về tài liệu bạn vừa upload:")
        if query:
            with st.spinner("Đang trả lời..."):
                result = qa_chain.run(query)
                st.write(result)
else:
    st.info("Hãy upload một file PDF để bắt đầu.")
