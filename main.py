from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from streamlit_chat import message  

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH_API')

UPLOAD_DIR = "uploaded_files"

def cleanup_files():
    if os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    if 'file_handle' in st.session_state:
        st.session_state.file_handle.close()

if 'cleanup_done' not in st.session_state:
    st.session_state.cleanup_done = False

if not st.session_state.cleanup_done:
    cleanup_files()

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

st.title("Chat with Your PDF!!")
uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    file_path = os.path.abspath(file_path)

    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write("You're Ready For a Chat with your PDF")

    docs = PyPDFLoader(file_path).load_and_split()

    embedding = HuggingFaceBgeEmbeddings(
        model_name='BAAI/llm-embedder',
    )

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding, store, namespace='embeddings'
    )

    vector_base = FAISS.from_documents(
        docs,
        embedding
    )

    template = '''You are an Experienced Business Person Having a 
                great Knowledge About Various Types of Business Activities
                but here you are hired to Give the answers to the {question} only based on {context}
                .if you are unaware of the question Just reply it with I\'m Unaware of your Query.
                Use three sentences maximum and keep the answer concise.'''

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector_base.as_retriever()

    llm = ChatGroq(
        model='mixtral-8x7b-32768',
        temperature=0,
    )

    if 'history' not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Enter your question")

    if st.button("Submit !"):
        if query:
            chain = (
                {'context': retriever, 'question': RunnablePassthrough()} 
                | prompt | llm | StrOutputParser()
            )
            answer = chain.invoke(query)
            st.session_state.history.append({'question': query, 'answer': answer})

    if st.session_state.history:
        st.write("### Previous Questions and Answers")
        for idx, entry in enumerate(st.session_state.history):
            message(f"**Q{idx + 1}:** {entry['question']}", is_table=True, key=f"question_{idx}",avatar_style='no-avatar')
            message(f"**A{idx + 1}:** {entry['answer']}", is_table=True, key=f"answer_{idx}",avatar_style='no-avatar')


if st.session_state.cleanup_done:
    cleanup_files()
