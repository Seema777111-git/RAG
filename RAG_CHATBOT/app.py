import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Environment & API Key Verification
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="HR Policy Assistant", page_icon="💼")
st.title("💼 HR Policy Intelligent Assistant")

if not api_key:
    st.error("❌ `OPENAI_API_KEY` missing from your `.env` file. Please configure it to continue.")
    st.stop()

def clean_text(text: str) -> str:
    """Normalize excessive whitespace and isolated line breaks."""
    return re.sub(r"\s+", " ", text).strip()

def format_docs(docs):
    """Concatenate retrieved context chunks."""
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Document Upload & Processing Section
st.subheader("1. Upload Policy Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
process_button = st.button("Process & Index PDF", type="primary")

if process_button and uploaded_file is not None:
    with st.spinner("Processing PDF and creating vector index..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Document Loading
            loader = PyPDFLoader(tmp_path)
            raw_pages = loader.load()

            # Text Cleaning
            for doc in raw_pages:
                doc.page_content = clean_text(doc.page_content)

            # Text Splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=[". ", "\n", " "]
            )
            chunks = splitter.split_documents(raw_pages)

            # Embeddings & FAISS Vector Store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Persist in Streamlit Session State
            st.session_state.vector_store = vector_store
            st.session_state.doc_name = uploaded_file.name
            st.session_state.total_chunks = len(chunks)

            st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

elif process_button and uploaded_file is None:
    st.warning("Please select a PDF file first before clicking Process.")

# 3. Question Answering Section
st.divider()
st.subheader("2. Ask a Question")

if "vector_store" in st.session_state:
    st.info(f"Active document: **{st.session_state.doc_name}** ({st.session_state.total_chunks} chunks ready)")

    with st.form("qa_form"):
        user_question = st.text_input(
            "Ask about company policies:",
            placeholder="e.g., How many days of annual leave do I get?"
        )
        submit_button = st.form_submit_button("Get Answer", type="primary")

    if submit_button and user_question:
        # LangChain Semantic Retrieval
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(user_question)

        # Prompt & OpenAI Model
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful HR policy assistant.
Answer the user's question using the policy excerpts below. If the answer cannot be found in the context, say: "I cannot find this information in the uploaded policy."

Context:
{context}

Question:
{question}

Answer:"""
        )

        rag_chain = (
            {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        with st.spinner("Analyzing policy..."):
            response = rag_chain.invoke(user_question)

        # Clear Answer Display
        st.subheader("💡 Answer:")
        st.write(response)

        # Grounding / Verification
        with st.expander("🔍 View Retrieved Sources / Page References"):
            for i, doc in enumerate(retrieved_docs, start=1):
                page_num = doc.metadata.get("page", 0) + 1
                st.markdown(f"**Reference {i} (Page {page_num})**")
                st.write(doc.page_content)
                st.divider()
else:
    st.write("Upload a PDF and click **Process & Index PDF** above to begin asking questions.")