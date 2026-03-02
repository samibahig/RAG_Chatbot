import os
import streamlit as st

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============================================================
# 1. PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="RAG Support Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 RAG Customer Support Chatbot")
st.caption("Powered by LangChain · OpenAI GPT-3.5 · ChromaDB")

# ============================================================
# 2. API KEY — works with Streamlit Cloud secrets OR env var
# ============================================================
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

if not api_key:
    st.error(
        "OpenAI API key not found.  \n"
        "Add it to your app's **Secrets** on Streamlit Cloud:  \n"
        "`OPENAI_API_KEY = \"sk-...\"`"
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ============================================================
# 3. LOAD & INDEX DOCUMENTS (cached — runs only once)
# ============================================================
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_rag_chain():
    doc_path = "data/support_doc.txt"

    if not os.path.exists(doc_path):
        st.error(
            f"Document not found: `{doc_path}`  \n"
            "Make sure your repo contains a `data/support_doc.txt` file."
        )
        st.stop()

    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        collection_name="support_docs"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=api_key)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return rag_chain

rag_chain = load_rag_chain()

# ============================================================
# 4. CHAT INTERFACE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a support question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = rag_chain.invoke({"query": user_input})
            answer = response["result"]
            sources = response.get("source_documents", [])

        st.markdown(answer)

        if sources:
            with st.expander("Sources used"):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Source {i}:** `{source_name}`")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
