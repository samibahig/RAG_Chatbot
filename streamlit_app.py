import os
import streamlit as st

# ============================================================
# IMPORTS — LangChain v0.2+ (packages non dépréciés)
# ============================================================
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RAG Support Chatbot",
    page_icon="💬",
    layout="centered"
)
st.title("💬 RAG Customer Support Chatbot")
st.caption("Powered by LangChain · OpenAI GPT-3.5 · ChromaDB (persistant)")

# ============================================================
# 2. API KEY
# ============================================================
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if not api_key:
    st.error(
        "Clé OpenAI introuvable.  \n"
        "Ajoutez-la dans les **Secrets** de Streamlit Cloud :  \n"
        "`OPENAI_API_KEY = \"sk-...\"`"
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ============================================================
# 3. CONSTANTES
# ============================================================
DATA_DIR       = "data"           # dossier contenant vos fichiers .txt
CHROMA_DIR     = "./chroma_db"    # dossier de persistance ChromaDB
COLLECTION     = "support_docs"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
TOP_K          = 3
MODEL_CHAT     = "gpt-3.5-turbo"
MODEL_EMBED    = "text-embedding-3-small"
TEMPERATURE    = 0

# ============================================================
# 4. CHARGEMENT & INDEXATION (cached + persistant)
# ============================================================
@st.cache_resource(show_spinner="Chargement de la base de connaissances...")
def load_vectorstore() -> Chroma:
    """Charge ou recrée l'index ChromaDB persistant."""
    embeddings = OpenAIEmbeddings(model=MODEL_EMBED, openai_api_key=api_key)

    # Si l'index existe déjà sur disque, on le charge directement
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )

    # Sinon, on construit l'index depuis les documents
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.error(
            f"Dossier `{DATA_DIR}/` introuvable ou vide.  \n"
            "Placez vos fichiers `.txt` dans ce dossier."
        )
        st.stop()

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR      # ← persistance sur disque
    )
    return vectorstore


@st.cache_resource(show_spinner="Initialisation du chatbot...")
def load_rag_chain() -> ConversationalRetrievalChain:
    """Crée la chaîne RAG avec mémoire conversationnelle."""
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    llm = ChatOpenAI(
        model_name=MODEL_CHAT,
        temperature=TEMPERATURE,
        openai_api_key=api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain


rag_chain = load_rag_chain()

# ============================================================
# 5. SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# 6. BOUTON RESET
# ============================================================
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🗑️ Réinitialiser la conversation"):
        st.session_state.messages = []
        # Vider la mémoire de la chaîne
        rag_chain.memory.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Modèle : `{MODEL_CHAT}`")
    st.caption(f"Embeddings : `{MODEL_EMBED}`")
    st.caption(f"Top-K sources : `{TOP_K}`")

# ============================================================
# 7. INTERFACE CHAT
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Posez votre question...")

if user_input:
    # Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Générer et afficher la réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                response = rag_chain.invoke({"question": user_input})
                answer   = response["answer"]
                sources  = response.get("source_documents", [])
            except Exception as e:
                answer  = f"❌ Une erreur est survenue : `{e}`"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander(f"📄 {len(sources)} source(s) utilisée(s)"):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get("source", "inconnue")
                    st.markdown(f"**Source {i} :** `{source_name}`")
                    st.caption(doc.page_content[:300] + "…")

    st.session_state.messages.append({"role": "assistant", "content": answer})