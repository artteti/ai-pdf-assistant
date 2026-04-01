import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – dark mode + gold accents ─────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E8E8E8;
    }

    /* ── Background layers ── */
    .stApp {
        background: #0D0D0D;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #141414;
        border-right: 1px solid #2A2A2A;
    }
    [data-testid="stSidebar"] *:not(button) {
        color: #E8E8E8 !important;
    }

    /* ── Header ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 28px 0 20px 0;
        border-bottom: 1px solid #2A2A2A;
        margin-bottom: 28px;
    }
    .app-header .icon {
        font-size: 2.4rem;
        line-height: 1;
    }
    .app-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #D4AF37 !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header .subtitle {
        font-size: 0.78rem;
        color: #666 !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #1A1A1A;
        border: 1px solid #2A2A2A;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 20px;
    }
    .status-badge .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #444;
    }
    .status-badge.active .dot {
        background: #D4AF37;
        box-shadow: 0 0 6px #D4AF3780;
    }
    .status-badge.active {
        border-color: #D4AF3740;
        color: #D4AF37;
    }

    /* ── Chat messages ── */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 8px 0;
    }
    .chat-message {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        animation: fadeIn 0.25s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-message .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .chat-message.user .avatar {
        background: #1E1E1E;
        border: 1px solid #333;
    }
    .chat-message.assistant .avatar {
        background: #1A160A;
        border: 1px solid #D4AF3740;
        color: #D4AF37;
    }
    .chat-message .bubble {
        background: #161616;
        border: 1px solid #242424;
        border-radius: 12px;
        padding: 14px 18px;
        max-width: 820px;
        font-size: 0.92rem;
        line-height: 1.65;
        color: #DCDCDC;
    }
    .chat-message.user .bubble {
        background: #111;
        border-color: #2A2A2A;
        color: #BBBBBB;
    }
    .chat-message.assistant .bubble {
        border-color: #D4AF3722;
    }
    .role-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 5px;
        color: #555;
    }
    .chat-message.assistant .role-label {
        color: #D4AF3799;
    }

    /* ── Welcome card ── */
    .welcome-card {
        background: #111;
        border: 1px solid #D4AF3722;
        border-radius: 16px;
        padding: 36px 40px;
        text-align: center;
        max-width: 520px;
        margin: 60px auto;
    }
    .welcome-card .icon-big {
        font-size: 3rem;
        margin-bottom: 16px;
    }
    .welcome-card h2 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #D4AF37 !important;
        margin-bottom: 10px;
    }
    .welcome-card p {
        font-size: 0.88rem;
        color: #666;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Sidebar upload zone ── */
    .upload-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #D4AF37 !important;
        margin-bottom: 10px;
        display: block;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid #2A2A2A;
        margin: 20px 0;
    }
    .sidebar-hint {
        font-size: 0.75rem;
        color: #999 !important;
        line-height: 1.55;
    }

    /* ── Streamlit overrides ── */
    .stFileUploader > div {
        background: #1A1A1A !important;
        border: 1px dashed #555 !important;
        border-radius: 10px !important;
    }
    .stFileUploader > div:hover {
        border-color: #D4AF3766 !important;
    }
    /* File uploader — all inner text white */
    .stFileUploader label,
    .stFileUploader span,
    .stFileUploader p,
    .stFileUploader small,
    [data-testid="stFileUploaderDropzone"] * {
        color: #E0E0E0 !important;
    }
    /* Browse-files button inside uploader */
    [data-testid="stFileUploaderDropzone"] button {
        background: #2A2A2A !important;
        color: #E0E0E0 !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #D4AF37 !important;
        color: #D4AF37 !important;
    }
    .stTextInput > div > div > input {
        background: #111 !important;
        border: 1px solid #2A2A2A !important;
        color: #E8E8E8 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #D4AF37 !important;
        box-shadow: 0 0 0 2px #D4AF3722 !important;
    }
    .stButton > button {
        background: #D4AF37 !important;
        color: #0D0D0D !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: opacity 0.15s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }
    .stSpinner > div {
        border-top-color: #D4AF37 !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #E8E8E8;
    }
    /* Remove default Streamlit padding top */
    .block-container {
        padding-top: 1.5rem !important;
    }
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #111; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #D4AF37; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "qa_chain": None,
        "pdf_name": None,
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        st.error("OPENAI_API_KEY not found. Add it to your .env file.")
        st.stop()
    return key


def process_pdf(uploaded_file) -> None:
    """Load, split, embed and store the PDF."""
    api_key = get_api_key()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        vector_store = Chroma.from_documents(chunks, embeddings)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3,
        )

        prompt_template = """You are an expert AI PDF Assistant. Use the provided context from the document to answer the question thoroughly and accurately. If the answer is not in the context, clearly state that the information is not available in the document.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )

        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history = []

    finally:
        os.unlink(tmp_path)


def ask_question(question: str) -> str:
    """Query the QA chain and return the answer."""
    if st.session_state.qa_chain is None:
        return "Please upload a PDF document first."
    result = st.session_state.qa_chain.invoke({"query": question})
    return result.get("result", "No answer returned.")


def render_message(role: str, content: str) -> None:
    """Render a single chat bubble."""
    is_assistant = role == "assistant"
    avatar = "🔍" if is_assistant else "👤"
    label = "AI Assistant" if is_assistant else "You"
    css_class = "assistant" if is_assistant else "user"

    st.markdown(
        f"""
        <div class="chat-message {css_class}">
            <div class="avatar">{avatar}</div>
            <div class="bubble">
                <div class="role-label">{label}</div>
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 20px 0 8px 0;">
            <span style="font-size:1.6rem;">🔍</span>
            <span style="font-size:1.1rem; font-weight:700; color:#D4AF37; margin-left:10px;">
                AI PDF Assistant
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<span class="upload-label">📄 Document</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing document…"):
                try:
                    process_pdf(uploaded_file)
                    st.success("Document ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if st.session_state.pdf_name:
        st.markdown(
            f"""
            <div style="background:#1A160A; border:1px solid #D4AF3730; border-radius:10px;
                        padding:12px 14px; margin-bottom:14px;">
                <div style="font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
                            text-transform:uppercase; color:#D4AF37; margin-bottom:4px;">
                    Active Document
                </div>
                <div style="font-size:0.82rem; color:#CCC; word-break:break-word;">
                    📄 {st.session_state.pdf_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Clear Document", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="sidebar-hint">
            Upload a PDF to enable intelligent Q&amp;A powered by GPT-4o Mini
            and Chroma vector search.<br><br>
            Chunks: 1 000 chars · Overlap: 200 chars · Top-K: 5 · DB: Chroma
        </p>
        """,
        unsafe_allow_html=True,
    )


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <div class="icon">🔍</div>
        <div>
            <h1>AI PDF Assistant</h1>
            <div class="subtitle">Powered by GPT-4o Mini · Chroma · LangChain</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status badge
if st.session_state.pdf_name:
    st.markdown(
        f'<div class="status-badge active"><div class="dot"></div>'
        f'Document loaded — {st.session_state.pdf_name}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-badge"><div class="dot"></div>No document loaded</div>',
        unsafe_allow_html=True,
    )

# Chat history display
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        render_message(msg["role"], msg["content"])
    st.markdown("</div>", unsafe_allow_html=True)
else:
    if not st.session_state.pdf_name:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="icon-big">🔍</div>
                <h2>Welcome to AI PDF Assistant</h2>
                <p>Upload a PDF document using the sidebar to start an intelligent conversation
                about its contents. Ask questions, extract insights, and explore your documents
                with the power of GPT-4o Mini.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="icon-big">💬</div>
                <h2>Document Ready</h2>
                <p>Your document has been processed and indexed. Ask any question below
                to begin exploring its contents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        label="Question",
        placeholder="Ask anything about your document…",
        label_visibility="collapsed",
        key="user_input",
        disabled=st.session_state.pdf_name is None,
    )

with col_btn:
    send = st.button(
        "Send",
        use_container_width=True,
        disabled=st.session_state.pdf_name is None,
    )

if send and user_input.strip():
    question = user_input.strip()

    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        try:
            answer = ask_question(question)
        except Exception as e:
            answer = f"An error occurred: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
