import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from smart_extract import check_and_extract_missing, force_extract_all
from prompts import get_default_qa_prompt
import shutil

load_dotenv()

DATA_DIR = Path(os.getenv('DATA_DIR', 'extracted'))
DATA_FILE = DATA_DIR / 'documents.txt'
CHROMA_DB_DIR = Path(os.getenv('CHROMA_DB_DIR', 'chroma_db'))
PDF_DIR = Path(os.getenv('PDF_DIR', 'data'))

st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")

# --- Sidebar configuration ---
st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input(
    "🔑 Google API Key",
    value=os.getenv('GOOGLE_API_KEY', ''),
    type="password",
    help="Get your API key from Google AI Studio",
    key="api_key_input"
)
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key

model_name = st.sidebar.selectbox("🤖 Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0, key="model_select")
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.1, 0.1, key="temp_slider")

# PDF Management & File Explorer
st.sidebar.markdown("---")
st.sidebar.subheader("📁 PDF Management")

pdf_files = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
txt_files = list(DATA_DIR.glob("*.txt")) if DATA_DIR.exists() else []

col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.metric("📄 PDFs", len(pdf_files))
with col2:
    st.sidebar.metric("📝 Extracted", len(txt_files))

if st.sidebar.button("🔍 Check & Extract Missing", key="check_extract_btn"):
    if api_key:
        with st.spinner("Checking for missing extractions..."):
            newly_extracted = check_and_extract_missing(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
            if newly_extracted:
                st.sidebar.info(f"📊 Extracted {len(newly_extracted)} new file(s)")
                if CHROMA_DB_DIR.exists():
                    shutil.rmtree(CHROMA_DB_DIR)
                st.session_state.qa_chain = None
            else:
                st.sidebar.info("All PDFs already extracted!")
    else:
        st.sidebar.error("Please provide API key first!")

if st.sidebar.button("🔄 Force Re-extract All", key="force_extract_btn"):
    if api_key:
        with st.spinner("Re-extracting all PDFs..."):
            force_extract_all(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
            st.sidebar.info("📊 All PDFs re-extracted successfully")
            if CHROMA_DB_DIR.exists():
                shutil.rmtree(CHROMA_DB_DIR)
            st.session_state.qa_chain = None
    else:
        st.sidebar.error("Please provide API key first!")

if pdf_files:
    with st.sidebar.expander(f"📄 View PDFs ({len(pdf_files)})", expanded=False):
        for pdf_file in pdf_files:
            file_size = pdf_file.stat().st_size / 1024
            st.write(f"• {pdf_file.name} ({file_size:.1f} KB)")
elif PDF_DIR.exists():
    st.sidebar.info("📄 No PDFs found in data folder")

if txt_files:
    with st.sidebar.expander(f"📝 View Extracted ({len(txt_files)})", expanded=False):
        for txt_file in txt_files:
            file_size = txt_file.stat().st_size / 1024
            if txt_file.name == "documents.txt":
                st.write(f"📋 {txt_file.name} ({file_size:.1f} KB) - Combined")
            else:
                st.write(f"• {txt_file.name} ({file_size:.1f} KB)")
elif DATA_DIR.exists():
    st.sidebar.info("📝 No extracted files found")

# Vector Database Management
st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Vector Database")

vector_status = "❌ Not accessible"
vector_count = 0

if not api_key:
    vector_status = "🔑 API key required"
elif not CHROMA_DB_DIR.exists():
    vector_status = "📁 Directory missing"
elif not DATA_FILE.exists():
    vector_status = "📄 No documents extracted"
else:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings
        )
        vector_count = vectorstore._collection.count()
        if vector_count > 0:
            vector_status = f"✅ Ready ({vector_count} vectors)"
        else:
            vector_status = "🔄 Needs rebuilding"
    except Exception as e:
        if "readonly" in str(e).lower() or "permission" in str(e).lower():
            vector_status = "⚠️ Permission issues"
        else:
            vector_status = f"❌ Error: {str(e)[:30]}..."

st.sidebar.text(f"Status: {vector_status}")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("🗑️ Clear", help="Clear vector database", key="clear_db_btn"):
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
            st.sidebar.success("Database cleared!")
            st.session_state.qa_chain = None
        else:
            st.sidebar.info("No database to clear")

with col2:
    if st.sidebar.button("🔧 Fix", help="Fix database permission issues", key="fix_db_btn"):
        try:
            if CHROMA_DB_DIR.exists():
                import stat
                try:
                    for root, dirs, files in os.walk(CHROMA_DB_DIR):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                        for f in files:
                            os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    st.sidebar.success("Permissions fixed!")
                except:
                    shutil.rmtree(CHROMA_DB_DIR)
                    st.sidebar.info("Cleared database for fresh start")
            else:
                CHROMA_DB_DIR.mkdir(exist_ok=True, mode=0o755)
                st.sidebar.success("Database directory created!")
            st.session_state.qa_chain = None
        except Exception as e:
            st.sidebar.error(f"Fix failed: {str(e)}")

if st.sidebar.button("🔄 Rebuild DB", help="Rebuild vector database from documents", key="rebuild_db_btn"):
    if api_key:
        if DATA_FILE.exists():
            try:
                if CHROMA_DB_DIR.exists():
                    shutil.rmtree(CHROMA_DB_DIR)
                st.session_state.qa_chain = None
                st.sidebar.success("Database will rebuild on next query!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Rebuild failed: {str(e)}")
        else:
            st.sidebar.error("No documents found to rebuild from")
    else:
        st.sidebar.error("API key required for rebuild")

# Session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource(show_spinner=False)
def load_qa_chain(api_key: str, model: str, temp: float):
    global CHROMA_DB_DIR  # Add global declaration
    if not api_key:
        raise ValueError("API key is required")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temp,
        convert_system_message_to_human=True
    )
    CHROMA_DB_DIR.mkdir(exist_ok=True)
    try:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings
        )
        if vectorstore._collection.count() == 0:
            if not DATA_FILE.exists():
                raise FileNotFoundError("No extracted text found. Upload PDFs and extract first.")
            loader = TextLoader(str(DATA_FILE), encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
    except Exception as e:
        st.warning(f"Database issue detected: {e}")
        st.info("Clearing and recreating vector database...")
        try:
            if CHROMA_DB_DIR.exists():
                import stat
                for root, dirs, files in os.walk(CHROMA_DB_DIR):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                        except:
                            pass
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                        except:
                            pass
                shutil.rmtree(CHROMA_DB_DIR)
        except Exception as cleanup_error:
            st.error(f"Could not clean directory: {cleanup_error}")
            import time
            # Use a local variable for alternative directory
            alt_chroma_dir = Path(f'chroma_db_{int(time.time())}')
            st.info(f"Using alternative directory: {alt_chroma_dir}")
            CHROMA_DB_DIR = alt_chroma_dir  # Update global variable
        CHROMA_DB_DIR.mkdir(exist_ok=True, mode=0o755)
        if not DATA_FILE.exists():
            raise FileNotFoundError("No extracted text found. Upload PDFs and extract first.")
        loader = TextLoader(str(DATA_FILE), encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        try:
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            st.success("Vector database recreated successfully!")
        except Exception as create_error:
            st.error(f"Failed to create new database: {create_error}")
            st.warning("Trying alternative database configuration...")
            try:
                import chromadb
                client = chromadb.PersistentClient(
                    path=str(CHROMA_DB_DIR),
                    settings=chromadb.config.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                collection = client.get_or_create_collection("documents")
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
                collection.add(
                    documents=[doc.page_content for doc in texts],
                    ids=ids
                )
                vectorstore = Chroma(
                    client=client,
                    collection_name="documents",
                    embedding_function=embeddings
                )
                st.success("Database created with alternative method!")
            except Exception as final_error:
                st.error(f"All database creation methods failed: {final_error}")
                st.warning("Using in-memory database (will not persist)")
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings
                )
    PROMPT = get_default_qa_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

# PDF upload section
with st.expander("➕ Upload PDFs & Extract", expanded=False):
    uploads = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    if uploads and st.button("Process PDFs"):
        DATA_DIR.mkdir(exist_ok=True)
        PDF_DIR.mkdir(exist_ok=True)
        saved_files = []
        for file in uploads:
            bytes_data = file.read()
            tmp_path = PDF_DIR / file.name
            with open(tmp_path, 'wb') as f:
                f.write(bytes_data)
            saved_files.append(tmp_path)
        st.success(f"✅ Saved {len(uploads)} PDF file(s)")
        with st.spinner("📄 Extracting text from PDFs..."):
            newly_extracted = check_and_extract_missing(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
            if newly_extracted:
                st.success("✅ **TEXT EXTRACTION COMPLETED!**")
                st.info(f"📊 **Extraction Summary:**\n"
                        f"- **Files processed:** {len(uploads)} PDF(s)\n"
                        f"- **New extractions:** {len(newly_extracted)} file(s)")
            else:
                st.success("✅ **PDF PROCESSED SUCCESSFULLY!**")
                st.info("ℹ️ All uploaded PDFs were already extracted previously")
        st.session_state.qa_chain = None
        st.rerun()

# Build / load QA chain
if st.session_state.qa_chain is None:
    try:
        with st.spinner("🔗 Loading Google Gemini and building knowledge base..."):
            st.session_state.qa_chain = load_qa_chain(api_key, model_name, temperature)
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

# Chat Interface
st.subheader("💬 Chat")
if st.session_state.qa_chain is None:
    st.info("System not ready yet. Please check the API key and ensure documents are available.")
else:
    for user, assistant, sources in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            st.markdown(assistant)
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {source}")

    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    sources = result["source_documents"]
                    st.markdown(answer)
                    source_texts = []
                    if sources:
                        for i, doc in enumerate(sources[:3], 1):
                            content_preview = doc.page_content[:200].replace('\n', ' ')
                            source_texts.append(f"**Source {i}:** {content_preview}...")
                    st.session_state.history.append((prompt, answer, source_texts))
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.history.append((prompt, error_msg, []))

# Sidebar utilities
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", key="clear_chat_btn"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("📚 **PDF Chatbot**")
    st.markdown("Built with **Streamlit**, **Google Gemini**, **LangChain**")
