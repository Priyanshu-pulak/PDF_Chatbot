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

DATA_DIR = Path('extracted')
DATA_FILE = DATA_DIR / 'documents.txt'
CHROMA_DB_DIR = Path('chroma_db')
PDF_DIR = Path('data')

def ensure_writable_chroma_dir():
    """Ensure we have a writable directory for ChromaDB"""
    global CHROMA_DB_DIR
    
    # Test if current directory is writable
    test_dirs = [CHROMA_DB_DIR, Path('chroma_db_alt'), Path('vector_db')]
    
    for test_dir in test_dirs:
        try:
            test_dir.mkdir(exist_ok=True, mode=0o755)
            # Test write permissions
            test_file = test_dir / 'test_write.txt'
            test_file.write_text('test')
            test_file.unlink()  # Delete test file
            CHROMA_DB_DIR = test_dir
            return test_dir
        except Exception:
            continue
    
    # If all fail, use a temporary directory
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix='chroma_db_'))
    CHROMA_DB_DIR = temp_dir
    return temp_dir

st.set_page_config(page_title="PDF Chatbot (Online)", page_icon="🌐", layout="wide")

# --- Sidebar configuration ---
st.sidebar.title("⚙️ Settings")

# API Key input
api_key = st.sidebar.text_input(
    "🔑 Google API Key", 
    value=os.getenv('GOOGLE_API_KEY', ''),
    type="password",
    help="Get your API key from Google AI Studio",
    key="api_key_input"
)

# Add button to get API key
if st.sidebar.button("🔗 Get API Key from Google AI Studio", key="get_api_key_btn"):
    st.sidebar.markdown("[Click here to get your API key](https://makersuite.google.com/app/apikey)")

if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key

# Model selection
model_name = st.sidebar.selectbox("🤖 Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0, key="model_select")
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.1, 0.1, key="temp_slider")

# PDF Management & File Explorer
st.sidebar.markdown("---")
st.sidebar.subheader("� PDF Management")

# File status overview
pdf_files = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
txt_files = list(DATA_DIR.glob("*.txt")) if DATA_DIR.exists() else []

col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.metric("📄 PDFs", len(pdf_files))
with col2:
    st.sidebar.metric("📝 Extracted", len(txt_files))

# Management actions
if st.sidebar.button("🔍 Check & Extract Missing", key="check_extract_btn"):
    if api_key:
        with st.spinner("Checking for missing extractions..."):
            newly_extracted = check_and_extract_missing(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
            if newly_extracted:
                st.sidebar.info(f"📊 Extracted {len(newly_extracted)} new file(s)")
                # Clear vector database to force rebuild
                if CHROMA_DB_DIR.exists():
                    shutil.rmtree(CHROMA_DB_DIR)
                st.session_state.qa_chain = None  # Force rebuild
            else:
                st.sidebar.info("All PDFs already extracted!")
    else:
        st.sidebar.error("Please provide API key first!")

if st.sidebar.button("🔄 Force Re-extract All", key="force_extract_btn"):
    if api_key:
        with st.spinner("Re-extracting all PDFs..."):
            force_extract_all(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
            st.sidebar.info("📊 All PDFs re-extracted successfully")
            # Clear vector database to force rebuild
            if CHROMA_DB_DIR.exists():
                shutil.rmtree(CHROMA_DB_DIR)
            st.session_state.qa_chain = None  # Force rebuild
    else:
        st.sidebar.error("Please provide API key first!")

# File browser
if pdf_files:
    with st.sidebar.expander(f"� View PDFs ({len(pdf_files)})", expanded=False):
        for pdf_file in pdf_files:
            file_size = pdf_file.stat().st_size / 1024  # Size in KB
            st.write(f"• {pdf_file.name} ({file_size:.1f} KB)")
elif PDF_DIR.exists():
    st.sidebar.info("📄 No PDFs found in data folder")

if txt_files:
    with st.sidebar.expander(f"📝 View Extracted ({len(txt_files)})", expanded=False):
        for txt_file in txt_files:
            file_size = txt_file.stat().st_size / 1024  # Size in KB
            if txt_file.name == "documents.txt":
                st.write(f"📋 {txt_file.name} ({file_size:.1f} KB) - Combined")
            else:
                st.write(f"• {txt_file.name} ({file_size:.1f} KB)")
elif DATA_DIR.exists():
    st.sidebar.info("📝 No extracted files found")

# Vector Database Management
st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Vector Database")

# Show vector database stats
vector_status = "❌ Not accessible"
vector_count = 0
if CHROMA_DB_DIR.exists() and api_key:
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
        vector_status = f"✅ Ready ({vector_count} vectors)"
    except:
        vector_status = "⚠️ Permission issues"

st.sidebar.text(f"Status: {vector_status}")

# Database management buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("�️ Clear", help="Clear vector database", key="clear_db_btn"):
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
            st.sidebar.success("Database cleared!")
            st.session_state.qa_chain = None

with col2:
    if st.sidebar.button("🔧 Fix", help="Fix database permission issues", key="fix_db_btn"):
        if CHROMA_DB_DIR.exists():
            # Change permissions and recreate
            import stat
            try:
                # Try to change permissions recursively
                for root, dirs, files in os.walk(CHROMA_DB_DIR):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    for f in files:
                        os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                st.sidebar.success("Permissions fixed!")
            except:
                # If permission fix fails, clear and recreate
                shutil.rmtree(CHROMA_DB_DIR)
                st.sidebar.info("Cleared database for fresh start")
            st.session_state.qa_chain = None

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'history' not in st.session_state:
    st.session_state.history = []  # list of (user, assistant, sources)

# Helper functions
@st.cache_resource(show_spinner=False)
def load_qa_chain(api_key: str, model: str, temp: float):
    """Load or create the QA chain with caching."""
    if not api_key:
        raise ValueError("API key is required")
    
    # Initialize components
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
    
    # Load or create vectorstore
    CHROMA_DB_DIR = ensure_writable_chroma_dir()
    
    try:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings
        )
        
        # Check if it has any documents
        if vectorstore._collection.count() == 0:
            if not DATA_FILE.exists():
                raise FileNotFoundError("No extracted text found. Upload PDFs and extract first.")
            
            # Load and process documents
            loader = TextLoader(str(DATA_FILE), encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_documents(documents)
            
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            
    except Exception as e:
        # If there's a database error (like readonly), clear and recreate
        st.warning(f"Database issue detected: {e}")
        st.info("Clearing and recreating vector database...")
        
        # More aggressive cleanup - remove and recreate directory
        try:
            if CHROMA_DB_DIR.exists():
                import stat
                # First try to change permissions to allow deletion
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
            # Try using a different directory name
            import time
            CHROMA_DB_DIR = Path(f'chroma_db_{int(time.time())}')
            st.info(f"Using alternative directory: {CHROMA_DB_DIR}")
        
        # Recreate directory with proper permissions
        CHROMA_DB_DIR.mkdir(exist_ok=True, mode=0o755)
        
        # Check if we have documents to recreate the database
        if not DATA_FILE.exists():
            raise FileNotFoundError("No extracted text found. Upload PDFs and extract first.")
        
        # Load and process documents
        loader = TextLoader(str(DATA_FILE), encoding='utf-8')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        try:
            # Create new vectorstore with fresh directory
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            st.success("Vector database recreated successfully!")
        except Exception as create_error:
            st.error(f"Failed to create new database: {create_error}")
            # As a last resort, try with different client settings
            st.warning("Trying alternative database configuration...")
            try:
                import chromadb
                # Try creating a client with specific settings
                client = chromadb.PersistentClient(
                    path=str(CHROMA_DB_DIR),
                    settings=chromadb.config.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                collection = client.get_or_create_collection("documents")
                
                # Add documents to collection
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
                collection.add(
                    documents=[doc.page_content for doc in texts],
                    ids=ids
                )
                
                # Create vectorstore from existing collection
                vectorstore = Chroma(
                    client=client,
                    collection_name="documents",
                    embedding_function=embeddings
                )
                st.success("Database created with alternative method!")
            except Exception as final_error:
                st.error(f"All database creation methods failed: {final_error}")
                # Absolute fallback - in-memory only
                st.warning("Using in-memory database (will not persist)")
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings
                )
    
    # Create QA chain using prompt from prompts module
    PROMPT = get_default_qa_prompt()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def extract_text_from_uploads(uploaded_files):
    """Handle PDF uploads and save to data directory."""
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    
    saved_files = []
    for file in uploaded_files:
        bytes_data = file.read()
        # Save to data directory
        tmp_path = PDF_DIR / file.name
        with open(tmp_path, 'wb') as f:
            f.write(bytes_data)
        saved_files.append(tmp_path)
        print(f"Saved: {tmp_path}")  # Debug print
    
    # Clear old vector database so it'll rebuild
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)
        CHROMA_DB_DIR.mkdir(exist_ok=True)
    
    return saved_files

# Top layout
st.title("🌐 PDF Chatbot (Online - Google Gemini)")
st.caption("Chat with your PDFs using Google Gemini API + LangChain")

# API Key check
if not api_key:
    st.warning("🔑 Please enter your Google API Key in the sidebar to start chatting!")
    st.stop()

# Auto-check for missing extractions on app startup
if 'auto_check_done' not in st.session_state:
    with st.spinner("🔍 Checking for PDFs that need extraction..."):
        newly_extracted = check_and_extract_missing(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
        if newly_extracted:
            st.info(f"📊 Found and extracted {len(newly_extracted)} missing file(s) on startup")
            # Clear vector database to force rebuild
            if CHROMA_DB_DIR.exists():
                shutil.rmtree(CHROMA_DB_DIR)
            st.session_state.qa_chain = None  # Force rebuild
        st.session_state.auto_check_done = True

with st.expander("➕ Upload PDFs & Extract", expanded=False):
    uploads = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    if uploads and st.button("Process PDFs"):
        with st.spinner("📁 Saving uploaded PDFs..."):
            saved_files = extract_text_from_uploads(uploads)
            st.success(f"✅ Saved {len(uploads)} PDF file(s) to data folder")
            
            # Show saved files for debugging
            st.write("**Saved files:**")
            for saved_file in saved_files:
                st.write(f"- {saved_file}")
        
        with st.spinner("📄 Extracting text from PDFs..."):
            # After upload, check for any missing extractions
            newly_extracted = check_and_extract_missing(pdf_dir=PDF_DIR, output_dir=DATA_DIR)
            
            if newly_extracted:
                # Alert-style success message
                st.success("✅ **TEXT EXTRACTION COMPLETED!**")
                
                st.info(f"📊 **Extraction Summary:**\n"
                        f"- **Files processed:** {len(uploads)} PDF(s)\n"
                        f"- **New extractions:** {len(newly_extracted)} file(s)\n"
                        f"- **Output location:** `extracted/` folder")
                
                # Show which files were extracted
                with st.expander("📋 View Extracted Files", expanded=False):
                    for extracted_file in newly_extracted:
                        st.write(f"✓ {extracted_file.name}")
                
                st.session_state.qa_chain = None  # Force rebuild
            else:
                # Alert-style message for already processed files
                st.success("✅ **PDF PROCESSED SUCCESSFULLY!**")
                st.info("ℹ️ All uploaded PDFs were already extracted previously")
        
        # Set flag to show ready message after rerun
        st.session_state.show_ready_message = True
        st.rerun()

# Build / load QA chain
if st.session_state.qa_chain is None:
    try:
        with st.spinner("🔗 Loading Google Gemini and building knowledge base..."):
            st.session_state.qa_chain = load_qa_chain(api_key, model_name, temperature)
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

# Show ready message if flag is set (after rerun from PDF processing)
if st.session_state.get("show_ready_message", False):
    st.success("🎯 **PDF processing complete!**")
    st.success("🔄 **SYSTEM READY FOR CHATTING!**")
    st.info("💬 You can now ask questions about your documents.")
    st.session_state.show_ready_message = False  # Reset the flag

# Chat Interface
st.subheader("💬 Chat")
if st.session_state.qa_chain is None:
    st.info("System not ready yet. Please check the API key and ensure documents are available.")
else:
    # Display chat history
    for user, assistant, sources in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            st.markdown(assistant)
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {source}")

    # Chat input
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
                    
                    # Prepare sources for storage
                    source_texts = []
                    if sources:
                        for i, doc in enumerate(sources[:3], 1):
                            content_preview = doc.page_content[:200].replace('\n', ' ')
                            source_texts.append(f"**Source {i}:** {content_preview}...")
                    
                    # Store in history
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
    st.markdown("🌐 **Online Mode**")
    st.markdown("Built with **Streamlit**, **Google Gemini**, **LangChain**")
