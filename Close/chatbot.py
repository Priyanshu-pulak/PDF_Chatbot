import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from smart_extract import check_and_extract_missing
from prompts import get_default_qa_prompt
import shutil

load_dotenv()

DATA_FILE = Path('extracted', 'documents.txt')
CHROMA_DB_DIR = Path('chroma_db')
PERSIST_DIR = Path('data')

def setup_environment():
    """Setup API key and check requirements."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your Google API key:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        exit(1)
    
    print("Google API key found!")
    return api_key

def initialize_components(api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    return embeddings, llm

def load_or_create_vectorstore(embeddings, force_rebuild=False):
    CHROMA_DB_DIR.mkdir(exist_ok=True)
    
    # Check if we need to rebuild
    if force_rebuild and CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)
        CHROMA_DB_DIR.mkdir(exist_ok=True)
    
    # Check if vectorstore exists and has data
    try:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings
        )
        
        # Check if it has any documents
        if vectorstore._collection.count() > 0 and not force_rebuild:
            print(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
            return vectorstore
        else:
            print("Creating new vectorstore from documents...")
            return create_vectorstore(embeddings)
            
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        print("Creating new vectorstore...")
        return create_vectorstore(embeddings)

def create_vectorstore(embeddings):
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"No documents found at {DATA_FILE}. Please run smart extraction first.")
    
    print(f"Loading documents from {DATA_FILE}...")
    loader = TextLoader(str(DATA_FILE), encoding='utf-8')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks")
    
    print("Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    
    print(f"Vectorstore created with {len(texts)} documents")
    return vectorstore

def create_qa_chain(vectorstore, llm):
    PROMPT = get_default_qa_prompt()
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    print("Starting LangChain + Google Gemini PDF Chatbot")
    print("=" * 50)
    
    api_key = setup_environment()
    
    print("\nChecking for new PDFs that need extraction...")
    newly_extracted = check_and_extract_missing()
    
    force_rebuild = len(newly_extracted) > 0
    if newly_extracted:
        print(f"\nExtracted {len(newly_extracted)} new file(s). Will rebuild vector database...")
    
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Please ensure PDFs are in the data/ folder.")
        exit(1)
    
    # Initialize components
    embeddings, llm = initialize_components(api_key)
    
    # Load or create vectorstore
    print("\nSetting up vector database...")
    try:
        vectorstore = load_or_create_vectorstore(embeddings, force_rebuild)
    except Exception as e:
        print(f"❌ Error creating vectorstore: {e}")
        exit(1)
    
    print("\nCreating question-answering chain...")
    qa_chain = create_qa_chain(vectorstore, llm)
    
    # Interactive chat loop
    print("\n" + "=" * 60)
    print("Chatbot ready! Ask questions about your documents.")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\n🤔 You: ")
            
            if question.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break
            
            if not question.strip():
                continue
            
            print("\n🤖 AI: ", end="", flush=True)
            
            # Get response
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            
            print(answer)
            
            # Show sources if available
            if sources:
                print(f"\n📚 Sources ({len(sources)} found):")
                for i, doc in enumerate(sources[:3], 1):  # Show top 3 sources
                    content_preview = doc.page_content[:150].replace('\n', ' ')
                    print(f"  {i}. {content_preview}...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
