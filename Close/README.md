# 🌐 PDF Chatbot using Google Gemini API (LangChain + Online)

This project allows you to chat with one or more PDF documents using **Google Gemini API** and **LangChain**. It supports **text-based** and **image-based (scanned)** PDFs by applying OCR automatically. It builds a searchable vector database using **ChromaDB** and answers your questions using Google's powerful Gemini models.

---

## 📁 Project Structure
```
Close/
├── chatbot.py              # Command-line chatbot using LangChain & Gemini
├── app.py                  # Streamlit web interface
├── smart_extract.py        # Smart PDF extraction (new files only)
├── environment.yml         # Conda environment with all dependencies
├── .env.example            # Template for API key configuration
├── data/                   # Put your PDF files here
├── extracted/documents.txt # Combined extracted text for LangChain
├── extracted/             # Individual extracted text files
└── chroma_db/             # ChromaDB vector database storage
```

---

## ✅ Features

- 🔍 Supports both **text** and **scanned image** PDFs  
- 🌐 Uses **Google Gemini API** (gemini-pro, gemini-pro-latest)
- 🔗 Built with **LangChain** framework for robust AI workflows
- 🗄️ **ChromaDB vector database** for efficient similarity search
- 🔑 **API key management** with environment variables
- 🚀 Smart extraction - only processes new/missing PDFs
- 📊 **Real-time vector count** and database management
- 💬 **Source citations** in responses
- 🎯 **Customizable prompts** and temperature settings

---

## ⚙️ Setup Instructions

### 1. Get Google API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in the next steps

### 2. Install Anaconda or Miniconda  
Download and install from the official site.

### 3. Create & activate the conda environment
```bash
conda env create -f environment.yml
conda activate close_env
```

### 4. Configure API Key
Choose one of these methods:

#### Method 1: Environment File (Recommended)
```bash
cp .env.example .env
# Edit .env file and add your API key:
# GOOGLE_API_KEY=your_actual_api_key_here
```

#### Method 2: System Environment Variable
```bash
export GOOGLE_API_KEY=your_actual_api_key_here
```

#### Method 3: Streamlit Interface
- Run the Streamlit app and enter the API key in the sidebar

---

## 📝 Processing PDFs

### Smart Extraction (Recommended)
The system automatically checks which PDFs need extraction:
```bash
python smart_extract.py
```
This will:
- ✅ Check each PDF in `data/` folder
- ✅ Extract only missing files (not already in `extracted/`)
- ✅ Update the combined `documents.txt` file
- ✅ Show progress for each file

---

## 🚀 Usage

### 🖥️ Command-Line Chatbot
```bash
python chatbot.py
```
Features:
- **🔍 Auto-detection**: Automatically finds and extracts new PDFs
- **🤖 Smart responses**: Uses Gemini-pro model
- **📚 Source citations**: Shows relevant document excerpts
- **💬 Interactive chat**: Type questions, get instant answers

### 🌐 Streamlit Web App
```bash
streamlit run app.py
```
Features:
- **🔑 API key input**: Enter your key in the sidebar
- **📤 Upload PDFs**: Add new PDFs from the browser
- **🔄 Smart extraction**: Only processes missing files
- **🏗️ Vector database management**: Clear, rebuild, view stats
- **💬 Chat interface**: Web-based conversation with source citations
- **🎛️ Model controls**: Select model, adjust temperature

---

## 🔄 Rebuild the Vector Database
```bash
rm -r chroma_db/
python chatbot.py
```
Or use the "🗑️ Clear Vector DB" button in the Streamlit app.

---

## 🛠 Command Cheat‑Sheet
| Purpose                      | Command                                    |
|------------------------------|--------------------------------------------|
| Create env                   | `conda env create -f environment.yml`      |
| Activate env                 | `conda activate close_env`                |
| Smart extract PDFs           | `python smart_extract.py`                 |
| Start chatbot                | `python chatbot.py`                        |
| Start web app                | `streamlit run app.py`                     |
| Clear vector database        | `rm -r chroma_db/`                         |

---

## 📦 Dependencies Explained

### Core LangChain Stack:
- **langchain**: Core framework for AI workflows
- **langchain-google-genai**: Google Gemini API integration
- **langchain-community**: Community document loaders
- **langchain-chroma**: ChromaDB vector store integration

### Google AI:
- **google-generativeai**: Official Google Gemini Python client

### Vector Database:
- **chromadb**: Vector database for similarity search

### PDF Processing:
- **pdfplumber**: PDF text extraction
- **pytesseract**: OCR for scanned documents
- **pdf2image**: Convert PDF pages to images
- **pillow**: Image processing

### Utilities:
- **python-dotenv**: Environment variable management
- **streamlit**: Web interface framework

---

## 🆚 Comparison: Open vs Close

| Feature | Open (Local) | Close (Online) |
|---------|-------------|----------------|
| **LLM** | Ollama (mistral/llama3) | Google Gemini API |
| **Framework** | LlamaIndex | LangChain |
| **Cost** | Free (local compute) | Pay-per-use API |
| **Internet** | Not required | Required |
| **Setup** | Install Ollama + models | Get API key |
| **Performance** | Depends on hardware | Consistent (cloud) |
| **Privacy** | Fully private | Data sent to Google |
| **Models** | Limited to downloaded | Latest Gemini models |

---

## ⚠️ Troubleshooting

### API Key Issues:
- **Invalid API key**: Verify key from Google AI Studio
- **Quota exceeded**: Check your Google Cloud billing/limits
- **Network errors**: Ensure internet connection

### Vector Database Issues:
- **ChromaDB errors**: Delete `chroma_db/` folder and rebuild
- **Memory issues**: Reduce chunk size in text splitter

### PDF Processing:
- **OCR failures**: Ensure tesseract is installed via conda
- **Large files**: May take time to process, be patient

---

## 🔒 Security Notes

- Never commit your `.env` file with actual API keys
- Keep your API key private and secure
- Monitor your Google Cloud usage and billing
- Consider setting up usage limits in Google Cloud Console

---

## 🚀 Advanced Usage

### Custom Prompts
Edit the `prompt_template` in `chatbot.py` or `app.py` to customize AI behavior.

### Model Selection
- **gemini-pro**: Standard model, good balance
- **gemini-pro-latest**: Latest version with improvements

### Temperature Settings
- **0.0-0.3**: More focused, factual responses
- **0.4-0.7**: Balanced creativity and accuracy
- **0.8-1.0**: More creative, less predictable

---

This online version provides enterprise-grade AI capabilities through Google's Gemini API while maintaining the same smart extraction and user-friendly interface as the local version.
