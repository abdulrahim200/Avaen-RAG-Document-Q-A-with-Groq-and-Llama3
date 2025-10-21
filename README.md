# 🧠 RAG Document Q&A with Groq and Llama3

This project is a **web-based application** built with **Streamlit** that allows you to **ask questions about your own PDF documents**.  
It uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide **accurate answers** based on the content of the documents.

The application leverages the **high-speed inference capabilities of the Groq API** with the **powerful Llama3 language model** to generate responses.  
It offers a choice between **OpenAI** and **Hugging Face** models for creating text embeddings.

---

## ✨ Features

- **Interactive UI:** A simple and intuitive web interface powered by Streamlit.  
- **PDF Document Support:** Ingests and processes multiple PDF files from a specified directory.  
- **High-Speed LLM:** Utilizes the Groq API for near-instantaneous responses from the Llama3 model.  
- **Flexible Embedding Options:** Supports both OpenAI and Hugging Face (`all-MiniLM-L6-v2`) embedding models.  
- **Vector Storage:** Uses **FAISS** for efficient similarity searches and retrieval of relevant document chunks.  
- **Contextual Answers:** Displays the source text from the documents used to generate the answer.

---

## 🧩 Technology Stack

| Component | Technology |
|------------|-------------|
| **Backend/UI** | Streamlit |
| **LLM Orchestration** | LangChain |
| **LLM Provider** | Groq (Llama3-8b-8192) |
| **Embedding Models** | OpenAI, Hugging Face |
| **Vector Store** | FAISS |
| **Document Loading** | PyPDF |

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone <https://github.com/abdulrahim200/Avaen-RAG-Document-Q-A-with-Groq-and-Llama3>
cd <your-repository-name>
## ⚙️ Setup and Installation
```
### 2. Create a Virtual Environment
It’s recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
### 4. Set Up Environment Variables

Create a file named `.env` in the root directory and add your API keys:

```bash
GROQ_API_KEY="your-groq-api-key"
OPENAI_API_KEY="your-openai-api-key"   # Required if using OpenAI embeddings
HF_TOKEN="your-huggingface-token"      # Required if using Hugging Face embeddings
```
### 5. Add Your Documents

Create a folder named `research_papers` in the root directory and place all your **PDF files** inside it.
## 🚀 Usage

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py
```
Then navigate to the local URL provided by Streamlit (usually http://localhost:8501).

