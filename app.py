import streamlit as st
import os
import time
import tempfile
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API Keys from environment variables
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Conversational Document Q&A", layout="wide")

st.title("Conversational Q&A with your Documents")
st.markdown("Powered by Groq, Llama3, and LangChain")

def get_documents_from_files(uploaded_files):
    """Loads documents from uploaded files."""
    docs = []
    temp_dir = tempfile.mkdtemp()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        loader = None
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_filepath)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_filepath)
        elif file.name.endswith('.txt'):
            loader = TextLoader(temp_filepath)
        
        if loader:
            docs.extend(loader.load())
    return docs

def create_vector_store(docs, embedding_model_name):
    """Creates a FAISS vector store from documents."""
    with st.spinner(f"Creating vector store with {embedding_model_name}..."):
        if embedding_model_name == "OpenAI":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API Key is not set.")
                return None
            embeddings = OpenAIEmbeddings()
        else: # Hugging Face
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    embedding_choice = st.selectbox(
        "Choose your embedding model:",
        ("Hugging Face (Sentence Transformers)", "OpenAI"),
        key="embedding_model"
    )
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            st.session_state.documents = get_documents_from_files(uploaded_files)
            st.session_state.vector_store = create_vector_store(st.session_state.documents, embedding_choice)
            st.success("Documents processed and vector store created!")
            st.session_state.chat_history = [AIMessage(content="Hello! I'm your document assistant. How can I help you today?")]
        else:
            st.warning("Please upload at least one document.")
            
    if st.button("Clear Chat"):
        st.session_state.chat_history = [AIMessage(content="Hello! I'm your document assistant. How can I help you today?")]


# --- Main Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I'm your document assistant. How can I help you today?")]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Contextualizer prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Main QA prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer concise and helpful.
\n\n
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_prompt = st.chat_input("Ask a question about your documents...")

if user_prompt:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process your documents first.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_prompt))
        
        with st.chat_message("Human"):
            st.write(user_prompt)
            
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vector_store.as_retriever()
                
                # Create chains
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                # Get response
                response = rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_prompt
                })
                
                st.write(response["answer"])
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))

