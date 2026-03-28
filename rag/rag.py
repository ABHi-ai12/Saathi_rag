from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import os

def load_rag():
    # Load knowledge data
    # Note: ensure data/knowledge.txt or similar exists if using TextLoader
    # Based on main.py, you use data/knowledge.json, but here we expect a text file for simplicity in LangChain
    
    # Check if rag/data/knowledge.txt exists
    data_path = "data/knowledge.txt"
    if not os.path.exists(data_path):
        # Create a basic knowledge file if it doesn't exist
        with open(data_path, "w") as f:
            f.write("AI Dost is your friend. He speaks Hinglish.")

    loader = TextLoader(data_path)
    docs = loader.load()

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="rag/vector_db"
    )

    retriever = db.as_retriever()

    # OpenRouter LLM configuration
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model="mistralai/mistral-7b-instruct"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa
