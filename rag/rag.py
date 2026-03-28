from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import os
import json

# Keep compatibility across LangChain package split versions.
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma

def load_rag():
    # 1. Load data from knowledge.json
    json_path = "data/knowledge.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Creating empty knowledge.")
        knowledge_data = []
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            knowledge_data = json.load(f)

    # 2. Convert JSON items to LangChain Documents
    documents = []
    for item in knowledge_data:
        doc = Document(
            page_content=item.get("content", ""),
            metadata={
                "id": item.get("id"),
                "topic": item.get("topic"),
                "tags": item.get("tags", [])
            }
        )
        documents.append(doc)

    if not documents:
        # Fallback if no data
        documents = [Document(page_content="No information available yet.")]

    # 3. Create/Load Vector Database (Chroma)
    persist_dir = "rag/vector_db"
    embeddings = OpenAIEmbeddings() # Default OpenAI embeddings provider

    # Note: In production, we should avoid re-indexing every time if possible.
    # But for a small knowledge base, from_documents is fine.
    # To use existing, we would use Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_dir
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    # 4. Configure LLM (OpenRouter)
    # Note: Ensure OPENROUTER_API_KEY is in your .env
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model="mistralai/mistral-7b-instruct"
    )

    # 5. Create Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa
