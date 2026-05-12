import os
from dotenv import load_dotenv

load_dotenv()

from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Loading embeddings...")

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)

print("Connecting to Pinecone...")

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("Loading Groq LLM...")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

print("RAG chain ready.")


def get_rag_response(question, chat_history=[]):
    try:
        
        docs = retriever.invoke(question)

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
You are DocuMind AI.

Answer ONLY using the provided document context.

If the answer is not available in the documents,
say:
"I could not find this information in the uploaded documents."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)

       
        formatted_sources = []

        for doc in docs:
         formatted_sources.append({
        "file": doc.metadata.get("source_file", "Unknown"),
        "page": int(doc.metadata.get("page", 0))
        })

        return {
            "answer": response.content,
            "sources": formatted_sources
        }

    except Exception as e:
        return {
            "answer": f"Error occurred: {str(e)}",
            "sources": []
        }
