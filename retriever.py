import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "documind")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("Loading Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)


system_message = (
    "You are DocuMind, a precise corporate knowledge assistant. "
    "STRICT RULES: "
    "1. Answer ONLY using the provided context below. "
    "2. If the answer is NOT in the context, respond exactly: "
    "I don't have information on this topic in the provided documents. "
    "3. NEVER use your own knowledge or external information. "
    "4. Always cite your source at the end: Source: [filename], Page [page_number]. "
    "5. Be concise and professional. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_response(question: str, chat_history: list):
    messages = []
    for human, ai in chat_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))

    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": messages
    })

    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append({
            "file": meta.get("source_file", "Unknown"),
            "page": meta.get("page_number", meta.get("page", "N/A"))
        })

    unique_sources = [dict(t) for t in {tuple(s.items()) for s in sources}]

    return {
        "answer": answer,
        "sources": unique_sources
    }


print("RAG chain ready.")