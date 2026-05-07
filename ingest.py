import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "documind")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("New index created.")
    else:
        print("Index already exists.")
    print("Index ready.")


def load_documents(docs_folder="./docs"):
    all_docs = []
    for file in os.listdir(docs_folder):
        if file.endswith(".pdf"):
            path = os.path.join(docs_folder, file)
            print(f"Loading: {file}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = file
            all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages.")
    return all_docs


def chunk_documents(docs):
    docs = [d for d in docs if len(d.page_content.strip()) > 10]
    print(f"Pages with content: {len(docs)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def embed_and_upsert(chunks):
    if not chunks:
        print("ERROR: No chunks to embed.")
        return

    print("Generating embeddings with Cohere...")
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    print("Upserted to Pinecone successfully.")


def run_ingestion():
    print("=== DocuMind Ingestion Pipeline ===")
    print(f"OpenAI Key loaded:  {'YES' if OPENAI_API_KEY else 'NO'}")
    print(f"Pinecone Key loaded: {'YES' if PINECONE_API_KEY else 'NO'}")
    print(f"Cohere Key loaded:  {'YES' if COHERE_API_KEY else 'NO'}")
    print("===================================")
    create_pinecone_index()
    docs = load_documents()
    chunks = chunk_documents(docs)
    embed_and_upsert(chunks)
    print("===================================")
    print("Ingestion complete! Ready to query.")
    print("===================================")


if __name__ == "__main__":
    run_ingestion()