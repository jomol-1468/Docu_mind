import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tkinter import Tk, filedialog


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "documind"
)


def create_pinecone_index():

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i.name for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:

        print("Creating new Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Index created successfully.")

    else:
        print("Index already exists.")

    print("Pinecone index ready.")

def select_pdf_file():

    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select PDF File",
        filetypes=[("PDF Files", "*.pdf")]
    )

    return file_path


def load_pdf():

    pdf_path = select_pdf_file()

    if not pdf_path:
        print("No PDF selected.")
        return []

    print(f"\nSelected PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    # Add filename metadata
    file_name = os.path.basename(pdf_path)

    for doc in docs:
        doc.metadata["source_file"] = file_name

    print(f"Loaded {len(docs)} pages.")

    return docs


def chunk_documents(docs):

    docs = [
        d for d in docs
        if len(d.page_content.strip()) > 10
    ]

    print(f"Pages with valid content: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def embed_and_store(chunks):

    if not chunks:
        print("No chunks available.")
        return

    print("\nGenerating embeddings using Cohere...")

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    print("Uploading vectors to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )

    print("Upload completed successfully.")


def run_ingestion():

    print("\n========== DOCUMIND INGESTION ==========\n")

    print(
        f"OpenAI API Key Loaded  : {'YES' if OPENAI_API_KEY else 'NO'}"
    )

    print(
        f"Pinecone API Key Loaded: {'YES' if PINECONE_API_KEY else 'NO'}"
    )

    print(
        f"Cohere API Key Loaded  : {'YES' if COHERE_API_KEY else 'NO'}"
    )

    print("\n========================================\n")

    create_pinecone_index()

   
    docs = load_pdf()

    if not docs:
        print("PDF loading failed.")
        return

    chunks = chunk_documents(docs)

 
    embed_and_store(chunks)

    print("\n========================================")
    print("DocuMind ingestion completed successfully.")
    print("PDF is now searchable using RAG.")
    print("========================================\n")

if __name__ == "__main__":
    run_ingestion()
