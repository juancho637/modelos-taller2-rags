
import os
import re
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
persist_directory = "./chroma_langchain_db"
vector_store = Chroma(
    collection_name="project_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

def auto_load_data_pdfs(
    pdf_dir: str = "./data",
    pdf_names: list[str] = ["colombia1.pdf"]
) -> None:
    """
    Al iniciar el proyecto, indexa en Chroma los PDFs listados en pdf_names
    (ubicados en pdf_dir) sólo si todavía no hay nada en la base de datos.
    """
    for pdf_fname in pdf_names:
        pdf_path = os.path.join(pdf_dir, pdf_fname)
        if not os.path.isfile(pdf_path):
            continue

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = pdf_fname

        vector_store.add_documents(chunks)
        print(f"✓ Indexado {pdf_fname}: {len(docs)} páginas, {len(chunks)} fragmentos")


    print(f"Finalizó indexado de pdf's")

def retrieve(query: str, k: int = 2) -> List[Document]:
    """
    Realiza una búsqueda semántica en el índice y devuelve
    los fragmentos encontrados serializados en un string.
    """
    return vector_store.similarity_search(query, k=k)

def strip_think_blocks(text: str) -> str:
    """
    Elimina completamente cualquier bloque <think>...</think>, incluidas las etiquetas.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
