import os
import re
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import json
from langchain_core.tools import tool

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
persist_directory = "./chroma_langchain_db"
vector_store = Chroma(
    collection_name="project_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """
        Herramienta para recuperar los k=2 fragmentos más relevantes.
        Devuelve tanto la lista de Document como un string serializado.
    """
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def auto_load_data_pdfs(
    pdf_dir: str = "./data",
    pdf_names: list[str] = ["colombia1.pdf"]
) -> None:
    """
    Al iniciar el proyecto, indexa en Chroma los PDFs listados en pdf_names
    (ubicados en pdf_dir) sólo si todavía no hay nada en la base de datos.
    """
    marker_file = os.path.join(persist_directory, "loaded_pdfs.json")
    
    try:
        with open(marker_file, 'r') as f:
            loaded = json.load(f)
    except Exception:
        loaded = []

    updated = False
    
    for pdf_fname in pdf_names:
        if pdf_fname in loaded:
            print(f"✓ {pdf_fname} ya indexado, se omite")
            continue
        pdf_path = os.path.join(pdf_dir, pdf_fname)
        if not os.path.isfile(pdf_path):
            continue

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = pdf_fname

        vector_store.add_documents(chunks)
        print(f"✓ Indexado {pdf_fname}: {len(docs)} páginas, {len(chunks)} fragmentos")
        loaded.append(pdf_fname)
        updated = True

    if updated:
        os.makedirs(persist_directory, exist_ok=True)
        with open(marker_file, 'w') as f:
            json.dump(loaded, f)
            
        if hasattr(vector_store, 'persist'):
            vector_store.persist()
        else:
            client = getattr(vector_store, '_client', None)
            
            if client and hasattr(client, 'persist'):
                client.persist()

    print(f"Finalizó indexado de pdf's")

def strip_think_blocks(text: str) -> str:
    """
    Elimina completamente cualquier bloque <think>...</think>, incluidas las etiquetas.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
