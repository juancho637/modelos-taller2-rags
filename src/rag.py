
import tempfile
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vector_store = Chroma(
    collection_name="project_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def load_pdf(uploaded_file) -> tuple[int, int]:
    """
    Carga un PDF, fragmenta sus páginas y los indexa en Chroma.
    Devuelve: (número de páginas, número de fragmentos).
    """
    vector_store.reset_collection()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.getvalue())
    tmp.close()

    docs = PyPDFLoader(file_path=tmp.name).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store.add_documents(chunks)

    return len(docs), len(chunks)

def retrieve(query: str, k: int = 2) -> str:
    """
    Realiza una búsqueda semántica en el índice y devuelve
    los fragmentos encontrados serializados en un string.
    """
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in docs
    )

def strip_think_blocks(text: str) -> str:
    """
    Elimina completamente cualquier bloque <think>...</think>, incluidas las etiquetas.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
