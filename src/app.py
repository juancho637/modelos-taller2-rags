import os
from dotenv import load_dotenv
import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from rag import auto_load_data_pdfs, retrieve, strip_think_blocks

load_dotenv()

def list_models():
    models_running = ollama.list()['models']
    available_models = [model["model"] for model in models_running]
    
    return available_models

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "True")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "")

auto_load_data_pdfs(pdf_dir="./data", pdf_names=["constitucion.pdf", "historia.pdf"])

st.title('📚 Historia de Colombia + contitución')

lista = list_models()

with st.sidebar:
    st.header("⚙️ Parámetros del modelo")
    st.session_state.model_selection = st.selectbox("Modelo", lista)
    st.session_state.temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
    st.session_state.top_k = st.slider("Top K", 0, 100, 50, 1)
    st.session_state.max_tokens = st.slider("Max Tokens", 1, 4096, 1000, 1)

llm = ChatOllama(
    model=st.session_state.model_selection,
    temperature=st.session_state.temperature,
    num_predict=st.session_state.max_tokens,
    top_p=st.session_state.top_p,
    top_k=st.session_state.top_k,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Escribe algo…"):
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        retrieved_docs = retrieve(user_input, k=3)
        context_blocks = []
        
        print(retrieved_docs)
        
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "desconocido")
            text = doc.page_content
            context_blocks.append(f"[{src}]\n{text}")
            
        context_payload = "\n\n---\n\n".join(context_blocks)
                    
        system_prompt = {
            "role": "system",
            "content": (
                f"Eres un experto en la información proporcionada por RAG, solo vas a responder segun el contexto dada por la siguente respuesta: \n{context_payload if context_payload else "No se encontro información relacionada a la pregunta"} \n\nsi no te llega información vas a contestar de una forma cortez que no tienes información relacionada a la pregunta. Responde solo en texto plano, sin markdown."
            )
        }
        
        llm_messages = [system_prompt]
        
        for m in st.session_state.messages:
            role = "user" if m["role"] == "user" else "ai"
            llm_messages.append({"role": role, "content": m["content"]})
            
        # st.write(llm_messages)
        
        try:
            with st.spinner("🧠 El modelo está pensando..."):
                response = llm.invoke(llm_messages)
                clean = strip_think_blocks(response.content)
                
                st.session_state.messages.append({"role": "assistant", "content": clean})
                
                # st.write(st.session_state.messages)
                
                st.chat_message("assistant").write(clean)
        except Exception as e:
            st.error(f"Error: {e}")
