import os
from dotenv import load_dotenv
import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from rag import load_pdf, retrieve, strip_think_blocks

def main():
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "True")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "")

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

    if user_input := st.chat_input("Escribe algo o sube un PDF…", accept_file=True, file_type=["pdf"]):
        if user_input.files:
            pages, chunks = load_pdf(user_input.files[0])
            st.chat_message("assistant").write(
                f"✅ PDF indexado: {pages} páginas, {chunks} fragmentos."
            )

        if user_input.text.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.text})
            st.chat_message("user").write(user_input.text)
            
            system_prompt = {
                "role": "system",
                "content": (
                    "Eres un experto en la información proporcionada por RAG. "
                    "Responde solo en texto plano, sin markdown."
                )
            }
            llm_messages = [system_prompt]
            
            for m in st.session_state.messages:
                role = "user" if m["role"] == "user" else "ai"
                llm_messages.append({"role": role, "content": m["content"]})
            
            try:
                with st.spinner("🧠 El modelo está pensando..."):
                    # context = retrieve(user_input.text)
                    response = llm.invoke(llm_messages)
                    clean = strip_think_blocks(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": clean})
                    st.chat_message("assistant").write(clean)
            except Exception as e:
                st.error(f"Error: {e}")

def list_models():
    models_running = ollama.list()['models']
    available_models = [model["model"] for model in models_running]
    
    return available_models

if __name__ == "__main__":
    main()
