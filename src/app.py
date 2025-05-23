import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import ChatOllama
from rag import load_pdf, retrieve, strip_think_blocks

def main():
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "True")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "")

    st.title('📚 Chatbot RAG')

    # — Estado de sesión —
    st.session_state.setdefault("model_selection", "qwen3:8b")

    # — Sidebar de parámetros —
    with st.sidebar:
        st.header("⚙️ Parámetros del modelo")
        st.session_state.model_selection = st.selectbox("Modelo", ["qwen3:8b"])
        st.session_state.temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
        st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
        st.session_state.top_k = st.slider("Top K", 0, 100, 50, 1)
        st.session_state.max_tokens = st.slider("Max Tokens", 1, 4096, 1000, 1)

    # — Inicialización del LLM —
    llm = ChatOllama(
        model=st.session_state.model_selection,
        temperature=st.session_state.temperature,
        num_predict=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
        top_k=st.session_state.top_k,
    )

    # — Lógica de chat —
    if user_input := st.chat_input("Escribe algo o sube un PDF…", accept_file=True, file_type=["pdf"]):
        # 📄 Si suben un PDF, indexarlo
        if user_input.files:
            pages, chunks = load_pdf(user_input.files[0])
            st.chat_message("assistant").write(
                f"✅ PDF indexado: {pages} páginas, {chunks} fragmentos."
            )

        # 💬 Si envían texto, hacer RAG + LLM
        if user_input.text.strip():
            st.chat_message("user").write(user_input.text)
            try:
                with st.spinner("🧠 El modelo está pensando..."):
                    context = retrieve(user_input.text)
                    response = llm.invoke([
                        {
                            "role": "system",
                            "content": (
                                "Eres un experto en la información proporcionada por RAG. "
                                "Responde solo en texto plano, sin markdown."
                            )
                        },
                        {"role": "ai",   "content": context},
                        {"role": "user", "content": user_input.text}
                    ])
                    clean = strip_think_blocks(response.content)
                    st.chat_message("assistant").write(clean)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
