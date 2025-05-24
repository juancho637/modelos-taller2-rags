import os
from dotenv import load_dotenv
import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from rag import auto_load_data_pdfs, strip_think_blocks
from graph import build_graph

load_dotenv()

def list_models():
    models_running = ollama.list()['models']
    available_models = [model["model"] for model in models_running]
    
    return available_models

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "True")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "")

auto_load_data_pdfs(pdf_dir="./data", pdf_names=["constitucion.pdf", "historia.pdf"])

st.title('📚 Historia de Colombia + constitución del 91')

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

graph = build_graph(llm)

if user_input := st.chat_input("Escribe algo…"):
    input_strip = user_input.strip()
    if input_strip:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        state_msgs = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                state_msgs.append(HumanMessage(m["content"]))
            else:
                state_msgs.append(AIMessage(m["content"]))
                
        try:
            with st.spinner("🧠 El modelo está pensando..."):
                # initial_state = {"messages": [{"role": "user", "content": input_strip}]}
                result = graph.compile().invoke({"messages": state_msgs})
                final_msg = result["messages"][-1]

                clean_answer = strip_think_blocks(final_msg.content)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})
                                
                st.chat_message("assistant").write(clean_answer)
        except Exception as e:
            st.error(f"Error: {e}")
