from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from rag import retrieve_tool

def build_graph(llm) -> StateGraph:
    graph = StateGraph(MessagesState)

    # Nodo A: LLM con herramientas
    def query_or_respond(state: MessagesState):
        agent = llm.bind_tools([retrieve_tool])
        # state["messages"] es List[BaseMessage] (Human, AI, Tool…)
        out = agent.invoke(state["messages"])
        return {"messages": [out]}

    # Nodo de ejecución de herramientas
    tools_node = ToolNode([retrieve_tool])

    # Nodo B: generar respuesta final
    def generate(state: MessagesState):
        # Extraemos solo los tool-messages
        tool_msgs = [m for m in state["messages"] if m.type == "tool"]
        contexto = "\n\n".join(m.content for m in tool_msgs)
        sys = SystemMessage(
            "Eres un asistente experto usando este contexto:\n\n" + contexto
        )
        # Filtramos la conversación “limpia” (usuario + asistente sin tool_calls)
        convo = [
            m for m in state["messages"]
            if m.type in ("system","human")
               or (m.type=="ai" and not m.tool_calls)
        ]
        prompt = [sys] + convo
        resp = llm.invoke(prompt)
        return {"messages": [resp]}

    # Registramos nodos y aristas
    graph.add_node(query_or_respond)
    graph.add_node(tools_node)
    graph.add_node(generate)

    graph.set_entry_point("query_or_respond")
    # Si LLM genera un tool_call va → tools_node, si no → fin
    graph.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "generate")
    graph.add_edge("generate", END)

    return graph
