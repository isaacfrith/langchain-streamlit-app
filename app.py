import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Set up Page Config
st.set_page_config(page_title="Math Assistant", page_icon="🧮")
st.title("🧮 LangGraph Math Assistant")
st.write("Ask me to perform addition, multiplication, or division!")

# 2. Define LangGraph Logic (Cached so it only builds once)
@st.cache_resource
def compile_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiplies a and b."""
        return a * b

    def divide(a: int, b: int) -> float:
        """Divide a and b."""
        return a / b

    tools = [add, multiply, divide]
    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

graph = compile_graph()

# 3. Manage Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content: # Only show text content, skip tool calls
        with st.chat_message("assistant"):
            st.write(msg.content)

# 4. Handle User Input
if user_input := st.chat_input("What would you like me to calculate?"):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Append to session state
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Run LangGraph with the updated history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Streamlit keeps history as a list of LangChain objects, which works perfectly with MessagesState
            inputs = {"messages": st.session_state.messages}
            result = graph.invoke(inputs)
            
            # Extract final message text
            final_reply = result["messages"][-1].content
            st.write(final_reply)
            
            # Append AI reply to session state to maintain conversation history
            st.session_state.messages.append(AIMessage(content=final_reply))