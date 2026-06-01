from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load your local .env file containing GOOGLE_API_KEY
load_dotenv()

# --- 1. LANGGRAPH CONFIGURATION ---

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

graph = builder.compile()

# --- 2. FASTAPI SERVER SETUP ---

app = FastAPI(title="LangGraph Math API")

# Define the data structure the API expects
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Format input for LangGraph's MessagesState
        inputs = {"messages": [("user", request.message)]}
        result = graph.invoke(inputs)
        
        # Get the final response from the graph
        final_message = result["messages"][-1].content
        return {"response": final_message}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "LangGraph FastAPI is healthy and running"}