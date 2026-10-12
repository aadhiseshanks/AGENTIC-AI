import streamlit as st
import os
import math
import requests

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_tavily import TavilySearch

# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Main Title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Chat Messages */
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

/* Cards */
.feature-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    text-align: center;
    color: white;
}

/* Button */
.stButton button {
    width: 100%;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()


# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="main-title">🤖 LangGraph Multi AI Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An Intelligent AI Assistant Powered by LangGraph & Tools</div>',
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ Multi AI Agent")

    st.divider()

    st.subheader("🔧 Available Tools")

    st.success("🔎 Web Search")
    st.success("🧮 Calculator")
    st.success("📈 Stock Price")
    st.success("🌤️ Weather")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption("Powered by LangGraph 🚀")


# =====================================================
# TOOLS
# =====================================================


# 🔎 SEARCH TOOL

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


# 🧮 CALCULATOR TOOL

@tool
def calculator(expression: str) -> str:
    """
    Useful for mathematical calculations.

    Examples:
    2 + 2
    10 * 5
    math.sqrt(16)
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:

        return f"Calculation Error: {str(e)}"


# 📈 STOCK TOOL

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get latest stock price.

    Example:
    AAPL
    TSLA
    MSFT
    """

    api_key = os.getenv("ALPHA_API_KEY")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    response = requests.get(url)

    return response.json()


# 🌤️ WEATHER TOOL

@tool
def get_current_weather(location: str) -> str:

    """
    Get current weather information.

    Example:
    Chennai
    London
    New York
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:

        return (
            "Weather API key is missing. "
            "Please configure OPENWEATHER_API_KEY."
        )

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}"
        f"&appid={api_key}"
        f"&units=metric"
    )

    try:

        response = requests.get(url)

        data = response.json()

        if response.status_code != 200:

            return f"Weather Error: {data.get('message')}"

        temperature = data["main"]["temp"]

        description = data["weather"][0]["description"]

        humidity = data["main"]["humidity"]

        return f"""
🌍 Location: {location}

🌡️ Temperature: {temperature} °C

☁️ Condition: {description.title()}

💧 Humidity: {humidity}%
"""

    except Exception as e:

        return f"Weather Error: {str(e)}"


# =====================================================
# ALL TOOLS
# =====================================================

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather
]


# =====================================================
# CREATE LANGGRAPH
# =====================================================

@st.cache_resource
def create_agent():

    llm = ChatGroq(

        model="openai/gpt-oss-120b",

        api_key=os.getenv("GROQ_API_KEY")

    )


    llm_with_tools = llm.bind_tools(tools)


    # -----------------------------
    # STATE
    # -----------------------------

    class ChatState(TypedDict):

        messages: Annotated[
            list[BaseMessage],
            add_messages
        ]


    # -----------------------------
    # CHAT NODE
    # -----------------------------

    def chat_node(state: ChatState):

        messages = state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }


    # -----------------------------
    # TOOL NODE
    # -----------------------------

    tool_node = ToolNode(tools)


    # -----------------------------
    # GRAPH
    # -----------------------------

    graph = StateGraph(ChatState)


    graph.add_node(
        "Chat Node",
        chat_node
    )

    graph.add_node(
        "tools",
        tool_node
    )


    graph.add_edge(
        START,
        "Chat Node"
    )


    graph.add_conditional_edges(
        "Chat Node",
        tools_condition
    )


    graph.add_edge(
        "tools",
        "Chat Node"
    )


    chatbot = graph.compile()


    return chatbot


# =====================================================
# INITIALIZE CHATBOT
# =====================================================

chatbot = create_agent()


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =====================================================
# FEATURE CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="feature-card">
    🔎<br>
    <b>Web Search</b><br>
    Search information online
    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="feature-card">
    🧮<br>
    <b>Calculator</b><br>
    Solve mathematical problems
    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="feature-card">
    📈<br>
    <b>Stock Market</b><br>
    Get stock information
    </div>
    """, unsafe_allow_html=True)


with col4:

    st.markdown("""
    <div class="feature-card">
    🌤️<br>
    <b>Weather</b><br>
    Get live weather updates
    </div>
    """, unsafe_allow_html=True)


st.write("")

st.divider()


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.write(message.content)


    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.write(message.content)


# =====================================================
# CHAT INPUT
# =====================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:


    # ---------------------------------
    # DISPLAY USER MESSAGE
    # ---------------------------------

    with st.chat_message("user"):

        st.write(user_input)


    # ---------------------------------
    # STORE USER MESSAGE
    # ---------------------------------

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )


    # ---------------------------------
    # AI RESPONSE
    # ---------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 AI is thinking..."):

            try:

                response = chatbot.invoke({

                    "messages":
                    st.session_state.messages

                })


                final_message = response[
                    "messages"
                ][-1]


                st.write(
                    final_message.content
                )


                # Store all messages

                st.session_state.messages = response[
                    "messages"
                ]


            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )