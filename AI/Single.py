import os
import streamlit as st
from dotenv import load_dotenv
import requests

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# ---------------------------
# Load environment variables
# ---------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

/* Title */
.title{
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:#38bdf8;
    margin-top:10px;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:24px;
    margin-bottom:35px;
}

/* Textbox */
.stTextInput input{
    background:#1e293b;
    color:white;
    border-radius:12px;
    border:1px solid #334155;
    padding:12px;
    font-size:15px;
}

/* Button */
.stButton>button{
    background:#38bdf8;
    color:black;
    border:none;
    border-radius:10px;
    padding:12px 28px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover{
    background:#0ea5e9;
    transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(56,189,248,.35);
}

/* Output Card */
.output-card{
    margin-top:30px;
    background:#1e293b;
    border-left:5px solid #38bdf8;
    border-radius:16px;
    padding:25px;
    color:white;
    box-shadow:0px 10px 30px rgba(0,0,0,.35);
}

.output-card h3{
    margin-top:0;
    color:#38bdf8;
    font-size:24px;
}

.output-card pre{
    white-space:pre-wrap;
    word-wrap:break-word;
    font-family:'Poppins',sans-serif;
    font-size:20px;
    line-height:1.8;
    color:#e2e8f0;
    margin:0;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Weather Tool
# ---------------------------
@tool
def get_weather(city: str) -> str:
    """Fetch weather for a city"""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weather API key not set"

    url = (
        f"http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    res = requests.get(url)
    data = res.json()

    if "current" not in data:
        return f"Error: {data}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# ---------------------------
# Tools
# ---------------------------
search_tool = TavilySearch(max_results=3)
tools = [search_tool, get_weather]

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"]
)

# ---------------------------
# Agent
# ---------------------------
agent = create_agent(
    model=llm,
    tools=tools
)

# ---------------------------
# UI
# ---------------------------
st.markdown("<div class='title'>🧠 AI Agent Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Weather + Web Search Powered by LangChain</div>", unsafe_allow_html=True)

user_input = st.text_input("Enter your query:", placeholder="e.g. Weather in Ongole or latest AI news")

if st.button("Run Agent"):
    if user_input.strip():

        with st.spinner("Thinking... 🤖"):
            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            })

        output = response["messages"][-1].content

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(output)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter a query first.")