import streamlit as st
import os

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Multi AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #312e81 100%
    );
}

header {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1e1b4b
    );
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

[data-testid="stChatMessage"] {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.15);
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-weight: bold;
    background: #6366f1;
    color: white;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR - MODEL SELECTION
# ==================================================

with st.sidebar:

    st.title("🤖 Multi AI Chatbot")

    st.markdown("---")

    provider = st.selectbox(
        "🌐 Select AI Provider",
        [
            "Groq",
            "Google Gemini",
            "Mistral AI"
        ]
    )


    # ==============================================
    # GROQ MODELS
    # ==============================================

    if provider == "Groq":

        model_name = st.selectbox(
            "Select Groq Model",
            [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b"
            ]
        )


    # ==============================================
    # GEMINI MODELS
    # ==============================================

    elif provider == "Google Gemini":

        model_name = st.selectbox(
            "Select Gemini Model",
            [
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite"
            ]
        )


    # ==============================================
    # MISTRAL MODELS
    # ==============================================

    else:

        model_name = st.selectbox(
            "Select Mistral Model",
            [
                "mistral-small-latest",
                "mistral-medium-latest"
            ]
        )


    st.markdown("---")

    st.write("### 🤖 Current Model")

    st.info(
        f"""
Provider: {provider}

Model: {model_name}
"""
    )

    st.markdown("---")


    # ==============================================
    # CLEAR CHAT
    # ==============================================

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


    st.markdown("---")

    st.write("### 🚀 Available Providers")

    st.write("""
🟢 Groq

🔵 Google Gemini

🟠 Mistral AI
""")


# ==================================================
# CREATE LLM
# ==================================================

def get_llm(provider, model_name):

    if provider == "Groq":

        return ChatGroq(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )


    elif provider == "Google Gemini":

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )


    elif provider == "Mistral AI":

        return ChatMistralAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("MISTRAL_API_KEY")
        )


# ==================================================
# CONVERT AI RESPONSE TO NORMAL TEXT
# ==================================================

def get_text_content(content):

    # Normal string response
    if isinstance(content, str):
        return content


    # Some models return a list of content blocks
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):

                text_parts.append(item)

            elif isinstance(item, dict):

                # Usually the text is stored in "text"
                if "text" in item:

                    text_parts.append(
                        str(item["text"])
                    )

        return "\n".join(text_parts)


    # Fallback
    return str(content)


# ==================================================
# LANGGRAPH STATE
# ==================================================

class ChatState(TypedDict):

    messages: Annotated[list, add_messages]


# ==================================================
# CREATE LANGGRAPH CHATBOT
# ==================================================

def create_chatbot(llm):

    def chat_node(state: ChatState):

        response = llm.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }


    graph = StateGraph(ChatState)

    graph.add_node(
        "chat_node",
        chat_node
    )

    graph.add_edge(
        START,
        "chat_node"
    )

    graph.add_edge(
        "chat_node",
        END
    )

    return graph.compile()


# ==================================================
# LOAD SELECTED MODEL
# ==================================================

llm = get_llm(
    provider,
    model_name
)

chatbot = create_chatbot(llm)


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    """
    <div class="main-title">
        🤖 Multi AI Chatbot
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="subtitle">
        Powered by {provider} ⚡ {model_name}
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# INITIALIZE CHAT MEMORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.write(
                get_text_content(message.content)
            )


    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.write(
                get_text_content(message.content)
            )


# ==================================================
# CHAT INPUT
# ==================================================

user_message = st.chat_input(
    "💬 Ask me anything..."
)


if user_message:

    # ==============================================
    # USER MESSAGE
    # ==============================================

    human_message = HumanMessage(
        content=user_message
    )

    st.session_state.messages.append(
        human_message
    )


    with st.chat_message("user"):

        st.write(user_message)


    # ==============================================
    # AI RESPONSE
    # ==============================================

    with st.chat_message("assistant"):

        with st.spinner(
            f"🤖 {provider} AI is thinking..."
        ):

            try:

                response = chatbot.invoke(
                    {
                        "messages":
                        st.session_state.messages
                    }
                )


                ai_message = response["messages"][-1]


                # ==================================
                # EXTRACT ONLY TEXT
                # ==================================

                ai_text = get_text_content(
                    ai_message.content
                )


                # ==================================
                # DISPLAY NORMAL TEXT
                # ==================================

                st.write(ai_text)


                # ==================================
                # SAVE MESSAGE
                # ==================================

                st.session_state.messages.append(
                    AIMessage(
                        content=ai_text
                    )
                )


            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        🚀 Built with Streamlit + LangGraph + Multi AI Models
    </div>
    """,
    unsafe_allow_html=True
)