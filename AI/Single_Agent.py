import os
import requests
import gradio as gr

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# =========================================================
# CHECK API KEYS
# =========================================================

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY is missing in your .env file"
    )

if not TAVILY_API_KEY:
    print("⚠️ TAVILY_API_KEY is missing. Web search may not work.")

if not WEATHERSTACK_API_KEY:
    print("⚠️ WEATHERSTACK_API_KEY is missing. Weather tool may not work.")


# =========================================================
# WEATHER TOOL
# =========================================================

@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a given city.
    Use this tool when the user asks about weather,
    temperature, humidity, wind, or weather conditions.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "❌ WEATHERSTACK_API_KEY is not configured."

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": city
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        data = response.json()

        # Check API error
        if "current" not in data:

            if "error" in data:

                error_message = data["error"].get(
                    "info",
                    "Unknown Weatherstack API error"
                )

                return f"❌ Weather API Error: {error_message}"

            return f"❌ Weather information not found for {city}."


        # Get weather information

        current = data["current"]
        location = data.get("location", {})


        weather_description = "N/A"

        if current.get("weather_descriptions"):
            weather_description = current[
                "weather_descriptions"
            ][0]


        return f"""
🌍 **Location:** {location.get('name', city)}, {location.get('country', '')}

🌡️ **Temperature:** {current.get('temperature', 'N/A')}°C

☁️ **Weather:** {weather_description}

💧 **Humidity:** {current.get('humidity', 'N/A')}%

💨 **Wind Speed:** {current.get('wind_speed', 'N/A')} km/h

🧭 **Wind Direction:** {current.get('wind_dir', 'N/A')}

👁️ **Visibility:** {current.get('visibility', 'N/A')} km
"""

    except requests.exceptions.Timeout:

        return "❌ Weather request timed out."

    except requests.exceptions.RequestException as e:

        return f"❌ Weather request failed: {str(e)}"

    except Exception as e:

        return f"❌ Unexpected weather error: {str(e)}"


# =========================================================
# TAVILY SEARCH TOOL
# =========================================================

search_tool = TavilySearch(
    max_results=3
)


# =========================================================
# TOOLS
# =========================================================

tools = [
    search_tool,
    get_weather
]


# =========================================================
# GROQ LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.3
)


# =========================================================
# CREATE AGENT
# =========================================================

agent = create_agent(
    model=llm,
    tools=tools
)


# =========================================================
# GET TEXT FROM LANGCHAIN RESPONSE
# =========================================================

def extract_response_text(response):

    try:

        messages = response.get("messages", [])

        if not messages:
            return "❌ No response received from the AI Agent."

        # Get last message
        last_message = messages[-1]

        content = last_message.content

        # Normal string response
        if isinstance(content, str):
            return content

        # List response
        if isinstance(content, list):

            final_text = ""

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        final_text += item.get("text", "")

                    elif "text" in item:
                        final_text += str(item["text"])

                else:
                    final_text += str(item)

            return final_text

        return str(content)

    except Exception as e:

        return f"❌ Error processing AI response: {str(e)}"


# =========================================================
# RUN AGENT
# =========================================================

def run_agent(user_input, history):

    # Initialize history

    if history is None:
        history = []


    # Validate input

    if not user_input or not user_input.strip():

        return history, ""


    try:

        # Add user message to Gradio history FIRST

        history.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # ---------------------------------------------
        # CREATE CONVERSATION FOR LANGCHAIN
        # ---------------------------------------------

        langchain_messages = []


        # Add previous chat messages

        for message in history:

            if isinstance(message, dict):

                role = message.get("role")
                content = message.get("content")

                # Only valid roles
                if role in ["user", "assistant"]:

                    langchain_messages.append(
                        {
                            "role": role,
                            "content": content
                        }
                    )


        # ---------------------------------------------
        # CALL LANGCHAIN AGENT
        # ---------------------------------------------

        response = agent.invoke(
            {
                "messages": langchain_messages
            }
        )


        # ---------------------------------------------
        # EXTRACT FINAL RESPONSE
        # ---------------------------------------------

        output = extract_response_text(response)


        # ---------------------------------------------
        # ADD AI RESPONSE TO CHAT HISTORY
        # ---------------------------------------------

        history.append(
            {
                "role": "assistant",
                "content": output
            }
        )


        return history, ""


    except Exception as e:

        error_message = f"""
❌ **Agent Error**

{str(e)}
"""


        history.append(
            {
                "role": "assistant",
                "content": error_message
            }
        )


        return history, ""


# =========================================================
# CLEAR CHAT
# =========================================================

def clear_chat():

    return [], ""


# =========================================================
# CUSTOM CSS
# =========================================================

css = """

.gradio-container {

    max-width: 1150px !important;

    margin: auto !important;

    background:
        linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #1e293b
        ) !important;

}


/* HEADER */

#title {

    text-align: center;

    font-size: 48px;

    font-weight: bold;

    color: #38bdf8;

    margin-top: 20px;

}


#subtitle {

    text-align: center;

    font-size: 20px;

    color: #cbd5e1;

    margin-bottom: 30px;

}


/* MAIN CARD */

#main-container {

    background: #111827;

    border-radius: 20px;

    padding: 25px;

    border: 1px solid #334155;

    box-shadow:
        0px 10px 30px
        rgba(0, 0, 0, 0.4);

}


/* CHATBOT */

#chatbot {

    border-radius: 15px !important;

    border: 1px solid #334155 !important;

}


/* TEXTBOX */

textarea {

    background: #1e293b !important;

    color: white !important;

    border-radius: 12px !important;

}


/* RUN BUTTON */

.run-button button {

    background:
        linear-gradient(
            135deg,
            #38bdf8,
            #0ea5e9
        ) !important;

    color: #020617 !important;

    font-weight: bold !important;

    border-radius: 12px !important;

    border: none !important;

}


/* CLEAR BUTTON */

.clear-button button {

    background: #334155 !important;

    color: white !important;

    border-radius: 12px !important;

}


/* FOOTER */

#footer {

    text-align: center;

    color: #94a3b8;

    margin-top: 25px;

    font-size: 14px;

}

"""


# =========================================================
# CREATE GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="AI Agent Assistant"
) as demo:


    # HEADER

    gr.HTML(
        """
        <div id="title">
            🧠 AI Agent Assistant
        </div>

        <div id="subtitle">
            Weather 🌦️ + Web Search 🔎 Powered by LangChain & Groq
        </div>
        """
    )


    # MAIN CONTAINER

    with gr.Column(
        elem_id="main-container"
    ):


        # CHATBOT

        chatbot = gr.Chatbot(
            label="🤖 AI Assistant",
            height=500,
            elem_id="chatbot",
            value=[]
        )


        # INPUT AREA

        with gr.Row():

            user_input = gr.Textbox(
                label="💬 Ask Your Question",
                placeholder=(
                    "Example: What is the weather in Ongole?"
                ),
                lines=2,
                scale=4
            )


            with gr.Column(scale=1):

                run_button = gr.Button(
                    "🚀 Run Agent",
                    variant="primary",
                    elem_classes="run-button"
                )


                clear_button = gr.Button(
                    "🗑️ Clear Chat",
                    elem_classes="clear-button"
                )


        # EXAMPLES

        gr.Markdown("## 💡 Example Questions")


        gr.Examples(

            examples=[

                "What is the weather in Ongole?",

                "What is the weather in Hyderabad?",

                "What are the latest developments in Artificial Intelligence?",

                "Search for latest Generative AI news",

                "Who is the CEO of OpenAI?"

            ],

            inputs=user_input

        )


        # RUN BUTTON

        run_button.click(

            fn=run_agent,

            inputs=[
                user_input,
                chatbot
            ],

            outputs=[
                chatbot,
                user_input
            ]

        )


        # ENTER KEY

        user_input.submit(

            fn=run_agent,

            inputs=[
                user_input,
                chatbot
            ],

            outputs=[
                chatbot,
                user_input
            ]

        )


        # CLEAR CHAT

        clear_button.click(

            fn=clear_chat,

            outputs=[
                chatbot,
                user_input
            ]

        )


    # FOOTER

    gr.HTML(
        """
        <div id="footer">
            🚀 Powered by LangChain • Groq • Tavily • Weatherstack
        </div>
        """
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch(
        css=css
    )
