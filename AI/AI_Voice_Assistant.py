import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>

/* Full Background */
.stApp {
    background: linear-gradient(
        120deg,
        #0f172a,
        #1e293b,
        #111827
    );
    color: white;
}


/* Main Title */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    color: #38bdf8;
    margin-top: 20px;
    letter-spacing: 1px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 40px;
}


/* Audio Recorder */
[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.05);
    border: 2px solid #38bdf8;
    border-radius: 18px;
    padding: 15px;
}


/* Audio Player */
audio {
    width: 100%;
    border-radius: 15px;
    margin-top: 15px;
}


/* Question and Answer text */
.stMarkdown p {
    font-size: 20px;
    color: #f8fafc;
    line-height: 1.6;
}


/* Headers */
h2, h3 {
    color: #38bdf8 !important;
    font-weight: 700;
}


/* Recording label */
label {
    color: #e0f2fe !important;
    font-size: 18px !important;
    font-weight: 600;
}


/* Success Message */
.stSuccess {
    background: rgba(34,197,94,0.15);
    border-radius: 12px;
}


/* Warning Message */
.stWarning {
    border-radius: 12px;
}


/* Remove empty markdown boxes */
div:empty {
    display:none;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load API --------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API key is missing. Please set it in your .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -------------------- AI Response --------------------
def generate_ai_response(question):
    chat_completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI Voice Assistant. Answer clearly and concisely."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    return chat_completion.choices[0].message.content


# -------------------- Text to Speech --------------------
def convert_text_to_speech(text):
    audio_buffer = BytesIO()

    speech = gTTS(
        text=text,
        lang="en"
    )

    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer


# -------------------- UI --------------------
st.markdown('<div class="title">🎙️ AI Voice Assistant</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Ask your question using your voice and get an AI-powered spoken response.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

audio_value = st.audio_input("🎤 Record Your Question")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Processing --------------------
if audio_value is not None:

    st.audio(audio_value)

    with st.spinner("🎧 Converting speech to text..."):

        transcription = client.audio.transcriptions.create(
            file=(
                "recording.wav",
                audio_value.getvalue()
            ),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0.0
        )

        transcribed_text = transcription.text.strip()

    if transcribed_text:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🗣️ You Asked")
        st.write(transcribed_text)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("🤖 AI is thinking..."):
            ai_response = generate_ai_response(transcribed_text)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("💡 AI Assistant")
        st.write(ai_response)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("🔊 Generating voice..."):
            response_audio = convert_text_to_speech(ai_response)

        st.audio(response_audio, format="audio/mp3")

        st.success("✅ Response generated successfully!")

    else:
        st.warning("No speech was detected. Please record your question again.")

