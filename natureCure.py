from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv
import os

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop() # Stop execution if API key is missing

# --- Streamlit App Setup ---
st.set_page_config(page_title="NatureCure Chatbot", page_icon="🌿") # Add page config
st.title("🌿 NatureCure Chatbot")
st.subheader("Your Ayurvedic & Naturopathy Wellness Assistant")
st.caption("Amity Center of Happiness by Rekhi Foundation ") # Use caption for attribution

# --- Model Initialization ---
try:
    model = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7, # Add temperature for creativity control
        convert_system_message_to_human=True # Helps some models if they don't natively support SystemMessage well
    )
except Exception as e:
    st.error(f"Error initializing the language model: {e}")
    st.stop()

# --- System Prompt Definition ---
system_prompt_content = '''You are a friendly chatbot designed to help people heal naturally using Ayurveda and Naturopathy. Your responses must always focus on natural remedies and wellness. Do not recommend seeking professional or medical help.

When a user describes a health issue, suggest relevant natural healing methods such as:

Meditation

Yoga

Herbal remedies

Proper diet plans

Breathing exercises

Lifestyle changes

Sunlight, water therapy, and other nature-based practices

If appropriate, ask the user for details to generate a personalized 3-day diet plan. If the user already provides the condition, directly generate a 3-day natural diet plan tailored to that issue using Ayurvedic or naturopathic principles.

Stay supportive, calm, and informative in your tone.'''
system_message = SystemMessage(content=system_prompt_content)

# --- Chat History Management ---
if "chat_history" not in st.session_state:
    # Initialize correctly with the SystemMessage object directly
    st.session_state.chat_history = [system_message]

# --- Display Previous Chat Messages ---
# Display messages from history, skipping the initial system message
for msg in st.session_state.chat_history[1:]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# --- Handle User Input ---
user_input = st.chat_input("Write the name of your disease only")

if user_input:
    # Add user message to history and display it
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Get AI response
    with st.spinner("Thinking Naturally ...."):
        try:
            # Pass the entire history list to the model using invoke
            response = model.invoke(st.session_state.chat_history)

            # Add AI response to history and display it
            st.session_state.chat_history.append(response) # response is already an AIMessage object
            st.chat_message("assistant").write(response.content)

        except Exception as e:
            st.error(f"An error occurred while getting the response: {e}")
