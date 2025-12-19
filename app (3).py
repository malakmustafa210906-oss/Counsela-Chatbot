
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from datetime import datetime
import nltk
nltk.download('punkt')

# -------------------------
# Sample intents dataset
# -------------------------
data = {
    'intent': ['greeting', 'greeting', 'goodbye', 'goodbye', 'sad', 'happy', 'anxious', 'thanks'],
    'pattern': ['hi', 'hello', 'bye', 'goodbye', 'i feel sad', 'i feel happy', 'i am anxious', 'thank you'],
    'response': [
        'Hello! How are you feeling today? 😊',
        'Hi there! I am here to listen. 🌸',
        'Goodbye! Take care of yourself. 🌸',
        'Bye! Hope to see you soon. 🌸',
        'I am sorry you are feeling sad. Want to talk about it?',
        'That is wonderful! Keep smiling! 🌸',
        'Take a deep breath. Try to focus on positive thoughts. 🌸',
        'You are welcome! I am always here to help. 🌸'
    ]
}

df = pd.DataFrame(data)

# -------------------------
# Train ML model: TF-IDF + Naive Bayes
# -------------------------
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df['pattern'], df['intent'])

# Mapping intent to response
intent_responses = dict(zip(df['intent'], df['response']))

# -------------------------
# Initialize Streamlit session state
# -------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_box' not in st.session_state:
    st.session_state.input_box = ''

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Universal AI Chatbot", page_icon="🤖")
st.title("🌸 Universal AI Chatbot")
st.write("Hello! I am your friendly AI chatbot. You can ask me anything, and I am here to listen. 😊")

# Display chat history
for chat in st.session_state.chat_history:
    if chat['sender'] == 'You':
        st.markdown(f"**You ({chat['time']}):** {chat['message']}")
    else:
        st.markdown(f"**Bot ({chat['time']}):** {chat['message']}")

# Clear chat function
def clear_chat():
    st.session_state.chat_history = []
    st.session_state.input_box = ''

st.button("Clear Chat", on_click=clear_chat)

# Send message function
def send_message():
    user_input = st.session_state.input_box.strip()
    if user_input:
        # Append user message
        st.session_state.chat_history.append({
            'sender': 'You',
            'message': user_input,
            'time': datetime.now().strftime('%H:%M')
        })
        # Predict intent
        predicted_intent = model.predict([user_input])[0]
        bot_reply = intent_responses.get(predicted_intent, "I'm here to listen. 🌸")
        # Append bot reply
        st.session_state.chat_history.append({
            'sender': 'Bot',
            'message': bot_reply,
            'time': datetime.now().strftime('%H:%M')
        })
        # Clear input box
        st.session_state.input_box = ''

# Input box
st.text_input("You:", key='input_box', on_change=send_message)
