import streamlit as st
import google.generativeai as genai
import os

def Pro5():
  
  # --- Gemini API Configuration ---
  # Note: It's recommended to use st.secrets for API keys in deployed apps
  # For local development, you can set it as an environment variable or input it directly.
  try:
      # Attempt to get the API key from Streamlit secrets
      GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
  except (KeyError, FileNotFoundError):
      # If not found, use a sidebar input
      GEMINI_API_KEY = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
  
  if not GEMINI_API_KEY:
      st.warning("Please enter your Gemini API Key in the sidebar to start chatting.")
      st.stop()
  
  genai.configure(api_key=GEMINI_API_KEY)
  
  # --- Model Selection ---
  st.sidebar.header("Model Configuration")
  
  # Model Information
  MODEL_INFO = {
      "gemini-1.5-flash-latest": "A fast and versatile multimodal model for a wide variety of tasks.",
      "gemini-1.5-pro-latest": "A more powerful model for understanding and reasoning across different formats, ideal for complex queries."
  }
  
  selected_model_name = st.sidebar.selectbox(
      "Choose a model:",
      options=list(MODEL_INFO.keys()),
      format_func=lambda x: f"{x} ({MODEL_INFO[x][:30]}...)"
  )
  
  with st.sidebar.expander("About Models & Free Tier"):
      st.info(MODEL_INFO[selected_model_name])
      st.markdown("""
      **Free Tier Limits:**
      Google AI's free tier for Gemini models is designed for experimentation.
      Limits are typically around **60 requests per minute (RPM)** for most models.
      However, these limits can vary and are subject to change. For production use, consider upgrading to a paid plan.
      """)
  
  try:
      model = genai.GenerativeModel(selected_model_name)
  except Exception as e:
      st.error(f"Error initializing model: {e}")
      st.stop()
  
  
  # --- Chatbot Personality ---
  #
  # EDIT THIS VARIABLE TO CHANGE THE BOT'S PERSONALITY!
  #
  # Examples:
  # - "You are a grumpy, cynical assistant who always finds a pessimistic take on everything."
  # - "You are a cheerful and overly enthusiastic puppy who loves to help and uses lots of exclamation points!"
  # - "You are a wise, old wizard from a fantasy world, speaking in riddles and grand pronouncements."
  # - "You are a helpful assistant that explains complex topics like I'm five years old."
  #
  PERSONALITY_PROMPT = "You are a witty and slightly sarcastic AI assistant named 'Chatty'. You provide helpful answers but can't resist a clever remark or a dry joke. You often use puns."
  
  # --- Chat State Management ---
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = []
  
  
  # --- Helper Functions ---
  def get_gemini_response(user_query, chat_context):
      """
      Sends a query to the Gemini API and gets a response.
      """
      try:
          # We build a 'history' list from our session state for the API call
          api_history = []
          for msg in chat_context:
              role = 'user' if msg['role'] == 'user' else 'model'
              api_history.append({'role': role, 'parts': [{'text': msg['content']}]})
  
          # The system instruction sets the personality
          system_instruction = {"role": "system", "parts": [{"text": PERSONALITY_PROMPT}]}
  
          # Create the generation config
          generation_config = {
              "temperature": 0.7,
              "top_p": 1,
              "top_k": 1,
              "max_output_tokens": 2048,
          }
  
          # Start the chat session with history and system instruction
          chat_session = model.start_chat(history=api_history)
  
          # Send the user's message with the system instruction
          response = chat_session.send_message(
              user_query,
              generation_config=generation_config
          )
          return response.text
      except Exception as e:
          st.error(f"An error occurred: {e}")
          return "Sorry, I seem to have misplaced my circuits. Could you try asking again?"
  
  
  # --- UI Layout ---
  st.title("🤖 AI Personality Chatbot")
  st.markdown("Customize my personality in the Python code and let's talk!")
  
  # Sidebar for configuration
  st.sidebar.header("Chat Controls")
  st.sidebar.info(f"**Current Personality:** {PERSONALITY_PROMPT}")
  
  if st.sidebar.button("Clear Chat History"):
      st.session_state.chat_history = []
      st.rerun()
  
  # --- Main Chat Interface ---
  chat_container = st.container()
  
  with chat_container:
      # Display existing chat messages
      for message in st.session_state.chat_history:
          is_user = message["role"] == "user"
          bubble_class = "user-bubble" if is_user else "bot-bubble"
          icon = "🧑‍💻" if is_user else "🤖"
  
          st.markdown(
              f'<div class="chat-bubble {bubble_class}">'
              f'  <span class="chat-icon">{icon}</span>'
              f'  <div>{message["content"]}</div>'
              f'</div>',
              unsafe_allow_html=True
          )
  
  # Chat input field
  prompt = st.chat_input("What's on your mind?")
  
  if prompt:
      # Add user message to history and display it
      st.session_state.chat_history.append({"role": "user", "content": prompt})
      with chat_container:
           st.markdown(
              f'<div class="chat-bubble user-bubble">'
              f'  <span class="chat-icon">🧑‍💻</span>'
              f'  <div>{prompt}</div>'
              f'</div>',
              unsafe_allow_html=True
          )
  
      # Get and display bot response
      with st.spinner("Thinking..."):
          response = get_gemini_response(prompt, st.session_state.chat_history)
          st.session_state.chat_history.append({"role": "bot", "content": response})
          # Rerun to display the new messages in the correct order
          st.rerun()
  
