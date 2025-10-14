import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()

# -------------------------
# Initialize Gemini with streaming
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
    streaming=True
)

# -------------------------
# Initialize Search Tool
# -------------------------
search_tool = DuckDuckGoSearchRun()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 Gemini Chatbot with Web Search (Streaming)")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    # Display user message
    st.chat_message("user").write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Check if we need to search
            search_keywords = ["latest", "current", "recent", "today", "now", "news", "weather"]
            should_search = any(keyword in user_input.lower() for keyword in search_keywords)
            
            # Perform search if needed
            search_results = ""
            if should_search:
                with st.spinner("🔍 Searching the web..."):
                    try:
                        search_results = search_tool.run(user_input)
                        search_results = f"\n\nSearch Results:\n{search_results[:1000]}"  # Limit results
                    except:
                        search_results = ""
            
            # Create prompt with or without search results
            if search_results:
                prompt = f"""Based on the following search results, answer the user's question accurately and concisely.

Search Results:
{search_results}

User Question: {user_input}

Answer:"""
            else:
                prompt = f"""Answer the following question accurately and helpfully:

Question: {user_input}

Answer:"""
            
            # Stream the response
            for chunk in llm.stream(prompt):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
            
            # Save to history
            st.session_state.history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.history.append({"role": "assistant", "content": error_msg})