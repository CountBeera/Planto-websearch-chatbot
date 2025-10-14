import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.base import BaseCallbackHandler

# -------------------------
# Streaming Handler
# -------------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        """Display new tokens as they stream in."""
        self.text += token
        self.container.markdown(self.text)

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()

# -------------------------
# Initialize Gemini
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
    streaming=True  # 👈 Enable streaming
)

# -------------------------
# Search Tool + Agent
# -------------------------
search_tool = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Use for searching the web for factual or current info."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

# -------------------------
# Consistency Check Function
# -------------------------
def consistency_check(question: str, answer: str) -> str:
    """Ask LLM to check if the answer is consistent and sufficient."""
    check_prompt = f"""
    You are tasked with evaluating the following answer for consistency and sufficiency.

    Question: {question}
    Answer: {answer}

    1. Check if the answer is contradictory, factually insufficient, or unclear.
    2. If the answer is good, respond with: "CONSISTENT: <short justification>".
    3. If the answer has issues, respond with: "INCONSISTENT: <short justification>. Refined Answer: <better answer>".
    """
    reflection = llm.invoke(check_prompt).content
    print("\n--- Consistency Check Reflection ---")
    print(reflection)
    print("------------------------------------\n")

    if reflection.startswith("INCONSISTENT:") and "Refined Answer:" in reflection:
        return reflection.split("Refined Answer:")[-1].strip()
    return answer

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 Gemini Chatbot with Web Search + Consistency Check (Streaming)")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Stream container
            stream_container = st.empty()
            stream_handler = StreamHandler(stream_container)

            try:
                # Step 1: Generate streamed response
                raw_response = agent.run(user_input, callbacks=[stream_handler])

                # Step 2: Consistency check (non-streaming)
                final_response = consistency_check(user_input, raw_response)

            except Exception as e:
                final_response = f"⚠️ Sorry, I had trouble parsing my response. Error: {str(e)}"

        # Final render (replace partial stream with checked text)
        st.markdown(final_response)
        st.session_state.history.append({"role": "assistant", "content": final_response})
