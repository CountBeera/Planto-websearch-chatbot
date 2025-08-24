import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True
)

# Search Tool
search_tool = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Use for searching the web for factual or current info."
    )
]

# Agent
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
    """Ask LLM to check if the answer is consistent, sufficient, and not contradictory.
       If not, regenerate/refine the answer."""
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

    # Parse reflection
    if reflection.startswith("INCONSISTENT:"):
        # Extract refined answer if present
        if "Refined Answer:" in reflection:
            refined_answer = reflection.split("Refined Answer:")[-1].strip()
            return refined_answer
        else:
            return answer + "\n\n(Note: Model flagged some inconsistency, but no refinement provided.)"
    else:
        return answer


# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 Gemini Chatbot with Web Search + Consistency Check")
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
            try:
                # Step 1: Generate answer
                raw_response = agent.run(user_input)

                # Step 2: Consistency check
                final_response = consistency_check(user_input, raw_response)

            except Exception as e:
                final_response = f"⚠️ Sorry, I had trouble parsing my response. Error: {str(e)}"

        st.write(final_response)
        st.session_state.history.append({"role": "assistant", "content": final_response})

    # with st.chat_message("assistant"):
    #     with st.spinner("Thinking..."):
    #         # Step 1: Generate answer
    #         raw_response = agent.run(user_input)

    #         # Step 2: Consistency check
    #         final_response = consistency_check(user_input, raw_response)

    #     st.write(final_response)
    #     st.session_state.history.append({"role": "assistant", "content": final_response})
