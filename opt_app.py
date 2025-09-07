import os
import time  # Import the time module
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

# --- 1. Configuration and Setup ---
def load_api_key():
    """Loads the Google API key from Streamlit secrets or .env file."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
    return api_key

GOOGLE_API_KEY = load_api_key()
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file or Streamlit secrets.")
    st.stop()

# --- 2. Caching LLM and Tools Initialization ---
@st.cache_resource
def get_llm():
    """Initializes and returns the Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )

@st.cache_resource
def get_tools():
    """Initializes and returns the tools for the agent."""
    search_tool = DuckDuckGoSearchRun()
    return [
        Tool(
            name="WebSearch",
            func=search_tool.run,
            description="Use for searching the web for factual or current info."
        )
    ]

# --- 3. Modern Agent Setup (LCEL) ---
@st.cache_resource
def create_agent_executor():
    """Creates the LangChain agent using modern LCEL approach."""
    llm = get_llm()
    tools = get_tools()
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

llm = get_llm()
agent_executor = create_agent_executor()

# --- 4. Caching for Core Logic Functions with Timing ---
@st.cache_data(show_spinner=False)
def get_initial_answer(question: str) -> tuple[str, float]:
    """Invokes the agent and returns the answer and the execution time."""
    print(f"\n>> AGENT: Getting initial answer for: '{question}' (CACHE MISS)")
    start_time = time.perf_counter()
    try:
        response = agent_executor.invoke({"input": question})
        answer = response.get("output", "Sorry, I couldn't find an answer.")
    except Exception as e:
        answer = f"⚠️ Agent Error: {str(e)}"
    end_time = time.perf_counter()
    duration = end_time - start_time
    return answer, duration

@st.cache_data(show_spinner=False)
def run_consistency_check(_llm, question: str, answer: str) -> tuple[str, float]:
    """Runs consistency check and returns the final answer and execution time."""
    print(f"\n>> LLM: Running consistency check for: '{question}' (CACHE MISS)")
    start_time = time.perf_counter()
    
    check_prompt = f"""
    You are an expert evaluator. Your task is to check the provided Answer for consistency, sufficiency, and clarity based on the Question.

    Question: {question}
    Answer: {answer}

    Critique the answer.
    1. Is it contradictory?
    2. Is it factually insufficient or incomplete?
    3. Is it unclear or poorly structured?

    Based on your critique, decide on a final response:
    - If the answer is good, respond ONLY with: CONSISTENT: The answer directly and sufficiently addresses the question without contradictions.
    - If the answer has issues, respond ONLY with: INCONSISTENT: <Provide a brief, one-sentence justification>. Refined Answer: <Rewrite the answer to be better, clearer, and more complete>.
    """

    reflection = _llm.invoke(check_prompt).content
    print("\n--- Consistency Check Reflection ---")
    print(reflection)
    print("------------------------------------\n")

    if reflection.startswith("INCONSISTENT:"):
        if "Refined Answer:" in reflection:
            final_answer = reflection.split("Refined Answer:")[-1].strip()
        else:
            final_answer = answer + "\n\n*(Self-correction note: The initial answer was flagged for inconsistency but a refined version was not provided.)*"
    else:
        final_answer = answer
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    return final_answer, duration

# --- 5. Streamlit UI ---
st.title("⏱️ Gemini Chatbot with Performance Metrics")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])
    # Also render the performance stats if they exist
    if msg["role"] == "assistant" and "performance" in msg:
        with st.expander("Show Performance Stats"):
            st.json(msg["performance"])


if user_input := st.chat_input("Type your question..."):
    st.chat_message("user").write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            total_start_time = time.perf_counter()

            # Step 1: Get initial answer and its duration
            initial_answer, agent_duration = get_initial_answer(user_input)

            # Step 2: Run consistency check and its duration
            final_response, check_duration = run_consistency_check(llm, user_input, initial_answer)
            
            total_end_time = time.perf_counter()
            total_duration = total_end_time - total_start_time
        
        # Display the final response
        st.write(final_response)

        # Create a dictionary for performance stats
        performance_stats = {
            "Agent Execution Time": f"{agent_duration:.2f}s",
            "Consistency Check Time": f"{check_duration:.2f}s",
            "Total Response Time": f"{total_duration:.2f}s",
            "Cached Agent Response": "Yes" if agent_duration < 0.01 else "No",
            "Cached Consistency Check": "Yes" if check_duration < 0.01 else "No",
        }
        
        # Display performance stats in an expander
        with st.expander("Show Performance Stats"):
            st.json(performance_stats)
            
        # Append to history with performance data
        st.session_state.history.append({
            "role": "assistant",
            "content": final_response,
            "performance": performance_stats
        })