import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain.prompts import PromptTemplate
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# Load API Keys
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()
if not TAVILY_API_KEY:
    st.error("Please set TAVILY_API_KEY in .env file")
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

# Non-streaming LLM for query decomposition
llm_non_streaming = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
    streaming=False
)

# -------------------------
# Initialize Tavily Client
# -------------------------
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# -------------------------
# Intelligent Search Decision Function
# -------------------------
def should_perform_search(user_query: str) -> tuple[bool, str]:
    """Use LLM to intelligently decide if web search is needed"""
    decision_prompt = f"""Analyze if the following user query requires current web search or can be answered from general knowledge.

User Query: "{user_query}"

Consider:
1. Does this require real-time/current information? (news, weather, events, prices, schedules)
2. Does this need specific factual data that changes? (business info, travel details, product specs)
3. Is this about recent events or developments?
4. Would search provide significantly better/more accurate information?

Return ONLY a JSON object with this exact format:
{{"needs_search": true/false, "reasoning": "brief explanation"}}

Examples:
- "What is Python?" → {{"needs_search": false, "reasoning": "General programming concept, well-established knowledge"}}
- "Plan a trip to Pondicherry" → {{"needs_search": true, "reasoning": "Requires current travel info, attractions, and recommendations"}}
- "Latest news on AI" → {{"needs_search": true, "reasoning": "Explicitly asks for current/latest information"}}
- "TCS NQT exam details" → {{"needs_search": true, "reasoning": "Specific exam info that may have recent updates"}}
- "Explain quantum physics" → {{"needs_search": false, "reasoning": "Established scientific concept"}}

Return only the JSON, nothing else:"""
    
    try:
        response = llm_non_streaming.invoke(decision_prompt)
        response_text = response.content.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        decision = json.loads(response_text)
        
        return decision.get("needs_search", False), decision.get("reasoning", "No reasoning provided")
    except Exception as e:
        # Default to search on error to be safe
        return True, f"Error in decision making: {e}. Defaulting to search for safety."
# -------------------------
# Query Decomposition Function
# -------------------------
def decompose_query(user_query: str) -> list:
    """Use LLM to break down complex query into sub-queries"""
    decomposition_prompt = f"""Analyze the following user query and determine if it needs to be broken down into multiple search queries.

User Query: "{user_query}"

Instructions:
1. If the query is simple and focused, return it as a single query
2. If the query is complex or asks multiple things, break it down into 2-5 focused sub-queries
3. Each sub-query should be specific and searchable
4. Return ONLY a JSON array of queries, nothing else

Examples:
- Simple: "What is the weather in Paris?" → ["What is the weather in Paris?"]
- Complex: "Compare the economies of Japan and Germany in 2024" → ["Japan economy 2024", "Germany economy 2024"]
- Complex: "Latest news on AI and climate change" → ["latest AI news 2024", "latest climate change news 2024"]

Return your response as a JSON array only:"""
    
    try:
        response = llm_non_streaming.invoke(decomposition_prompt)
        response_text = response.content.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        queries = json.loads(response_text)
        
        # Validate it's a list
        if not isinstance(queries, list) or len(queries) == 0:
            return [user_query]
        
        return queries[:5]  # Limit to max 5 queries
    except Exception as e:
        st.warning(f"Query decomposition failed: {e}. Using original query.")
        return [user_query]

# -------------------------
# Batch Search Function
# -------------------------
def batch_search_tavily(queries: list) -> dict:
    """Perform batch search using Tavily for multiple queries"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {
            executor.submit(tavily_client.search, query, max_results=3): query 
            for query in queries
        }
        
        for future in future_to_query:
            query = future_to_query[future]
            try:
                search_result = future.result()
                results[query] = search_result
            except Exception as e:
                results[query] = {"error": str(e)}
    
    return results

# -------------------------
# Format Search Results
# -------------------------
def format_batch_results(batch_results: dict) -> str:
    """Format batch search results for LLM context"""
    formatted = "\n\n=== SEARCH RESULTS ===\n"
    
    for query, result in batch_results.items():
        formatted += f"\n--- Query: {query} ---\n"
        
        if "error" in result:
            formatted += f"Error: {result['error']}\n"
            continue
        
        if "results" in result:
            for i, item in enumerate(result["results"][:3], 1):
                formatted += f"\n{i}. {item.get('title', 'No title')}\n"
                formatted += f"   URL: {item.get('url', 'N/A')}\n"
                formatted += f"   {item.get('content', 'No content')[:300]}...\n"
        else:
            formatted += "No results found.\n"
    
    return formatted

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 Gemini Chatbot with Tavily Batch Search")
st.caption("Powered by intelligent query decomposition and parallel search")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Show thinking process for past messages
        if msg["role"] == "assistant" and "thinking" in msg:
            with st.expander("🧠 View AI Thinking Process"):
                st.markdown(msg["thinking"], unsafe_allow_html=True)

user_input = st.chat_input("Type your question...")

if user_input:
    # Display user message
    st.chat_message("user").write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        # Create expandable sections for the thinking process
        thinking_container = st.container()
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # LLM-based search decision
            with thinking_container:
                with st.expander("🧠 AI Thinking Process", expanded=True):
                    st.markdown("**Step 1: Evaluating Query**")
                    st.write(f"📝 Query: `{user_input}`")
                    
                    decision_status = st.empty()
                    decision_status.info("🤔 Analyzing if web search is needed...")
                    
                    should_search, reasoning = should_perform_search(user_input)
                    
                    if should_search:
                        decision_status.success(f"✅ **Decision: Search Required**")
                        st.write(f"💡 Reasoning: {reasoning}")
                    else:
                        decision_status.info(f"ℹ️ **Decision: No Search Needed**")
                        st.write(f"💡 Reasoning: {reasoning}")
            
            # Perform batch search if needed
            search_context = ""
            queries_used = []
            batch_results = {}
            
            if should_search:
                with thinking_container:
                    with st.expander("🧠 AI Thinking Process", expanded=True):
                        st.markdown("**Step 2: Query Decomposition**")
                        
                        analysis_status = st.empty()
                        analysis_status.info("⚙️ Breaking down query into searchable components...")
                        
                        queries = decompose_query(user_input)
                        queries_used = queries
                        
                        analysis_status.success("✅ Query decomposition complete")
                        
                        st.markdown("**Step 2: Query Decomposition Results**")
                        if len(queries) == 1:
                            st.write("💡 Query is focused - using single search")
                            st.code(queries[0], language=None)
                        else:
                            st.write(f"💡 Complex query detected - split into {len(queries)} sub-queries:")
                            for i, q in enumerate(queries, 1):
                                st.code(f"{i}. {q}", language=None)
                        
                        st.markdown("**Step 3: Parallel Search Execution**")
                        search_status = st.empty()
                        search_status.info(f"🌐 Executing {len(queries)} searches in parallel via Tavily...")
                        
                        batch_results = batch_search_tavily(queries)
                        search_context = format_batch_results(batch_results)
                        
                        search_status.success(f"✅ All searches completed")
                        
                        st.markdown("**Step 4: Search Results Summary**")
                        results_summary = st.container()
                        with results_summary:
                            for query, result in batch_results.items():
                                with st.container():
                                    st.markdown(f"**Query:** `{query}`")
                                    if "error" in result:
                                        st.error(f"❌ Error: {result['error']}")
                                    elif "results" in result and len(result["results"]) > 0:
                                        st.success(f"✅ Found {len(result['results'])} results")
                                        for i, item in enumerate(result["results"][:2], 1):
                                            st.markdown(f"  {i}. [{item.get('title', 'No title')}]({item.get('url', '#')})")
                                    else:
                                        st.warning("⚠️ No results found")
                        
                        st.markdown("**Step 5: Synthesizing Response**")
                        st.info("🤖 Generating comprehensive answer from search results...")
            
            # Show raw context data if available
            if search_context:
                with thinking_container:
                    with st.expander("📊 Raw Search Data (Context for LLM)", expanded=False):
                        st.text(search_context[:2000] + "..." if len(search_context) > 2000 else search_context)
            
            # Create prompt with or without search results
            if search_context:
                prompt = f"""You are a helpful AI assistant. Based on the search results below, provide a comprehensive and accurate answer to the user's question.

{search_context}

User Question: {user_input}

Instructions:
- Synthesize information from all search results
- Provide a clear, well-structured answer
- Cite sources when making specific claims
- If results are contradictory, mention it
- Be concise but thorough

Answer:"""
            else:
                with thinking_container:
                    with st.expander("🧠 AI Thinking Process", expanded=True):
                        st.markdown("**Decision: No Web Search Needed**")
                        st.write("💡 Query can be answered from existing knowledge")
                        st.write("🤖 Generating response directly...")
                
                prompt = f"""You are a helpful AI assistant. Answer the following question accurately and helpfully:

Question: {user_input}

Answer:"""
            
            # Add a separator before the actual response
            st.markdown("---")
            st.markdown("### 💬 Response:")
            
            # Stream the response with proper placeholder
            response_container = st.container()
            with response_container:
                response_placeholder = st.empty()
                for chunk in llm.stream(prompt):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                
                # Final response without cursor
                response_placeholder.markdown(full_response)
            
            # Save to history with thinking process
            history_entry = {"role": "assistant", "content": full_response}
            
            # Capture thinking process for history
            if queries_used:
                thinking_html = f"""
                **Query Decomposition:**
                """
                for i, q in enumerate(queries_used, 1):
                    thinking_html += f"\n- `{q}`"
                
                thinking_html += "\n\n**Search Results:**\n"
                for query, result in batch_results.items():
                    thinking_html += f"\n**{query}**\n"
                    if "results" in result:
                        thinking_html += f"- Found {len(result['results'])} results\n"
                
                history_entry["thinking"] = thinking_html
                history_entry["queries"] = queries_used
            
            st.session_state.history.append(history_entry)
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.history.append({"role": "assistant", "content": error_msg})

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot uses:
    - **Tavily Search API** for web searches
    - **Batch Processing** for parallel queries
    - **LLM Query Decomposition** to break down complex questions
    - **Gemini 2.0 Flash** for responses
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()