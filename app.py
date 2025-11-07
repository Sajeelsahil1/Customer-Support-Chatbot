import os
import re 
import streamlit as st 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# --- 1. SET UP GOOGLE API KEY ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop() 

# --- 2. DEFINE CONSTANTS ---
DATA_PATH = "./data"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- 3. CACHED FUNCTIONS TO LOAD MODELS ---
@st.cache_resource
def load_llm():
    # Keeping your "stunner" model
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

@st.cache_resource
def load_retriever():
    if not os.path.exists(DB_PATH):
        st.error("Chroma DB not found. Please run one of the v1-v4 scripts first to create it.")
        return None
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_db.as_retriever(search_kwargs={"k": 2})

LLM = load_llm()
RETRIEVER = load_retriever()

# --- 4. SIMULATED WEBHOOKS / API FUNCTIONS ---
# (Your working functions)
def track_order_api(order_id: str) -> str:
    """Simulates calling a CRM API to get order status."""
    print(f"\n[API CALL] Tracking order_id: {order_id}")
    if order_id == "12345":
        return "Status: Shipped. Expected delivery: Nov 10, 2025."
    elif order_id == "A-987":
        return "Status: Processing. Will ship within 2 business days."
    else:
        return f"Status: Order ID '{order_id}' not found."

def get_refund_status_api(order_id: str) -> str:
    """Simulates calling a payments API to get refund status."""
    print(f"\n[API CALL] Checking refund status for order_id: {order_id}")
    if order_id == "12345":
        return "Status: Refund approved. Will be processed in 3-5 business days."
    elif order_id == "A-987":
        return "Status: No refund request found for this order."
    else:
        return f"Status: Order ID '{order_id}' not found."

def escalate_to_human_api(chat_history: list) -> str:
    """Simulates escalating the chat to a human agent."""
    print("\n[API CALL] Escalating to human agent...")
    return "I understand. I am transferring you to a human agent now. They will have your chat history and will be with you shortly."

# --- 5. TOOL-SPECIFIC CHAINS ---
# (Your working chains)

# --- Tool 1: Order Tracking ---
class OrderInfo(BaseModel):
    order_id: str = Field(description="The user's order ID, e.g., '12345' or 'A-987'")

def create_order_tracking_chain():
    structured_llm = LLM.with_structured_output(OrderInfo)
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting order IDs. The user's message is below. Extract the order_id from it."),
        ("human", "{input}")
    ])
    order_chain = (
        extraction_prompt
        | structured_llm
        | RunnableLambda(lambda x: track_order_api(x.order_id))
    )
    return order_chain

# --- Tool 2: Refund Status ---
def create_refund_status_chain():
    structured_llm = LLM.with_structured_output(OrderInfo)
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting order IDs for refund requests. The user's message is below. Extract the order_id from it."),
        ("human", "{input}")
    ])
    refund_chain = (
        extraction_prompt
        | structured_llm
        | RunnableLambda(lambda x: get_refund_status_api(x.order_id))
    )
    return refund_chain

# --- Tool 3: FAQ Conversational Chain ---
# (Your working `RunnablePassthrough.assign` version)
def create_faq_chain(retriever):
    rephrasing_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    rephraser_chain = rephrasing_prompt | LLM | StrOutputParser()
    
    document_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    answer_chain = document_answer_prompt | LLM | StrOutputParser()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval_chain = rephraser_chain | retriever | format_docs
    conversational_rag_chain = RunnablePassthrough.assign(
        context=lambda x: retrieval_chain.invoke(x)
    ) | answer_chain
    
    return conversational_rag_chain

# --- Tool 4: Human Handover Chain ---
def create_human_handover_chain():
    return RunnableLambda(lambda x: escalate_to_human_api(x['chat_history']))

# --- 6. THE ROUTER: The "Brain" of the Bot ---
# (Your working, strict router)
def create_router_chain(faq_chain, order_chain, refund_chain, handover_chain):
    """Creates the main router chain that decides which tool to use."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a strict and expert router. You must classify the user's input into one of the following categories.
You must output *only* the category name and nothing else.

- 'FAQ': Use this for general questions, greetings, simple comments, how-to questions (like "how do I track my order"), refund policies, shipping info, etc. **This is the default category.**
- 'ORDER_TRACKING': Use this *only* if the user is asking for the status of a *specific* order and provides an order ID.
- 'REFUND_STATUS': Use this *only* if the user is asking for the status of a *specific* refund and provides an order ID.
- 'HUMAN_HANDOVER': Use this *only* if the user is frustrated, angry, or *explicitly* asks to speak to a human, a person, or an agent. Do not use this for simple greetings or neutral comments.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    router_chain = prompt | LLM | StrOutputParser()
    
    def route(info):
        topic = info['topic']
        if "ORDER_TRACKING" in topic:
            return order_chain
        if "REFUND_STATUS" in topic:
            return refund_chain
        if "HUMAN_HANDOVER" in topic:
            return handover_chain
        else: # Default to FAQ
            return faq_chain

    main_chain = {
        "topic": router_chain,
        "input": lambda x: x['input'], 
        "chat_history": lambda x: x['chat_history']
    } | RunnableLambda(lambda x: route(x).invoke(x)) 

    return main_chain


# --- 7. CREATE THE FINAL CHATBOT ---
@st.cache_resource
def get_chatbot():
    if RETRIEVER is None:
        return None
        
    faq_chain = create_faq_chain(RETRIEVER)
    order_chain = create_order_tracking_chain()
    refund_chain = create_refund_status_chain()
    handover_chain = create_human_handover_chain()
    
    return create_router_chain(
        faq_chain, order_chain, refund_chain, handover_chain
    )

full_chatbot = get_chatbot()

# --- 8. STREAMLIT UI LOGIC ---
# --- THIS IS THE ONLY SECTION THAT HAS CHANGED ---

st.title("🚀 Customer Support Chatbot")
st.caption("I can answer FAQs, track orders, check refunds, and escalate to a human.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! How can I help you today?")
    ]

# Display past messages
for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# --- NEW: Refactored function to handle input ---
def handle_user_input(prompt_text):
    """
    Handles both typed and button-clicked user input.
    """
    if not full_chatbot:
        st.error("Chatbot is not initialized. Please ensure the Chroma DB exists.")
        return

    # Add user message to UI and history
    st.chat_message("user").write(prompt_text)
    st.session_state.chat_history.append(HumanMessage(content=prompt_text))
    
    # Get bot response
    try:
        response = full_chatbot.invoke({
            "chat_history": st.session_state.chat_history,
            "input": prompt_text
        })
        
        # Add bot response to UI and history
        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append(AIMessage(content=response))
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    # Limit history size
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

# --- NEW: Display suggested prompts as buttons ---
st.divider()
st.caption("Or try a suggested prompt:")

suggested_prompts = [
    "What's the refund policy?",
    "Where is order 12345?",
    "Can I talk to an agent?"
]

cols = st.columns(len(suggested_prompts))
for i, suggestion in enumerate(suggested_prompts):
    if cols[i].button(suggestion, use_container_width=True):
        handle_user_input(suggestion)
        st.rerun() # Rerun the script to show the new messages immediately

# --- UPDATED: Get new user input from the text box ---
if prompt := st.chat_input("What's on your mind?"):
    handle_user_input(prompt) # Call the same handler function
    st.rerun() # Force a rerun to maintain state consistency