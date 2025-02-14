import streamlit as st
import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# ✅ Streamlit App Title
st.title("🛒 AI-Powered E-Commerce Chatbot")

# ✅ Load Product Catalog
json_path = "PRODUCT_catalog.json"  # Ensure the correct file path
if not os.path.exists(json_path):
    st.error(f"ERROR: The JSON file '{json_path}' is missing.")
    st.stop()
df_catalog = pd.read_json(json_path)

# ✅ Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    st.warning('CUDA is not available. Running on CPU.')

# ✅ Load Hugging Face Model (Use Streamlit Caching to Prevent Reloading)
@st.cache_resource()
def load_model():
    model_id = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300, temperature=0.2, do_sample=True)

pipe = load_model()
local_llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Streamlit Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chatbot Function
def chat_with_bot(query):
    global df_catalog

    # ✅ Step 1: Retrieve relevant information from JSON
    search_results = df_catalog[df_catalog.apply(lambda row: query.lower() in str(row).lower(), axis=1)]

    if search_results.empty:
        extracted_info = "I'm sorry, I couldn't find relevant information in the product catalog."
    else:
        extracted_info = search_results.to_string(index=False)  # Convert relevant data to string

    # ✅ Step 2: Format input for LLM
    formatted_history = " ".join(st.session_state.chat_history[-5:])  # Keep last 5 messages
    input_text = f"{formatted_history} User: {query} \n Product Info: {extracted_info} \n AI:"

    # ✅ Step 3: Generate response using LLM
    response = local_llm(input_text)

    if isinstance(response, str):  # Fix potential list error
        generated_text = response.strip()
    else:
        generated_text = response[0]["generated_text"].split("AI:")[-1].strip()

    # ✅ Step 4: Store conversation
    st.session_state.chat_history.append(f"User: {query}")
    st.session_state.chat_history.append(f"AI: {generated_text}")

    return generated_text

# ✅ Streamlit Chat Interface
st.subheader("Chat with your AI Assistant")
user_query = st.text_input("Ask me anything about our products:")

if st.button("Send"):
    if user_query:
        response = chat_with_bot(user_query)
        st.write(f"**AI:** {response}")

# ✅ Display Chat History
st.subheader("Chat History")
for msg in st.session_state.chat_history:
    st.write(msg)

