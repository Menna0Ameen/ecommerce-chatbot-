import streamlit as st
import torch
import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# ✅ Streamlit App Title
st.title("🛒 AI-Powered E-Commerce Chatbot")

# ✅ Load Product Catalog
json_path = "PRODUCT_catalog.json"  # Ensure correct file path
if not os.path.exists(json_path):
    st.error(f"ERROR: The JSON file '{json_path}' is missing.")
    st.stop()
df_catalog = pd.read_json(json_path)

# ✅ Set up device (Streamlit Cloud does NOT have CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    st.warning('CUDA is not available. Running on CPU.')

# ✅ Load a Smaller AI Model Instead of `microsoft/phi-1_5`
@st.cache_resource()
def load_model():
    model_id = "google/flan-t5-small"  # Smaller and faster to load
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

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
