import streamlit as st
import requests
import os
import pandas as pd

# ✅ Streamlit App Title
st.title("🛒 AI-Powered E-Commerce Chatbot (Free Version)")

# ✅ Check if the API key is available in secrets
if "HUGGINGFACE_API_KEY" in st.secrets:
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
else:
    st.error("🚨 Hugging Face API Key is missing! Add it in Streamlit Secrets.")
    st.stop()

# ✅ Load Product Catalog
json_path = "PRODUCT_catalog.json"  
if not os.path.exists(json_path):
    st.error(f"ERROR: The JSON file '{json_path}' is missing.")
    st.stop()
df_catalog = pd.read_json(json_path)

# ✅ Debug: Print loaded JSON data
st.write("✅ Loaded Product Catalog:", df_catalog.head())

# ✅ Hugging Face API Key (Set it in Streamlit Secrets)
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# ✅ Choose a Free Model (LLM)
MODEL_NAME = "tiiuae/falcon-7b-instruct"

# ✅ Function to Call Hugging Face API
def chat_with_bot(query):
    global df_catalog

    # ✅ Handle category-based queries
    if "category" in query.lower() or "categories" in query.lower():
        unique_categories = df_catalog["category"].unique()
        extracted_info = f"Available Categories: {', '.join(unique_categories)}"
    
    # ✅ Handle price-based queries
    elif "under" in query.lower() and any(char.isdigit() for char in query):
        price_limit = int(''.join(filter(str.isdigit, query)))
        search_results = df_catalog[df_catalog["price"] <= price_limit]
        extracted_info = search_results.to_string(index=False) if not search_results.empty else "No products found under this price range."
    
    # ✅ Default search (matches any column)
    else:
        search_results = df_catalog[df_catalog.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
        extracted_info = search_results.to_string(index=False) if not search_results.empty else "No matching products found in the catalog."

    # ✅ Format input for AI model
    input_text = f"User: {query} \n Product Info: {extracted_info} \n AI:"

    # ✅ Generate AI response
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        json={"inputs": input_text}
    )

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"AI Error: {response.json()}"

# ✅ Streamlit Chat Interface
st.subheader("Chat with your AI Assistant")
user_query = st.text_input("Ask me anything about our products:")

if st.button("Send"):
    if user_query:
        response = chat_with_bot(user_query)
        st.write(f"**AI:** {response}")

# ✅ Display Chat History
st.subheader("Chat History")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for msg in st.session_state.chat_history:
    st.write(msg)
