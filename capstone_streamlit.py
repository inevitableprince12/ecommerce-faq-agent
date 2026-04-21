import streamlit as st
from agent import ask

st.title("🛒 E-Commerce FAQ Bot")

user_input = st.text_input("Ask your question:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        result = ask(user_input)

    st.write("### 🤖 Answer")
    st.write(result["answer"])

    st.write("### 📚 Sources")
    st.write(result.get("sources", []))