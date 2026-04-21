# E-Commerce FAQ Agent (RAG + LangGraph)

## Description
AI-powered FAQ assistant for e-commerce using Retrieval-Augmented Generation.

## Features
- RAG with ChromaDB
- LangGraph workflow
- Memory support
- Streamlit UI

## Architecture
User Query → Embedding → ChromaDB Retrieval → LangGraph Workflow → Answer Generation → Streamlit UI

## Sample Query
Q: How can I cancel my order?
A: You may cancel an order before it is dispatched without any charges...

## Tech Stack
- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- LangGraph

## Run Locally

```bash
pip install -r requirements.txt
streamlit run capstone_streamlit.py

