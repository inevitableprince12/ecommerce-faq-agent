from sentence_transformers import SentenceTransformer
import chromadb
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# DB
client = chromadb.Client()
collection = client.get_or_create_collection(name="ecommerce_kb")

# 👉 paste your FULL knowledge_base here

# embeddings setup
documents_text = [doc["text"] for doc in knowledge_base]
documents_ids = [doc["id"] for doc in knowledge_base]
documents_meta = [{"topic": doc["topic"]} for doc in knowledge_base]

embeddings = embedder.encode(documents_text).tolist()

if len(collection.get()["ids"]) == 0:
    collection.add(
        documents=documents_text,
        ids=documents_ids,
        metadatas=documents_meta,
        embeddings=embeddings
    )

# STATE
class CapstoneState(TypedDict):
    question: str
    retrieved_docs: List[str]
    retrieved_topics: List[str]
    sources: List[str]
    answer: str

# RETRIEVAL
def retrieval_node(state):
    q = state["question"]
    results = collection.query(query_texts=[q], n_results=3)

    state["retrieved_docs"] = results["documents"][0]
    state["retrieved_topics"] = [m["topic"] for m in results["metadatas"][0]]
    state["sources"] = results["ids"][0]
    return state

# ANSWER
def answer_node(state):
    docs = state["retrieved_docs"]
    state["answer"] = "Based on our policy:\n\n" + docs[0]
    return state

# GRAPH
graph = StateGraph(CapstoneState)
graph.add_node("retrieve", retrieval_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")

app_graph = graph.compile(checkpointer=MemorySaver())

# ASK
def ask(question):
    return app_graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": "user1"}}
    )